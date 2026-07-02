//! gRPC service implementation for the worker

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use async_stream::try_stream;
use futures::{Stream, StreamExt};
use dashmap::DashMap;
use tokio::sync::RwLock;
use tokio::time::timeout;
use tonic::{Request, Response, Status};
use tracing::{info, warn, error, debug, instrument};
use uuid::Uuid;

use crate::cluster::*;
use crate::cluster::worker_server::Worker;
use crate::gpu_manager::GPUManager;
use crate::model_loader::ModelLoader;
use crate::models::ModelInstance;
use crate::config::WorkerConfig;
use crate::metrics::Metrics;

/// Worker service implementation
#[derive(Clone)]
pub struct WorkerService {
    /// Worker ID
    pub worker_id: String,

    /// GPU manager
    gpu_manager: Arc<GPUManager>,

    /// Model loader
    model_loader: Arc<ModelLoader>,

    /// Loaded models (model_name -> ModelInstance)
    loaded_models: Arc<RwLock<HashMap<String, ModelInstance>>>,

    /// Active inference requests (lock-free concurrent map)
    active_requests: Arc<DashMap<String, Instant>>,

    /// Service start time (for uptime reporting)
    start_time: Instant,

    /// Configuration
    config: WorkerConfig,

    /// Metrics
    metrics: Metrics,

    /// Bounds concurrent inference requests (RESOURCE_EXHAUSTED beyond this).
    infer_semaphore: Arc<tokio::sync::Semaphore>,
}

impl WorkerService {
    /// Create a new worker service
    pub fn new(
        worker_id: String,
        gpu_manager: Arc<GPUManager>,
        model_loader: Arc<ModelLoader>,
        config: WorkerConfig,
    ) -> Self {
        let infer_semaphore = Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_requests));
        Self {
            worker_id,
            gpu_manager,
            model_loader,
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            active_requests: Arc::new(DashMap::new()),
            start_time: Instant::now(),
            config,
            metrics: Metrics::new(),
            infer_semaphore,
        }
    }

    /// Get worker version
    pub fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }


}

/// Removes the request from the active map when dropped — even if the client
/// disconnects mid-stream and the response stream is dropped.
struct ActiveGuard {
    map: Arc<DashMap<String, Instant>>,
    id: String,
    metrics: Metrics,
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.map.remove(&self.id);
        self.metrics.set_active_requests(self.map.len());
    }
}

#[tonic::async_trait]
impl Worker for WorkerService {
    type InferStream = Pin<Box<dyn Stream<Item = Result<InferenceResponse, Status>> + Send>>;

    #[instrument(skip(self))]
    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = request.into_inner();
        info!("Loading model: {}", req.model_name);

        // Check if already loaded — report the REAL instance data, not zeros.
        {
            let models = self.loaded_models.read().await;
            if let Some(instance) = models.get(&req.model_name) {
                return Ok(Response::new(LoadModelResponse {
                    success: true,
                    message: "Model already loaded".to_string(),
                    memory_used: instance.memory_used() as u64,
                    loaded_on_gpus: instance.gpu_ids().iter().map(|&id| id as i32).collect(),
                }));
            }
        }

        // Validate GPU IDs — reject out-of-range ids from remote input up front.
        let device_count = self.gpu_manager.device_count();
        for &id in &req.gpu_ids {
            if id < 0 || (id as usize) >= device_count {
                return Err(Status::invalid_argument(format!(
                    "gpu_id {} out of range: this worker manages {} device(s)",
                    id, device_count
                )));
            }
        }
        let gpu_ids: Vec<u32> = if req.gpu_ids.is_empty() {
            (0..device_count as u32).collect()
        } else {
            req.gpu_ids.iter().map(|&id| id as u32).collect()
        };

        // Load model
        let load_start = Instant::now();
        let repo_override = if req.model_path.is_empty() {
            None
        } else {
            Some(req.model_path.as_str())
        };
        let result = self.model_loader.load_model(
            &req.model_name,
            repo_override,
            req.config.as_ref(),
            &gpu_ids,
            req.quantization(),
            req.parallelism(),
        ).await;

        match result {
            Ok(model_instance) => {
                let load_time = load_start.elapsed();
                let memory_used = model_instance.memory_used();

                // Store model
                self.loaded_models.write().await.insert(
                    req.model_name.clone(),
                    model_instance,
                );

                // Update metrics
                self.metrics.record_model_load(&req.model_name, load_time);
                self.metrics.set_model_memory(&req.model_name, memory_used as i64);
                self.metrics.set_loaded_models(self.loaded_models.read().await.len());

                info!(
                    "Model {} loaded successfully in {:?}, using {}MB VRAM",
                    req.model_name, load_time, memory_used / 1024 / 1024
                );

                Ok(Response::new(LoadModelResponse {
                    success: true,
                    message: "Model loaded successfully".to_string(),
                    memory_used: memory_used as u64,
                    loaded_on_gpus: gpu_ids.iter().map(|&id| id as i32).collect(),
                }))
            }
            Err(e) => {
                error!("Failed to load model {}: {}", req.model_name, e);
                self.metrics.record_error("model_load");
                Err(Status::internal(format!("Failed to load model: {}", e)))
            }
        }
    }

    #[instrument(skip(self))]
    async fn infer(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<Self::InferStream>, Status> {
        let req = request.into_inner();
        let request_id = if req.request_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            req.request_id.clone()
        };

        info!(
            "Inference request {}: model={}, prompt_len={}",
            request_id, req.model_name, req.prompt.len()
        );

        // Concurrency limit — reject instead of queueing unboundedly.
        let permit = match self.infer_semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                return Err(Status::resource_exhausted(format!(
                    "worker at max_concurrent_requests={}",
                    self.config.max_concurrent_requests
                )));
            }
        };

        // Track active request
        self.active_requests.insert(request_id.clone(), Instant::now());
        self.metrics.set_active_requests(self.active_requests.len());
        let active_guard = ActiveGuard {
            map: self.active_requests.clone(),
            id: request_id.clone(),
            metrics: self.metrics.clone(),
        };

        // Get model
        debug!("Inference request {}: waiting for loaded_models read lock", request_id);
        let model = {
            let models = self.loaded_models.read().await;
            models.get(&req.model_name).cloned()
        };
        debug!("Inference request {}: released loaded_models read lock (found: {})", request_id, model.is_some());

        let model = match model {
            Some(m) => m,
            None => {
                return Err(crate::error::WorkerError::ModelNotFound(req.model_name.clone()).into());
            }
        };

        // Apply timeout if configured
        let timeout_duration = std::time::Duration::from_secs(self.config.request_timeout_secs);

        let metrics = self.metrics.clone();
        let model_name = req.model_name.clone();
        let req_id = request_id.clone();

        // Create response stream
        let stream = try_stream! {
            let _permit = permit;           // released when the stream is dropped/finished
            let _active_guard = active_guard; // removes active_requests entry on drop
            let start_time = Instant::now();
            let mut tokens_generated: u32 = 0;

            // proto: seed == 0 means "random"
            let seed = if req.seed == 0 { None } else { Some(req.seed as u64) };
            // Run inference
            let inference_result = timeout(
                timeout_duration,
                model.generate(
                    &req.prompt,
                    req.max_tokens as usize,
                    req.temperature,
                    req.top_p,
                    req.top_k as usize,
                    seed,
                )
            ).await;

            match inference_result {
                Ok(Ok(mut token_stream)) => {
                    let deadline = tokio::time::Instant::now() + timeout_duration;
                    let mut stream_error = false;
                    let mut timed_out = false;

                    loop {
                        match tokio::time::timeout_at(deadline, token_stream.next()).await {
                            Err(_) => {
                                timed_out = true;
                                metrics.record_error("timeout");
                                break;
                            }
                            Ok(None) => break,
                            Ok(Some(Ok(text))) => {
                                tokens_generated += 1;
                                yield InferenceResponse {
                                    request_id: req_id.clone(),
                                    text,
                                    tokens_generated,
                                    finished: false,
                                    finish_reason: 0,
                                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                                };
                            }
                            Ok(Some(Err(e))) => {
                                tracing::error!("Generation error: {}", e);
                                stream_error = true;
                                metrics.record_error("inference");
                                break;
                            }
                        }
                    }

                    let finish_reason = if timed_out {
                        tracing::warn!("Request {} timed out after {:?}", req_id, timeout_duration);
                        FinishReason::Timeout
                    } else if stream_error {
                        FinishReason::Error
                    } else if tokens_generated >= req.max_tokens {
                        FinishReason::Length
                    } else {
                        FinishReason::Stop
                    };

                    // Send final response
                    yield InferenceResponse {
                        request_id: req_id.clone(),
                        text: String::new(),
                        tokens_generated,
                        finished: true,
                        finish_reason: finish_reason as i32,
                        processing_time_ms: start_time.elapsed().as_millis() as u64,
                    };

                    // Record metrics
                    let elapsed = start_time.elapsed();
                    metrics.record_inference(
                        &model_name,
                        elapsed,
                        tokens_generated as usize,
                    );

                    tracing::info!(
                        "Request {} completed ({:?}): {} tokens in {:?}",
                        req_id, finish_reason, tokens_generated, elapsed
                    );
                }
                Ok(Err(e)) => {
                    tracing::error!("Inference error for {}: {}", req_id, e);
                    metrics.record_error("inference");
                    yield InferenceResponse {
                        request_id: req_id.clone(),
                        text: format!("Error: {}", e),
                        tokens_generated,
                        finished: true,
                        finish_reason: FinishReason::Error as i32,
                        processing_time_ms: start_time.elapsed().as_millis() as u64,
                    };
                }
                Err(_) => {
                    tracing::warn!("Request {} timed out after {:?}", req_id, timeout_duration);
                    metrics.record_error("timeout");
                    yield InferenceResponse {
                        request_id: req_id.clone(),
                        text: String::new(),
                        tokens_generated,
                        finished: true,
                        finish_reason: FinishReason::Timeout as i32,
                        processing_time_ms: timeout_duration.as_millis() as u64,
                    };
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    #[instrument(skip(self))]
    async fn get_status(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<WorkerStatus>, Status> {
        debug!("Status request received - waiting for locks");

        // Get GPU info
        let gpu_infos = self.gpu_manager.get_all_gpu_info().await;
        debug!("Status: acquired GPU info");

        // Get loaded models info
        let loaded_models = {
            let models = self.loaded_models.read().await;
            models.iter().map(|(name, instance)| {
                LoadedModelInfo {
                    model_name: name.clone(),
                    memory_used: instance.memory_used() as u64,
                    gpu_ids: instance.gpu_ids().iter().map(|&id| id as i32).collect(),
                    quantization: instance.quantization(),
                    parallelism: instance.parallelism(),
                    loaded_at_timestamp: instance.loaded_at().timestamp() as u64,
                    num_inferences: instance.inference_count(),
                }
            }).collect()
        };

        // Get system info
        let active_requests = self.active_requests.len();
        debug!("Status: waiting for system memory info");
        let (memory_available, memory_total) = self.gpu_manager.system_memory().await;
        debug!("Status: all info collected");

        Ok(Response::new(WorkerStatus {
            worker_id: self.worker_id.clone(),
            version: self.version().to_string(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            gpus: gpu_infos,
            loaded_models,
            cpu_utilization: 0.0,
            memory_available,
            memory_total,
            active_requests: active_requests as u32,
            queued_requests: 0,
        }))
    }

    #[instrument(skip(self))]
    async fn unload_model(
        &self,
        request: Request<UnloadModelRequest>,
    ) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        info!("Unloading model: {}", req.model_name);

        let removed_from_service = {
            let mut models = self.loaded_models.write().await;
            models.remove(&req.model_name).is_some()
        };
        // The loader's DashMap holds the other Arc clone AND owns the GPU reservations.
        let removed_from_loader = self.model_loader.unload(&req.model_name).await;

        if removed_from_service || removed_from_loader {
            self.metrics.remove_model_metrics(&req.model_name);
            self.metrics.set_loaded_models(self.loaded_models.read().await.len());
            info!("Model {} unloaded successfully", req.model_name);
            Ok(Response::new(Empty {}))
        } else {
            warn!("Model {} not found for unloading", req.model_name);
            Err(Status::not_found(format!("Model {} not found", req.model_name)))
        }
    }

    #[instrument(skip(self))]
    async fn health_check(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let is_healthy = self.gpu_manager.is_healthy().await;
        let status = if is_healthy {
            health_check_response::ServingStatus::Serving
        } else {
            health_check_response::ServingStatus::NotServing
        };

        Ok(Response::new(HealthCheckResponse {
            status: status as i32,
            message: format!("Worker {} is {}", self.worker_id,
                if is_healthy { "healthy" } else { "unhealthy" }
            ),
        }))
    }
}
