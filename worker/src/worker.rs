//! gRPC service implementation for the worker

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use async_stream::try_stream;
use futures::{Stream, StreamExt};
use tokio::sync::{Mutex, RwLock};
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

    /// Active inference requests
    active_requests: Arc<Mutex<HashMap<String, Instant>>>,

    /// Configuration
    config: WorkerConfig,

    /// Metrics
    metrics: Metrics,
}

impl WorkerService {
    /// Create a new worker service
    pub fn new(
        worker_id: String,
        gpu_manager: Arc<GPUManager>,
        model_loader: Arc<ModelLoader>,
        config: WorkerConfig,
    ) -> Self {
        Self {
            worker_id,
            gpu_manager,
            model_loader,
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            config,
            metrics: Metrics::new(),
        }
    }

    /// Get worker version
    pub fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// Get number of active requests
    pub async fn active_request_count(&self) -> usize {
        self.active_requests.lock().await.len()
    }

    /// Get loaded model names
    pub async fn loaded_models(&self) -> Vec<String> {
        let models = self.loaded_models.read().await;
        models.keys().cloned().collect()
    }

    /// Check if worker is healthy
    pub async fn is_healthy(&self) -> bool {
        self.gpu_manager.is_healthy().await
    }

    /// Update metrics from loaded models
    pub async fn update_metrics(&self) {
        let models = self.loaded_models.read().await;
        self.metrics.set_loaded_models(models.len() as i64);

        for (name, model) in models.iter() {
            self.metrics.set_model_memory(
                name,
                model.memory_used() as i64,
            );
        }
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

        // Check if already loaded
        {
            let models = self.loaded_models.read().await;
            if models.contains_key(&req.model_name) {
                return Ok(Response::new(LoadModelResponse {
                    success: true,
                    message: "Model already loaded".to_string(),
                    memory_used: 0,
                    loaded_on_gpus: vec![],
                }));
            }
        }

        // Validate GPU IDs
        let gpu_ids: Vec<u32> = if req.gpu_ids.is_empty() {
            (0..self.gpu_manager.device_count() as u32).collect()
        } else {
            req.gpu_ids.iter().map(|&id| id as u32).collect()
        };

        // Load model
        let load_start = Instant::now();
        let result = self.model_loader.load_model(
            &req.model_name,
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

        // Track active request
        self.active_requests.lock().await.insert(request_id.clone(), Instant::now());

        // Get model
        let model = {
            let models = self.loaded_models.read().await;
            models.get(&req.model_name).cloned()
        };

        let model = match model {
            Some(m) => m,
            None => {
                self.active_requests.lock().await.remove(&request_id);
                return Err(Status::not_found(format!("Model {} not loaded", req.model_name)));
            }
        };

        // Apply timeout if configured
        let timeout_duration = std::time::Duration::from_secs(self.config.request_timeout_secs);

        let metrics = self.metrics.clone();
        let active_requests = self.active_requests.clone();
        let model_name = req.model_name.clone();
        let req_id = request_id.clone();

        // Create response stream
        let stream = try_stream! {
            let start_time = Instant::now();
            let mut tokens_generated: u32 = 0;

            // Run inference
            let inference_result = timeout(
                timeout_duration,
                model.generate(
                    &req.prompt,
                    req.max_tokens as usize,
                    req.temperature,
                    req.top_p,
                    req.top_k as usize,
                )
            ).await;

            match inference_result {
                Ok(Ok(mut token_stream)) => {
                    // Stream tokens as they're generated
                    while let Some(token) = token_stream.next().await {
                        match token {
                            Ok(text) => {
                                tokens_generated += 1;

                                // Send chunk
                                yield InferenceResponse {
                                    request_id: req_id.clone(),
                                    text,
                                    tokens_generated,
                                    finished: false,
                                    finish_reason: 0,
                                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                                };
                            }
                            Err(e) => {
                                tracing::error!("Generation error: {}", e);
                                break;
                            }
                        }
                    }

                    // Send final response
                    yield InferenceResponse {
                        request_id: req_id.clone(),
                        text: String::new(),
                        tokens_generated,
                        finished: true,
                        finish_reason: FinishReason::Stop as i32,
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
                        "Request {} completed: {} tokens in {:?}",
                        req_id, tokens_generated, elapsed
                    );
                }
                Ok(Err(e)) => {
                    tracing::error!("Inference error for {}: {}", req_id, e);
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

            // Clean up
            active_requests.lock().await.remove(&req_id);
        };

        Ok(Response::new(Box::pin(stream)))
    }

    #[instrument(skip(self))]
    async fn get_status(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<WorkerStatus>, Status> {
        debug!("Status request received");

        // Get GPU info
        let gpu_infos = self.gpu_manager.get_all_gpu_info().await;

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
        let active_requests = self.active_requests.lock().await.len();
        let (memory_available, memory_total) = self.gpu_manager.system_memory().await;

        Ok(Response::new(WorkerStatus {
            worker_id: self.worker_id.clone(),
            version: self.version().to_string(),
            uptime_seconds: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            gpus: gpu_infos,
            loaded_models,
            cpu_utilization: 0.0,
            memory_available: memory_available as u64,
            memory_total: memory_total as u64,
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

        let mut models = self.loaded_models.write().await;

        if let Some(model) = models.remove(&req.model_name) {
            // Drop model to free GPU memory
            drop(model);

            // Update metrics
            self.metrics.remove_model_metrics(&req.model_name);

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
