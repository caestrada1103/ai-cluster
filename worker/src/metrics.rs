//! Metrics collection and exposition
//!
//! This module provides Prometheus metrics for monitoring worker performance,
//! GPU utilization, request statistics, and more.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::State,
    response::IntoResponse,
    routing::get,
    Router,
};
use metrics::{
    describe_counter, describe_gauge, describe_histogram,
    counter, gauge, histogram,
};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use tokio::sync::RwLock;
use tracing::{info, error};

use crate::gpu_manager::GPUManager;
use crate::worker::WorkerService;

/// Metrics registry and server
pub struct Metrics {
    /// Prometheus recorder handle
    recorder_handle: Option<PrometheusHandle>,
    
    /// HTTP server address
    server_addr: Option<SocketAddr>,
    
    /// Server task handle
    server_task: Option<tokio::task::JoinHandle<()>>,
}

impl Metrics {
    /// Create new metrics registry
    pub fn new() -> Self {
        // Describe metrics
        describe_counter!(
            "worker_requests_total",
            "Total number of inference requests processed"
        );
        
        describe_gauge!(
            "worker_active_requests",
            "Number of currently active requests"
        );
        
        describe_histogram!(
            "worker_request_duration_seconds",
            "Request duration in seconds"
        );
        
        describe_gauge!(
            "worker_gpu_utilization_percent",
            "GPU utilization percentage"
        );
        
        describe_gauge!(
            "worker_gpu_memory_used_bytes",
            "GPU memory used in bytes"
        );
        
        describe_gauge!(
            "worker_gpu_temperature_celsius",
            "GPU temperature in Celsius"
        );
        
        describe_gauge!(
            "worker_gpu_power_watts",
            "GPU power usage in watts"
        );
        
        describe_counter!(
            "worker_tokens_generated_total",
            "Total number of tokens generated"
        );
        
        describe_histogram!(
            "worker_batch_size",
            "Batch size distribution"
        );
        
        describe_gauge!(
            "worker_loaded_models",
            "Number of loaded models"
        );
        
        describe_gauge!(
            "worker_model_memory_bytes",
            "Memory used by each model",
            labels: HashMap::from([("model".to_string(), String::new())])
        );
        
        describe_counter!(
            "worker_errors_total",
            "Total number of errors",
            labels: HashMap::from([("type".to_string(), String::new())])
        );
        
        Self {
            recorder_handle: None,
            server_addr: None,
            server_task: None,
        }
    }
    
    /// Start metrics server
    pub async fn start_server(
        &mut self,
        port: u16,
        gpu_manager: Arc<GPUManager>,
        worker_service: Arc<WorkerService>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Build Prometheus recorder
        let builder = PrometheusBuilder::new()
            .with_http_listener(([0, 0, 0, 0], port))
            .add_global_label("worker_id", worker_service.worker_id.clone())?;
        
        let (recorder, exporter) = builder.build()?;
        self.recorder_handle = Some(exporter.handle());
        
        // Install recorder
        metrics::set_global_recorder(recorder)?;
        
        // Create app state
        #[derive(Clone)]
        struct AppState {
            gpu_manager: Arc<GPUManager>,
            worker_service: Arc<WorkerService>,
        }
        
        let state = AppState {
            gpu_manager,
            worker_service,
        };
        
        // Create router
        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .with_state(state);
        
        // Start server
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let listener = tokio::net::TcpListener::bind(addr).await?;
        self.server_addr = Some(listener.local_addr()?);
        
        info!("Metrics server listening on {}", self.server_addr.unwrap());
        
        // Run server
        self.server_task = Some(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        }));
        
        Ok(())
    }
    
    /// Stop metrics server
    pub async fn stop_server(&mut self) {
        if let Some(task) = self.server_task.take() {
            task.abort();
            let _ = task.await;
        }
    }
    
    /// Record inference request
    pub fn record_inference(&self, model_name: &str, duration: Duration, tokens: usize) {
        counter!("worker_requests_total", 1);
        histogram!("worker_request_duration_seconds", duration.as_secs_f64());
        counter!("worker_tokens_generated_total", tokens as u64);
        
        // Record per-model metrics
        counter!(
            "worker_model_requests_total",
            1,
            "model" => model_name.to_string()
        );
    }
    
    /// Record model load
    pub fn record_model_load(&self, model_name: &str, duration: Duration) {
        histogram!(
            "worker_model_load_duration_seconds",
            duration.as_secs_f64(),
            "model" => model_name.to_string()
        );
    }
    
    /// Update GPU metrics
    pub fn update_gpu_metrics(&self, gpu_id: usize, utilization: f32, memory_used: u64, temperature: f32, power: u32) {
        gauge!("worker_gpu_utilization_percent", utilization as f64, "gpu" => gpu_id.to_string());
        gauge!("worker_gpu_memory_used_bytes", memory_used as f64, "gpu" => gpu_id.to_string());
        gauge!("worker_gpu_temperature_celsius", temperature as f64, "gpu" => gpu_id.to_string());
        gauge!("worker_gpu_power_watts", power as f64, "gpu" => gpu_id.to_string());
    }
    
    /// Set active requests
    pub fn set_active_requests(&self, count: i64) {
        gauge!("worker_active_requests", count as f64);
    }
    
    /// Set loaded models count
    pub fn set_loaded_models(&self, count: i64) {
        gauge!("worker_loaded_models", count as f64);
    }
    
    /// Set model memory usage
    pub fn set_model_memory(&self, model_name: &str, memory_bytes: i64) {
        gauge!(
            "worker_model_memory_bytes",
            memory_bytes as f64,
            "model" => model_name.to_string()
        );
    }
    
    /// Remove model metrics
    pub fn remove_model_metrics(&self, model_name: &str) {
        // Prometheus doesn't support removing metrics directly
        // We can set them to 0 as a workaround
        gauge!(
            "worker_model_memory_bytes",
            0.0,
            "model" => model_name.to_string()
        );
    }
    
    /// Record batch size
    pub fn record_batch_size(&self, size: usize) {
        histogram!("worker_batch_size", size as f64);
    }
    
    /// Record error
    pub fn record_error(&self, error_type: &str) {
        counter!("worker_errors_total", 1, "type" => error_type.to_string());
    }
    
    /// Record queue size
    pub fn set_queue_size(&self, size: i64) {
        gauge!("worker_queue_size", size as f64);
    }
}

/// Metrics HTTP handler
async fn metrics_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Update metrics before serving
    let gpu_manager = &state.gpu_manager;
    let worker_service = &state.worker_service;
    
    // Update GPU metrics
    for gpu_info in gpu_manager.get_all_gpu_info().await {
        metrics::gauge!("worker_gpu_utilization_percent", gpu_info.utilization as f64, "gpu" => gpu_info.id.to_string());
        metrics::gauge!("worker_gpu_memory_used_bytes", (gpu_info.total_memory - gpu_info.available_memory) as f64, "gpu" => gpu_info.id.to_string());
        metrics::gauge!("worker_gpu_temperature_celsius", gpu_info.temperature as f64, "gpu" => gpu_info.id.to_string());
        metrics::gauge!("worker_gpu_power_watts", gpu_info.power_usage as f64, "gpu" => gpu_info.id.to_string());
    }
    
    // Update request metrics
    metrics::gauge!("worker_active_requests", worker_service.active_request_count().await as f64);
    metrics::gauge!("worker_loaded_models", worker_service.loaded_models().await.len() as f64);
    
    // Get metrics from recorder
    if let Some(recorder) = &metrics::try_recorder() {
        // Convert to Prometheus format
        // This is simplified - actual implementation would use the exporter
        recorder.render()
    } else {
        "Metrics not available".to_string()
    }
}

/// Custom metrics recorder for detailed GPU stats
pub struct GPUMetricsCollector {
    gpu_manager: Arc<GPUManager>,
}

impl GPUMetricsCollector {
    pub fn new(gpu_manager: Arc<GPUManager>) -> Self {
        Self { gpu_manager }
    }
    
    pub async fn collect(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        for gpu_info in self.gpu_manager.get_all_gpu_info().await {
            metrics.insert(
                format!("gpu_{}_utilization", gpu_info.id),
                gpu_info.utilization as f64
            );
            metrics.insert(
                format!("gpu_{}_memory_used_gb", gpu_info.id),
                (gpu_info.total_memory - gpu_info.available_memory) as f64 / 1e9
            );
            metrics.insert(
                format!("gpu_{}_temperature", gpu_info.id),
                gpu_info.temperature as f64
            );
            metrics.insert(
                format!("gpu_{}_power_watts", gpu_info.id),
                gpu_info.power_usage as f64
            );
        }
        
        metrics
    }
}

/// Health check endpoint for Kubernetes
pub async fn health_check() -> impl IntoResponse {
    "OK"
}

/// Readiness probe endpoint
pub async fn readiness_check(
    State(worker_service): State<Arc<WorkerService>>,
) -> impl IntoResponse {
    if worker_service.is_healthy().await {
        "READY"
    } else {
        (axum::http::StatusCode::SERVICE_UNAVAILABLE, "NOT READY")
    }
}

/// Liveness probe endpoint
pub async fn liveness_check() -> impl IntoResponse {
    "ALIVE"
}

#[derive(Clone)]
struct AppState {
    gpu_manager: Arc<GPUManager>,
    worker_service: Arc<WorkerService>,
}

/// Metrics server
pub struct MetricsServer {
    port: u16,
    gpu_manager: Arc<GPUManager>,
    worker_service: Arc<WorkerService>,
}

impl MetricsServer {
    /// Create new metrics server
    pub fn new(
        port: u16,
        gpu_manager: Arc<GPUManager>,
        worker_service: Arc<WorkerService>,
    ) -> Self {
        Self {
            port,
            gpu_manager,
            worker_service,
        }
    }
    
    /// Run the metrics server
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let state = Arc::new(AppState {
            gpu_manager: self.gpu_manager.clone(),
            worker_service: self.worker_service.clone(),
        });
        
        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_check))
            .route("/ready", get(readiness_check))
            .route("/live", get(liveness_check))
            .with_state(state);
        
        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));
        info!("Metrics server listening on {}", addr);
        
        axum::serve(
            tokio::net::TcpListener::bind(addr).await?,
            app
        ).await?;
        
        Ok(())
    }
}