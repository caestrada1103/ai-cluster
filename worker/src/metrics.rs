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
use tracing::{info, error};

use crate::gpu_manager::GPUManager;
use crate::worker::WorkerService;

/// Metrics registry and server
#[derive(Clone)]
pub struct Metrics {
    _private: (),
}

impl Metrics {
    /// Create new metrics registry
    pub fn new() -> Self {
        // Describe metrics (safe to call multiple times — metrics crate deduplicates)
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
            "Memory used by each model"
        );

        describe_counter!(
            "worker_errors_total",
            "Total number of errors"
        );

        Self { _private: () }
    }

    /// Record inference request
    pub fn record_inference(&self, model_name: &str, duration: Duration, tokens: usize) {
        counter!("worker_requests_total", 1);
        histogram!("worker_request_duration_seconds", duration.as_secs_f64());
        counter!("worker_tokens_generated_total", tokens as u64);

        // Record per-model metrics
        counter!(
            "worker_model_requests_total", 1,
            "model" => model_name.to_string()
        );
    }

    /// Record model load
    pub fn record_model_load(&self, model_name: &str, duration: Duration) {
        histogram!(
            "worker_model_load_duration_seconds", duration.as_secs_f64(),
            "model" => model_name.to_string()
        );
    }

    /// Update GPU metrics
    pub fn update_gpu_metrics(&self, gpu_id: usize, utilization: f32, memory_used: u64, temperature: f32, power: u32) {
        let gpu_label = gpu_id.to_string();
        gauge!("worker_gpu_utilization_percent", utilization as f64, "gpu" => gpu_label.clone());
        gauge!("worker_gpu_memory_used_bytes", memory_used as f64, "gpu" => gpu_label.clone());
        gauge!("worker_gpu_temperature_celsius", temperature as f64, "gpu" => gpu_label.clone());
        gauge!("worker_gpu_power_watts", power as f64, "gpu" => gpu_label);
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
            "worker_model_memory_bytes", memory_bytes as f64,
            "model" => model_name.to_string()
        );
    }

    /// Remove model metrics
    pub fn remove_model_metrics(&self, model_name: &str) {
        gauge!(
            "worker_model_memory_bytes", 0.0,
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
    State(state): State<Arc<MetricsAppState>>,
) -> impl IntoResponse {
    // Update GPU metrics before serving
    let gpu_manager = &state.gpu_manager;

    for gpu_info in gpu_manager.get_all_gpu_info().await {
        let gpu_label = gpu_info.id.to_string();
        gauge!("worker_gpu_utilization_percent", gpu_info.utilization as f64, "gpu" => gpu_label.clone());
        gauge!("worker_gpu_memory_used_bytes", (gpu_info.total_memory - gpu_info.available_memory) as f64, "gpu" => gpu_label.clone());
        gauge!("worker_gpu_temperature_celsius", gpu_info.temperature as f64, "gpu" => gpu_label.clone());
        gauge!("worker_gpu_power_watts", gpu_info.power_usage as f64, "gpu" => gpu_label);
    }

    // Render from the handle
    match &state.prometheus_handle {
        Some(handle) => handle.render(),
        None => "# Metrics not available\n".to_string(),
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
                gpu_info.utilization as f64,
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
        "NOT READY"
    }
}

/// Liveness probe endpoint
pub async fn liveness_check() -> impl IntoResponse {
    "ALIVE"
}

#[derive(Clone)]
struct MetricsAppState {
    gpu_manager: Arc<GPUManager>,
    worker_service: Arc<WorkerService>,
    prometheus_handle: Option<Arc<PrometheusHandle>>,
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
        // Build Prometheus recorder and exporter
        let builder = PrometheusBuilder::new();
        let handle = builder.install_recorder()?;

        let state = Arc::new(MetricsAppState {
            gpu_manager: self.gpu_manager.clone(),
            worker_service: self.worker_service.clone(),
            prometheus_handle: Some(Arc::new(handle)),
        });

        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_check))
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