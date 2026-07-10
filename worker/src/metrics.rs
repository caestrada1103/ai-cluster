//! Metrics collection and exposition
//!
//! This module provides Prometheus metrics for monitoring worker performance,
//! GPU utilization, request statistics, and more.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{extract::State, response::IntoResponse, routing::get, Router};
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::PrometheusHandle;
use tracing::info;

use crate::gpu_manager::GPUManager;

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

        describe_gauge!("worker_gpu_memory_used_bytes", "GPU memory used in bytes");

        describe_gauge!(
            "worker_gpu_temperature_celsius",
            "GPU temperature in Celsius"
        );

        describe_gauge!("worker_gpu_power_watts", "GPU power usage in watts");

        describe_counter!(
            "worker_tokens_generated_total",
            "Total number of tokens generated"
        );

        describe_gauge!("worker_loaded_models", "Number of loaded models");

        describe_gauge!("worker_model_memory_bytes", "Memory used by each model");

        describe_counter!("worker_errors_total", "Total number of errors");

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

    /// Gauge: number of in-flight inference requests.
    pub fn set_active_requests(&self, n: usize) {
        gauge!("worker_active_requests", n as f64);
    }

    /// Gauge: number of models currently loaded.
    pub fn set_loaded_models(&self, n: usize) {
        gauge!("worker_loaded_models", n as f64);
    }

    /// Counter: errors by kind ("inference", "timeout", "model_load").
    pub fn record_error(&self, kind: &str) {
        counter!("worker_errors_total", 1, "kind" => kind.to_string());
    }
}

/// Metrics HTTP handler
async fn metrics_handler(State(state): State<Arc<MetricsAppState>>) -> impl IntoResponse {
    // Update GPU metrics before serving
    let gpu_manager = &state.gpu_manager;
    gpu_manager.refresh_telemetry().await;

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

/// Health check endpoint for Kubernetes
pub async fn health_check() -> impl IntoResponse {
    "OK"
}

/// Liveness probe endpoint
pub async fn liveness_check() -> impl IntoResponse {
    "ALIVE"
}

#[derive(Clone)]
struct MetricsAppState {
    gpu_manager: Arc<GPUManager>,
    prometheus_handle: Option<Arc<PrometheusHandle>>,
}

/// Metrics server
pub struct MetricsServer {
    port: u16,
    gpu_manager: Arc<GPUManager>,
    handle: PrometheusHandle,
}

impl MetricsServer {
    /// Create new metrics server. The recorder handle is installed by main()
    /// before any service starts, so early describe!/counter! calls register.
    pub fn new(port: u16, gpu_manager: Arc<GPUManager>, handle: PrometheusHandle) -> Self {
        Self {
            port,
            gpu_manager,
            handle,
        }
    }

    /// Run the metrics server
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let handle = self.handle.clone();

        let state = Arc::new(MetricsAppState {
            gpu_manager: self.gpu_manager.clone(),
            prometheus_handle: Some(Arc::new(handle)),
        });

        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_check))
            .route("/live", get(liveness_check))
            .with_state(state);

        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));
        info!("Metrics server listening on {}", addr);

        axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;

        Ok(())
    }
}
