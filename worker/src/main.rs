//! AI Worker - High-performance inference worker for distributed AI cluster
//!
//! This worker runs on each GPU node and handles:
//! - Model loading/unloading
//! - Inference execution
//! - Multi-GPU parallelism
//! - Metrics collection

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use tokio::runtime::Runtime;
use tonic::transport::Server;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Burn backend selection (wgpu / CUDA / ROCm / ndarray).
pub mod backend;
mod config;
mod error;
mod gpu_manager;
mod grpc_auth;
#[cfg(feature = "llamacpp")]
mod llamacpp_engine;
mod llamaserver_process;
mod metrics;
mod model_loader;
#[path = "../models/mod.rs"]
mod models;
mod parallelism;
mod worker;

/// Generated protobuf code
#[allow(missing_docs)]
pub mod cluster {
    tonic::include_proto!("cluster");
}

use crate::cluster::worker_server::WorkerServer;
use crate::config::WorkerConfig;
use crate::error::WorkerError;
use crate::grpc_auth::TokenInterceptor;
use crate::metrics::MetricsServer;
use crate::model_loader::{ModelLoader, ModelLoaderConfig};
use crate::worker::WorkerService;

/// Command line arguments
#[derive(Parser, Debug)]
#[clap(author, version, about = "AI Inference Worker")]
struct Args {
    /// Worker ID (auto-generated if not provided)
    #[clap(long, env = "WORKER_ID")]
    worker_id: Option<String>,

    /// gRPC server port (falls back to config file, then 50051)
    #[clap(short, long, env = "GRPC_PORT")]
    port: Option<u16>,

    /// Metrics server port (falls back to config file, then 9091)
    #[clap(long, env = "METRICS_PORT")]
    metrics_port: Option<u16>,

    /// GPU IDs to use (comma-separated, e.g., "0,1,2"; falls back to config file)
    #[clap(long, env = "GPU_IDS")]
    gpu_ids: Option<String>,

    /// Path to config file
    #[clap(short, long, default_value = "config/worker.toml", env = "CONFIG_FILE")]
    config: String,

    /// Log level (debug, info, warn, error)
    #[clap(long, default_value = "info", env = "LOG_LEVEL")]
    log_level: String,

    /// Enable JSON logging
    #[clap(long, env = "LOG_JSON")]
    log_json: bool,
}

fn main() -> Result<(), WorkerError> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logging
    init_logging(&args);

    // Load configuration
    let config = WorkerConfig::from_file(&args.config)?;

    info!("Starting AI Worker v{}", env!("CARGO_PKG_VERSION"));
    info!("Configuration: {:?}", config);

    // Parse GPU IDs — CLI/env wins, then config file, then [0]
    let gpu_ids = args
        .gpu_ids
        .as_ref()
        .map(|s| {
            s.split(',')
                .map(|id| id.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(|e| WorkerError::Configuration(format!("Invalid GPU IDs: {}", e)))?
        .unwrap_or_else(|| config.gpu_ids.clone());

    info!("Using GPUs: {:?}", gpu_ids);

    // Create tokio runtime
    let runtime = create_runtime()?;

    // Run the worker
    runtime.block_on(async_main(args, config, gpu_ids))
}

fn init_logging(args: &Args) {
    let base_filter = std::env::var("RUST_LOG").unwrap_or_else(|_| args.log_level.clone());

    // Aggressively silence driver-level noise from internal dependencies.
    // Use 'off' for crates that continue to leak INFO logs despite 'error' filters.
    let filter_str = format!(
        "{},wgpu=warn,wgpu_hal=off,naga=off,vulkan=off,vulkan_layer=off",
        base_filter
    );

    let env_filter = EnvFilter::new(filter_str);

    if args.log_json {
        // JSON logging for production
        let json_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_file(true)
            .with_line_number(true);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(json_layer)
            .init();
    } else {
        // Pretty logging for development
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    }
}

fn create_runtime() -> Result<Runtime, WorkerError> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("ai-worker")
        .build()
        .map_err(|e| WorkerError::Runtime(format!("Failed to create runtime: {}", e)))
}

async fn async_main(
    args: Args,
    config: WorkerConfig,
    gpu_ids: Vec<usize>,
) -> Result<(), WorkerError> {
    // Install the Prometheus recorder BEFORE any metric is described or recorded.
    // If this fails the process must not run blind — bail out.
    let prometheus_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| WorkerError::Runtime(format!("Failed to install metrics recorder: {}", e)))?;

    // Initialize GPU manager
    info!("Initializing GPU Manager...");
    let gpu_manager = Arc::new(gpu_manager::GPUManager::new(&gpu_ids).await?);

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::WgpuDevice;
        // WgpuDevice::default() picks the best available adapter automatically:
        //   Windows  → DX12 (prefers discrete GPU, e.g. RTX 3050)
        //   Linux    → Vulkan (requires a hardware Vulkan ICD from the driver)
        //   macOS    → Metal
        // NOTE: in Docker Desktop on Windows (WSL2), NVIDIA's driver only exposes CUDA—
        // no Vulkan ICD is injected—so wgpu falls back to Mesa llvmpipe (CPU).
        // On a native Linux host with NVIDIA Container Toolkit + graphics capability,
        // the NVIDIA Vulkan ICD is injected and the real GPU is selected.
        let device = WgpuDevice::default();
        info!("Selected WGPU Device: {:?}", device);
    }

    info!("GPU Manager initialized successfully");
    info!(
        "Initialized GPU manager with {} devices",
        gpu_manager.device_count()
    );

    // Initialize model loader
    let loader_config = ModelLoaderConfig {
        cache_dir: config.model_cache_dir.clone(),
        download_dir: config.download_dir.clone(),
        max_concurrent_loads: config.max_concurrent_loads,
        hf_token: config.hf_token.clone(),
        hf_cache_dir: config.hf_cache_dir.clone(),
        llamacpp_n_threads: config.llamacpp_n_threads,
        llamacpp_default_n_gpu_layers: config.llamacpp_default_n_gpu_layers,
        // env LLAMASERVER_BINARY_PATH wins over the config file (mirrors HF_TOKEN).
        llamaserver_binary_path: std::env::var("LLAMASERVER_BINARY_PATH")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| config.llamaserver_binary_path.clone()),
        llamaserver_bind_host: config.llamaserver_bind_host.clone(),
        // Reuse the worker's single load/inference timeout for /health polling.
        llamaserver_health_timeout_secs: config.request_timeout_secs,
        max_loaded_models: config.max_loaded_models,
        max_n_ctx: config.max_n_ctx,
        max_concurrent_requests: config.max_concurrent_requests,
        llamaserver_enable_slots_endpoint: config.llamaserver_enable_slots_endpoint,
        llamaserver_port_min: config.llamaserver_port_min,
        llamaserver_port_max: config.llamaserver_port_max,
    };
    let model_loader = Arc::new(ModelLoader::new(loader_config, gpu_manager.clone())?);

    // Effective values: CLI/env > config file > default
    let grpc_port = args.port.unwrap_or(config.grpc_port);
    let metrics_port = args.metrics_port.unwrap_or(config.metrics_port);
    let worker_id = args
        .worker_id
        .or_else(|| config.worker_id.clone())
        .unwrap_or_else(|| format!("worker-{}", gpu_ids[0]));

    // C1: bind interface is config-driven and defaults to loopback-only
    // (WorkerConfig::default's grpc_bind_host = "127.0.0.1") — a
    // non-loopback bind is an explicit opt-in (worker.toml/env), mirroring
    // how gpu_ids/ports already resolve. Computed BEFORE `config` moves into
    // `WorkerService::new` below.
    let bind_ip: std::net::IpAddr = config.grpc_bind_host.parse().map_err(|e| {
        WorkerError::Configuration(format!(
            "invalid grpc_bind_host '{}': {e}",
            config.grpc_bind_host
        ))
    })?;
    let addr = SocketAddr::from((bind_ip, grpc_port));

    // C1: shared-secret auth, gated on config. Env WORKER_GRPC_AUTH_TOKEN
    // wins over worker.toml's grpc_auth_token, mirroring HF_TOKEN/
    // LLAMASERVER_BINARY_PATH. `None`/empty leaves the server OPEN — the
    // existing behavior for single-host loopback-only deployments.
    let grpc_auth_token = std::env::var("WORKER_GRPC_AUTH_TOKEN")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| config.grpc_auth_token.clone());

    // Create worker service
    let worker_service = WorkerService::new(worker_id, gpu_manager.clone(), model_loader, config);

    // Start metrics server
    let metrics_server = MetricsServer::new(metrics_port, gpu_manager.clone(), prometheus_handle);
    tokio::spawn(async move {
        if let Err(e) = metrics_server.run().await {
            error!("Metrics server error: {}", e);
        }
    });
    info!("Metrics server listening on port {}", metrics_port);
    if grpc_auth_token.is_none() && !bind_ip.is_loopback() {
        warn!(
            "gRPC server binding to non-loopback address {} with NO grpc_auth_token/\
             WORKER_GRPC_AUTH_TOKEN configured — every gRPC RPC (LoadModel, Infer, ...) \
             is reachable by anyone who can route to this address. Set \
             WORKER_GRPC_AUTH_TOKEN (or worker.toml's grpc_auth_token) before exposing \
             this port beyond a container-internal network you already trust.",
            addr
        );
    }
    info!("gRPC server listening on {}", addr);

    // Health service
    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<WorkerServer<WorkerService>>()
        .await;

    let interceptor = TokenInterceptor::new(grpc_auth_token);

    Server::builder()
        // The plain gRPC health-check service is intentionally left
        // unauthenticated — orchestrators/load balancers/`grpc_health_probe`
        // need to reach it without a credential, and it exposes no model
        // data or control-plane actions.
        .add_service(health_service)
        .add_service(WorkerServer::with_interceptor(worker_service, interceptor))
        .serve(addr)
        .await
        .map_err(|e| WorkerError::Grpc(format!("Server error: {}", e)))?;

    Ok(())
}
