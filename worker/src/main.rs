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
use tracing::{info, warn, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub mod backend;
mod config;
mod error;
mod gpu_manager;
mod metrics;
mod model_loader;
#[path = "../models/mod.rs"]
mod models;
mod parallelism;
mod worker;

/// Generated protobuf code
pub mod cluster {
    tonic::include_proto!("cluster");
}

use crate::cluster::worker_server::WorkerServer;
use crate::config::WorkerConfig;
use crate::error::WorkerError;
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

    /// gRPC server port
    #[clap(short, long, default_value = "50051", env = "GRPC_PORT")]
    port: u16,

    /// Metrics server port
    #[clap(long, default_value = "9091", env = "METRICS_PORT")]
    metrics_port: u16,

    /// GPU IDs to use (comma-separated, e.g., "0,1,2")
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

    // Parse GPU IDs
    let gpu_ids = args.gpu_ids
        .as_ref()
        .map(|s| {
            s.split(',')
                .map(|id| id.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(|e| WorkerError::Configuration(format!("Invalid GPU IDs: {}", e)))?
        .unwrap_or_else(|| vec![0]);

    info!("Using GPUs: {:?}", gpu_ids);

    // Create tokio runtime
    let runtime = create_runtime()?;

    // Run the worker
    runtime.block_on(async_main(args, config, gpu_ids))
}

fn init_logging(args: &Args) {
    let base_filter = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| args.log_level.clone());

    // Aggressively silence driver-level noise from internal dependencies.
    // Use 'off' for crates that continue to leak INFO logs despite 'error' filters.
    let filter_str = format!("{},wgpu=warn,wgpu_hal=off,naga=off,vulkan=off,vulkan_layer=off", base_filter);

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

async fn async_main(args: Args, config: WorkerConfig, gpu_ids: Vec<usize>) -> Result<(), WorkerError> {
    // Initialize GPU manager
    info!("Initializing GPU Manager...");
    let gpu_manager = Arc::new(gpu_manager::GPUManager::new(&config.gpu_ids).await?);
    
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
    info!("Initialized GPU manager with {} devices", gpu_manager.device_count());

    // Initialize model loader
    let loader_config = ModelLoaderConfig {
        cache_dir: config.model_cache_dir.clone(),
        download_dir: config.download_dir.clone(),
        max_concurrent_loads: config.max_concurrent_loads,
        load_timeout_secs: config.load_timeout_secs,
        verify_checksums: config.verify_checksums,
        enable_mmap: config.enable_mmap,
        pin_memory: config.pin_memory,
        prefetch_size_gb: 2.0,
    };
    let model_loader = Arc::new(ModelLoader::new(loader_config, gpu_manager.clone())?);

    // Create worker service
    let worker_service = WorkerService::new(
        args.worker_id.unwrap_or_else(|| format!("worker-{}", gpu_ids[0])),
        gpu_manager.clone(),
        model_loader,
        config,
    );

    // Start metrics server
    let metrics_server = MetricsServer::new(
        args.metrics_port,
        gpu_manager.clone(),
        Arc::new(worker_service.clone()),
    );
    tokio::spawn(async move {
        if let Err(e) = metrics_server.run().await {
            error!("Metrics server error: {}", e);
        }
    });
    info!("Metrics server listening on port {}", args.metrics_port);

    // Build gRPC server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    info!("gRPC server listening on {}", addr);

    // Health service
    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter.set_serving::<WorkerServer<WorkerService>>().await;

    Server::builder()
        .add_service(health_service)
        .add_service(WorkerServer::new(worker_service))
        .serve(addr)
        .await
        .map_err(|e| WorkerError::Grpc(format!("Server error: {}", e)))?;

    Ok(())
}