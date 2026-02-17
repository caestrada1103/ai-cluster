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
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod config;
mod error;
mod gpu_manager;
mod metrics;
mod model_loader;
mod models;
mod parallelism;
mod worker;

use crate::config::WorkerConfig;
use crate::error::WorkerError;
use crate::metrics::MetricsServer;
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
        .unwrap_or_else(|| (0..num_cpus::get()).collect());
    
    info!("Using GPUs: {:?}", gpu_ids);
    
    // Create tokio runtime
    let runtime = create_runtime()?;
    
    // Run the worker
    runtime.block_on(async_main(args, config, gpu_ids))
}

fn init_logging(args: &Args) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&args.log_level));
    
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
        .worker_threads(num_cpus::get())
        .max_blocking_threads(num_cpus::get())
        .build()
        .map_err(|e| WorkerError::Runtime(format!("Failed to create runtime: {}", e)))
}

async fn async_main(args: Args, config: WorkerConfig, gpu_ids: Vec<usize>) -> Result<(), WorkerError> {
    // Initialize GPU manager
    let gpu_manager = Arc::new(gpu_manager::GPUManager::new(&gpu_ids).await?);
    info!("Initialized GPU manager with {} devices", gpu_manager.device_count());
    
    // Initialize model loader
    let model_loader = Arc::new(model_loader::ModelLoader::new(
        config.model_cache_dir.clone(),
        gpu_manager.clone(),
    )?);
    
    // Create worker service
    let worker_service = WorkerService::new(
        args.worker_id.unwrap_or_else(|| format!("worker-{}", gpu_ids[0])),
        gpu_manager.clone(),
        model_loader.clone(),
        config,
    );
    
    // Start metrics server
    let metrics_server = MetricsServer::new(
        args.metrics_port,
        gpu_manager.clone(),
        worker_service.clone(),
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
    
    let reflection = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(tonic::include_file_descriptor_set!("cluster_descriptor"))
        .build()
        .map_err(|e| WorkerError::Grpc(format!("Failed to build reflection: {}", e)))?;
    
    let health_service = tonic_health::server::health_reporter();
    
    Server::builder()
        .add_service(reflection)
        .add_service(health_service)
        .add_service(WorkerServer::new(worker_service))
        .serve(addr)
        .await
        .map_err(|e| WorkerError::Grpc(format!("Server error: {}", e)))?;
    
    Ok(())
}

use tonic::transport::Server;
use crate::worker::cluster::worker_server::WorkerServer;