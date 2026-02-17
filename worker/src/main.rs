use std::env;
use tracing_subscriber::fmt::Subscriber;
use worker::worker::start_worker_service;

fn main() {
    // Simple env‑driven config – override via Docker env
    let port = env::var("WORKER_PORT").unwrap_or_else(|_|
"50051".to_string());
    let port: u16 = port.parse().expect("WORKER_PORT must be a
number");

    // Init tracing (prints to stdout)
    Subscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Launch the gRPC worker
    start_worker_service(port).expect("Failed to start worker
service");
}