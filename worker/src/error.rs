//! Worker error types.
//!
//! Defines [`WorkerError`], the central error enum used across all worker
//! modules.  Variants are derived from every `WorkerError::*` usage site
//! found in the codebase.


use thiserror::Error;

/// Central error type for the AI worker.
#[derive(Error, Debug)]
pub enum WorkerError {
    /// No GPU devices were detected on this host.
    #[error("No GPU devices found on this system")]
    NoGpusFound,

    /// A generic GPU-level error.
    #[error("GPU error: {0}")]
    Gpu(String),

    /// An error with the worker configuration (parsing, validation, etc.).
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// config error
    #[error("Config error: {0}")]
    Config(String),

    /// Tokio runtime creation or usage error.
    #[error("Runtime error: {0}")]
    Runtime(String),

    /// gRPC transport or protocol error.
    #[error("gRPC error: {0}")]
    Grpc(String),

    /// A shared resource (semaphore, lock, etc.) could not be acquired.
    #[error("Resource error: {0}")]
    Resource(String),

    /// Insufficient GPU memory for the requested allocation.
    #[error("Out of memory on GPU {device}: requested {requested} bytes but only {available} bytes available")]
    OutOfMemory {
        /// Bytes requested.
        requested: usize,
        /// Bytes currently available.
        available: usize,
        /// Device index.
        device: usize,
    },

    /// Model loading / weight deserialization error.
    #[error("Model load error: {0}")]
    ModelLoad(String),

    /// The requested model is not currently loaded.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Multi-GPU parallelism setup or execution error.
    #[error("Parallelism error: {0}")]
    Parallelism(String),

    /// Standard I/O error wrapper.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Catch-all for other errors.
    #[error("{0}")]
    Other(String),

    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// Allow conversion from `anyhow::Error` via `.to_string()`.
impl From<anyhow::Error> for WorkerError {
    fn from(err: anyhow::Error) -> Self {
        WorkerError::Other(err.to_string())
    }
}

// Allow conversion from `tonic::Status`.
impl From<tonic::Status> for WorkerError {
    fn from(status: tonic::Status) -> Self {
        WorkerError::Grpc(status.message().to_string())
    }
}

// Convenience conversion *to* `tonic::Status` so handlers can use `?`.
impl From<WorkerError> for tonic::Status {
    fn from(err: WorkerError) -> Self {
        match &err {
            WorkerError::ModelNotFound(_) => tonic::Status::not_found(err.to_string()),
            WorkerError::OutOfMemory { .. } => {
                tonic::Status::resource_exhausted(err.to_string())
            }
            WorkerError::Configuration(_) => {
                tonic::Status::invalid_argument(err.to_string())
            }
            _ => tonic::Status::internal(err.to_string()),
        }
    }
}



