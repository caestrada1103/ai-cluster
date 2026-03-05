//! Worker configuration.
//!
//! [`WorkerConfig`] holds all tunable settings for the inference worker.
//! It can be loaded from a TOML file or constructed with defaults.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::WorkerError;

/// Configuration for the AI worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerConfig {
    /// Optional human-readable worker identifier.
    pub worker_id: Option<String>,

    /// Port for the gRPC inference server.
    pub grpc_port: u16,

    /// Port for the Prometheus metrics HTTP server.
    pub metrics_port: u16,

    /// GPU device indices this worker should use.
    pub gpu_ids: Vec<usize>,

    /// Directory for caching downloaded model weights.
    pub model_cache_dir: PathBuf,

    /// Directory for in-progress downloads.
    pub download_dir: PathBuf,

    /// Maximum number of models that can be loaded concurrently.
    pub max_concurrent_loads: usize,

    /// Timeout (seconds) for a single model load operation.
    pub load_timeout_secs: u64,

    /// Timeout (seconds) for a single inference request.
    pub request_timeout_secs: u64,

    /// Whether to verify file checksums after download.
    pub verify_checksums: bool,

    /// Use memory-mapped I/O when reading weight files.
    pub enable_mmap: bool,

    /// Pin host memory for faster GPU transfers.
    pub pin_memory: bool,

    /// Maximum number of concurrent inference requests per model.
    pub max_concurrent_requests: usize,

    /// HuggingFace Hub token for gated model downloads.
    pub hf_token: Option<String>,

    /// HuggingFace Hub cache directory override.
    pub hf_cache_dir: Option<PathBuf>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            worker_id: None,
            grpc_port: 50051,
            metrics_port: 9091,
            gpu_ids: vec![0],
            model_cache_dir: PathBuf::from("models"),
            download_dir: PathBuf::from("downloads"),
            max_concurrent_loads: 2,
            load_timeout_secs: 300,
            request_timeout_secs: 60,
            verify_checksums: true,
            enable_mmap: true,
            pin_memory: false,
            max_concurrent_requests: 32,
            hf_token: None,
            hf_cache_dir: None,
        }
    }
}

impl WorkerConfig {
    /// Load configuration from a TOML file.
    ///
    /// Falls back to [`Default`] values for any missing keys.  If the file
    /// does not exist, a warning is logged and pure defaults are returned.
    pub fn from_file(path: &str) -> Result<Self, WorkerError> {
        let path = PathBuf::from(path);

        if !path.exists() {
            tracing::warn!(
                "Config file {} not found, using defaults",
                path.display()
            );
            return Ok(Self::default());
        }

        let contents = std::fs::read_to_string(&path).map_err(|e| {
            WorkerError::Configuration(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;

        let config: Self = toml::from_str(&contents).map_err(|e| {
            WorkerError::Configuration(format!(
                "Failed to parse config file {}: {}",
                path.display(),
                e
            ))
        })?;

        Ok(config)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WorkerConfig::default();
        assert_eq!(config.grpc_port, 50051);
        assert_eq!(config.metrics_port, 9091);
        assert_eq!(config.gpu_ids, vec![0]);
        assert!(config.verify_checksums);
    }

    #[test]
    fn test_from_file_missing() {
        // Should return defaults when file doesn't exist
        let config = WorkerConfig::from_file("nonexistent.toml").unwrap();
        assert_eq!(config.grpc_port, 50051);
    }
}
