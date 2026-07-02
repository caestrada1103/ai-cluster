//! Worker configuration.
//!
//! [`WorkerConfig`] holds all tunable settings for the inference worker.
//! It can be loaded from a TOML file or constructed with defaults.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::WorkerError;

/// Configuration for the AI worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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

    /// Timeout (seconds) for a single inference request.
    pub request_timeout_secs: u64,

    /// Maximum number of concurrent inference requests per model.
    pub max_concurrent_requests: usize,

    /// HuggingFace Hub token for gated model downloads.
    pub hf_token: Option<String>,

    /// HuggingFace Hub cache directory override.
    pub hf_cache_dir: Option<PathBuf>,

    /// CPU threads for llama.cpp generation (0 = let llama.cpp auto-detect).
    /// Only used when the worker is built with the `llamacpp` feature.
    pub llamacpp_n_threads: i32,

    /// Default number of transformer layers to offload to the GPU for
    /// llama.cpp models (-1 = all layers). Per-model `n_gpu_layers` metadata
    /// from the registry overrides this.
    pub llamacpp_default_n_gpu_layers: i32,
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
            request_timeout_secs: 60,
            max_concurrent_requests: 32,
            hf_token: None,
            hf_cache_dir: None,
            llamacpp_n_threads: 0,
            llamacpp_default_n_gpu_layers: -1,
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
            tracing::warn!("Config file {} not found, using defaults", path.display());
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
        assert_eq!(config.max_concurrent_requests, 32);
    }

    #[test]
    fn test_from_file_missing() {
        // Should return defaults when file doesn't exist
        let config = WorkerConfig::from_file("nonexistent.toml").unwrap();
        assert_eq!(config.grpc_port, 50051);
    }

    #[test]
    fn test_shipped_config_parses_with_non_default_values() {
        // The file Docker ships MUST parse into real (non-default) values.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../config/worker.toml");
        let config = WorkerConfig::from_file(path).unwrap();
        // 600 differs from the removed default on purpose — proves parsing works.
        assert_eq!(config.request_timeout_secs, 120);
        assert_eq!(config.grpc_port, 50051);
    }

    #[test]
    fn test_unknown_keys_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.toml");
        std::fs::write(&p, "[grpc]\nport = 50051\n").unwrap();
        let err = WorkerConfig::from_file(p.to_str().unwrap()).unwrap_err();
        assert!(err.to_string().contains("Failed to parse config"));
    }

    #[test]
    fn test_llamacpp_defaults() {
        let config = WorkerConfig::default();
        assert_eq!(config.llamacpp_n_threads, 0);
        assert_eq!(config.llamacpp_default_n_gpu_layers, -1);
    }

    #[test]
    fn test_llamacpp_keys_parse_from_flat_toml() {
        let toml_str = "llamacpp_n_threads = 8\nllamacpp_default_n_gpu_layers = 20\n";
        let config: WorkerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.llamacpp_n_threads, 8);
        assert_eq!(config.llamacpp_default_n_gpu_layers, 20);
    }
}
