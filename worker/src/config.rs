//! Worker configuration.
//!
//! [`WorkerConfig`] holds all tunable settings for the inference worker.
//! It can be loaded from a TOML file or constructed with defaults.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::WorkerError;

/// Configuration for the AI worker.
///
/// `Debug` is hand-written (not derived) so `hf_token` and `grpc_auth_token`
/// never land in logs via `info!("Configuration: {:?}", config)`.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerConfig {
    /// Optional human-readable worker identifier.
    pub worker_id: Option<String>,

    /// Bind interface for the gRPC server. Default `127.0.0.1` (loopback,
    /// secure by default); pair a non-loopback value with `grpc_auth_token`.
    pub grpc_bind_host: String,

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

    /// Whether this node lends its GPU(s) to a distributed model's lead node
    /// as a ggml-RPC peer (`distributed_role=rpc_server` metadata).
    pub rpc_server_enabled: bool,

    /// Base TCP port this node's `rpc-server` process(es) bind to when
    /// `rpc_server_enabled=true`. One GPU per port, starting here.
    pub rpc_server_port: u16,

    /// Bind interface for the `rpc-server` process(es). ggml-RPC has no
    /// auth — never bind this to a public interface.
    pub rpc_server_bind_host: Option<String>,

    /// Path to the `llama-server` binary the worker spawns for
    /// `engine = "llamaserver"` models. Not built by cargo — install
    /// llama.cpp's `llama-server` separately. `LLAMASERVER_BINARY_PATH` wins.
    pub llamaserver_binary_path: String,

    /// Bind interface passed to `llama-server --host`. Default `"127.0.0.1"`
    /// (loopback, secure by default) — that port has no built-in auth of its
    /// own. See docs/configuration.md.
    pub llamaserver_bind_host: String,

    /// Expose `llama-server`'s `/slots` endpoint. Default `false` — it returns
    /// per-slot state including cached prompt text. See docs/configuration.md.
    pub llamaserver_enable_slots_endpoint: bool,

    /// Inclusive allowed range for `llamaserver.port` metadata. Defaults
    /// (1024-65535) exclude privileged ports.
    pub llamaserver_port_min: u16,
    pub llamaserver_port_max: u16,

    /// Maximum number of models kept resident at once. `0` (default) means
    /// unlimited. Above the limit, the oldest-loaded model(s) are evicted
    /// first. Set to `1` for one-model-at-a-time on memory-constrained hosts.
    pub max_loaded_models: usize,

    /// Ceiling applied to any caller-supplied per-slot `n_ctx` before it
    /// sizes a KV-cache reservation or reaches `llama-server -c`. Does not
    /// raise a model's own trained context ceiling. See docs/configuration.md.
    pub max_n_ctx: u32,

    /// Shared-secret token gRPC clients must present (metadata key
    /// `x-worker-token`) for every RPC. `None` (default) leaves the server
    /// open — only safe while `grpc_bind_host` stays loopback.
    /// `WORKER_GRPC_AUTH_TOKEN` wins over this value.
    pub grpc_auth_token: Option<String>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            worker_id: None,
            grpc_bind_host: "127.0.0.1".to_string(),
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
            rpc_server_enabled: false,
            rpc_server_port: 50151,
            rpc_server_bind_host: None,
            llamaserver_binary_path: "llama-server".to_string(),
            llamaserver_bind_host: "127.0.0.1".to_string(),
            llamaserver_enable_slots_endpoint: false,
            llamaserver_port_min: 1024,
            llamaserver_port_max: 65535,
            max_loaded_models: 0,
            max_n_ctx: 262_144,
            grpc_auth_token: None,
        }
    }
}

impl fmt::Debug for WorkerConfig {
    /// Hand-written: `hf_token`/`grpc_auth_token` are redacted so startup
    /// logging never leaks a secret.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerConfig")
            .field("worker_id", &self.worker_id)
            .field("grpc_bind_host", &self.grpc_bind_host)
            .field("grpc_port", &self.grpc_port)
            .field("metrics_port", &self.metrics_port)
            .field("gpu_ids", &self.gpu_ids)
            .field("model_cache_dir", &self.model_cache_dir)
            .field("download_dir", &self.download_dir)
            .field("max_concurrent_loads", &self.max_concurrent_loads)
            .field("request_timeout_secs", &self.request_timeout_secs)
            .field("max_concurrent_requests", &self.max_concurrent_requests)
            .field("hf_token", &self.hf_token.as_ref().map(|_| "<redacted>"))
            .field("hf_cache_dir", &self.hf_cache_dir)
            .field("llamacpp_n_threads", &self.llamacpp_n_threads)
            .field(
                "llamacpp_default_n_gpu_layers",
                &self.llamacpp_default_n_gpu_layers,
            )
            .field("rpc_server_enabled", &self.rpc_server_enabled)
            .field("rpc_server_port", &self.rpc_server_port)
            .field("rpc_server_bind_host", &self.rpc_server_bind_host)
            .field("llamaserver_binary_path", &self.llamaserver_binary_path)
            .field("llamaserver_bind_host", &self.llamaserver_bind_host)
            .field(
                "llamaserver_enable_slots_endpoint",
                &self.llamaserver_enable_slots_endpoint,
            )
            .field("llamaserver_port_min", &self.llamaserver_port_min)
            .field("llamaserver_port_max", &self.llamaserver_port_max)
            .field("max_loaded_models", &self.max_loaded_models)
            .field("max_n_ctx", &self.max_n_ctx)
            .field(
                "grpc_auth_token",
                &self.grpc_auth_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

impl WorkerConfig {
    /// Load configuration from a TOML file, falling back to [`Default`] for
    /// missing keys. Logs a warning and returns defaults if the file is absent.
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
        // The shipped Docker config explicitly opts into non-loopback binds
        // for cross-container reachability — see config/worker.toml.
        assert_eq!(config.grpc_bind_host, "0.0.0.0");
        assert_eq!(config.llamaserver_bind_host, "0.0.0.0");
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

    #[test]
    fn test_rpc_server_defaults() {
        let config = WorkerConfig::default();
        assert!(!config.rpc_server_enabled);
        assert_eq!(config.rpc_server_port, 50151);
        assert_eq!(config.rpc_server_bind_host, None);
    }

    #[test]
    fn test_rpc_server_keys_parse_from_flat_toml() {
        let toml_str = "rpc_server_enabled = true\nrpc_server_port = 60000\nrpc_server_bind_host = \"0.0.0.0\"\n";
        let config: WorkerConfig = toml::from_str(toml_str).unwrap();
        assert!(config.rpc_server_enabled);
        assert_eq!(config.rpc_server_port, 60000);
        assert_eq!(config.rpc_server_bind_host, Some("0.0.0.0".to_string()));
    }

    #[test]
    fn test_llamaserver_defaults() {
        let config = WorkerConfig::default();
        assert_eq!(config.llamaserver_binary_path, "llama-server");
        // Loopback-only by default — opt in to 0.0.0.0 explicitly.
        assert_eq!(config.llamaserver_bind_host, "127.0.0.1");
        assert!(!config.llamaserver_enable_slots_endpoint);
    }

    #[test]
    fn test_grpc_bind_host_defaults_to_loopback() {
        // Secure by default — no gRPC exposure without an explicit opt-in.
        let config = WorkerConfig::default();
        assert_eq!(config.grpc_bind_host, "127.0.0.1");
        assert_eq!(config.grpc_auth_token, None);
    }

    #[test]
    fn test_grpc_bind_host_parses_from_flat_toml() {
        let toml_str = "grpc_bind_host = \"0.0.0.0\"\ngrpc_auth_token = \"secret\"\n";
        let config: WorkerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.grpc_bind_host, "0.0.0.0");
        assert_eq!(config.grpc_auth_token, Some("secret".to_string()));
    }

    #[test]
    fn test_max_n_ctx_default_and_parses() {
        let config = WorkerConfig::default();
        assert_eq!(config.max_n_ctx, 262_144);
        let toml_str = "max_n_ctx = 8192\n";
        let config: WorkerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.max_n_ctx, 8192);
    }

    #[test]
    fn test_debug_redacts_secrets() {
        let mut config = WorkerConfig::default();
        config.hf_token = Some("hf_supersecret".to_string());
        config.grpc_auth_token = Some("worker-shared-secret".to_string());
        let rendered = format!("{:?}", config);
        assert!(!rendered.contains("hf_supersecret"));
        assert!(!rendered.contains("worker-shared-secret"));
        assert!(rendered.contains("<redacted>"));
    }

    #[test]
    fn test_llamaserver_keys_parse_from_flat_toml() {
        let toml_str =
            "llamaserver_binary_path = \"/opt/llama.cpp/llama-server\"\nllamaserver_bind_host = \"127.0.0.1\"\n";
        let config: WorkerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            config.llamaserver_binary_path,
            "/opt/llama.cpp/llama-server"
        );
        assert_eq!(config.llamaserver_bind_host, "127.0.0.1");
    }

    #[test]
    fn test_max_loaded_models_default_unlimited() {
        let config = WorkerConfig::default();
        assert_eq!(config.max_loaded_models, 0);
    }

    #[test]
    fn test_max_loaded_models_parses_from_flat_toml() {
        let toml_str = "max_loaded_models = 1\n";
        let config: WorkerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.max_loaded_models, 1);
    }

    #[test]
    fn test_shipped_config_parses_with_max_loaded_models_one() {
        // The shipped worker.toml targets a DGX Spark and pins one resident model.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../config/worker.toml");
        let config = WorkerConfig::from_file(path).unwrap();
        assert_eq!(config.max_loaded_models, 1);
    }
}
