//! `llama-server` process supervision (engine `"llamaserver"`, Plan 13).
//!
//! Unlike the in-process [`crate::llamacpp_engine`] (feature-gated, links
//! `libllama`), the `llamaserver` engine is **pure process management** and is
//! always compiled: the worker spawns and supervises a `llama-server` child
//! process per model and the coordinator proxies agentic HTTP inference
//! (OpenAI / Anthropic tool calling, streaming `tool_calls`, ...) straight to
//! it. The control plane stays gRPC (`LoadModel`/`UnloadModel`); the data plane
//! becomes an HTTP proxy owned by the coordinator.
//!
//! This module carries no `llama-cpp-2` dependency, so it builds and its unit
//! tests run under a default `cargo build`/`cargo test` (no libclang, no GPU).

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use tokio::process::{Child, Command};
use tokio::time::{sleep, Instant};
use tracing::{debug, info, warn};

use crate::error::WorkerError;
use crate::model_loader::parse_cache_type;

/// `--parallel` value when the `llamaserver.parallel` metadata key is omitted.
const DEFAULT_PARALLEL: u32 = 4;

/// Spawn/health/kill config parsed from `ModelConfig.metadata` for a model with
/// `engine = "llamaserver"`.
///
/// File resolution (`repo_id`/`file`) and context size (`n_ctx`) reuse the
/// **existing** `gguf.*` metadata keys the coordinator already emits for the
/// in-process llama.cpp engine — only the `llamaserver.*` keys are new. All keys
/// are part of the cross-language gRPC contract (see
/// `pending-work/13-agentic-serving-llama-server.md`); do not rename them.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaServerSpec {
    /// HuggingFace repo containing the GGUF (metadata key `gguf_repo_id`).
    pub repo_id: String,
    /// Exact `.gguf` filename inside the repo (metadata key `gguf_file`).
    pub file: String,
    /// TCP port `llama-server` listens on (metadata key `llamaserver.port`,
    /// REQUIRED — coordinator-assigned, unique per model).
    pub port: u16,
    /// Context window override, maps to `-c` (metadata key `n_ctx`; `None`
    /// leaves `llama-server`'s own default).
    pub n_ctx: Option<u32>,
    /// Continuous-batching slots, maps to `--parallel` (metadata key
    /// `llamaserver.parallel`, default [`DEFAULT_PARALLEL`]).
    pub parallel: u32,
    /// KV-cache K quantization, maps to `-ctk` (metadata key `cache_type_k`).
    pub cache_type_k: Option<String>,
    /// KV-cache V quantization, maps to `-ctv` (metadata key `cache_type_v`).
    pub cache_type_v: Option<String>,
    /// Extra `llama-server` flags appended verbatim (metadata key
    /// `llamaserver.extra_args`, whitespace-split).
    pub extra_args: Vec<String>,
}

impl LlamaServerSpec {
    /// Build the exact `llama-server` argv (excluding the binary itself).
    ///
    /// Order (see the Plan 13 contract):
    /// `-m <path> --host <bind> --port <port> [-c <n_ctx>] -ngl 999
    /// --parallel <N> [-ctk <t>] [-ctv <t>] <extra_args...>`.
    ///
    /// `--jinja` is intentionally NOT passed — it is default-on in current
    /// llama.cpp.
    pub fn build_args(&self, gguf_path: &Path, bind_host: &str) -> Vec<String> {
        let mut args: Vec<String> = vec![
            "-m".to_string(),
            gguf_path.to_string_lossy().into_owned(),
            "--host".to_string(),
            bind_host.to_string(),
            "--port".to_string(),
            self.port.to_string(),
        ];
        // `-c` only when an explicit n_ctx was configured; otherwise let
        // llama-server pick its own default (the metadata key is optional).
        if let Some(n_ctx) = self.n_ctx {
            args.push("-c".to_string());
            args.push(n_ctx.to_string());
        }
        // Offload everything to the GPU (llama.cpp clamps to the real layer
        // count; 999 == "all").
        args.push("-ngl".to_string());
        args.push("999".to_string());
        args.push("--parallel".to_string());
        args.push(self.parallel.to_string());
        if let Some(k) = &self.cache_type_k {
            args.push("-ctk".to_string());
            args.push(k.clone());
        }
        if let Some(v) = &self.cache_type_v {
            args.push("-ctv".to_string());
            args.push(v.clone());
        }
        args.extend(self.extra_args.iter().cloned());
        args
    }
}

/// Parse `engine = "llamaserver"` routing metadata.
///
/// * `Ok(None)` — no metadata, or `engine` is absent / not `"llamaserver"`
///   (burn and in-process llamacpp models fall through untouched).
/// * `Ok(Some(spec))` — a complete llamaserver spec.
/// * `Err(..)` — missing `llamaserver.port` / `gguf_repo_id` / `gguf_file`, or a
///   malformed numeric / cache-type value.
pub fn llamaserver_spec_from_metadata(
    metadata: Option<&HashMap<String, String>>,
) -> Result<Option<LlamaServerSpec>, WorkerError> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    if metadata.get("engine").map(String::as_str) != Some("llamaserver") {
        return Ok(None);
    }

    let repo_id = metadata
        .get("gguf_repo_id")
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            WorkerError::Configuration(
                "llamaserver engine requires metadata key 'gguf_repo_id'".to_string(),
            )
        })?
        .clone();
    let file = metadata
        .get("gguf_file")
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            WorkerError::Configuration(
                "llamaserver engine requires metadata key 'gguf_file'".to_string(),
            )
        })?
        .clone();

    let port_str = metadata
        .get("llamaserver.port")
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            WorkerError::Configuration(
                "llamaserver engine requires metadata key 'llamaserver.port'".to_string(),
            )
        })?;
    let port = port_str.parse::<u16>().map_err(|e| {
        WorkerError::Configuration(format!("invalid llamaserver.port '{port_str}': {e}"))
    })?;

    let parallel = match metadata.get("llamaserver.parallel") {
        Some(v) => v.parse::<u32>().map_err(|e| {
            WorkerError::Configuration(format!("invalid llamaserver.parallel '{v}': {e}"))
        })?,
        None => DEFAULT_PARALLEL,
    };

    let n_ctx = match metadata.get("n_ctx") {
        Some(v) => Some(
            v.parse::<u32>()
                .map_err(|e| WorkerError::Configuration(format!("invalid n_ctx '{v}': {e}")))?,
        ),
        None => None,
    };

    // Reuse the same validated cache-type keys as the in-process engine.
    let cache_type_k = parse_cache_type(metadata, "cache_type_k")?;
    let cache_type_v = parse_cache_type(metadata, "cache_type_v")?;

    let extra_args = metadata
        .get("llamaserver.extra_args")
        .map(|s| s.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default();

    Ok(Some(LlamaServerSpec {
        repo_id,
        file,
        port,
        n_ctx,
        parallel,
        cache_type_k,
        cache_type_v,
        extra_args,
    }))
}

/// A supervised `llama-server` child process.
///
/// Spawning does NOT block on readiness; call [`Self::wait_until_healthy`]
/// afterwards. The child is killed explicitly on unload
/// ([`Self::start_kill`]/[`Self::shutdown`]) and, as a backstop,
/// `kill_on_drop` ensures it never outlives the worker.
pub struct LlamaServerProcess {
    /// Registry key of the model this process serves (for logs/errors).
    model_name: String,
    /// Loopback + LAN port `llama-server` listens on.
    port: u16,
    /// The child handle. `try_wait`/`start_kill` need `&mut`, so callers reach
    /// this through the outer `tokio::sync::Mutex` the loader stores it behind.
    child: Child,
}

impl LlamaServerProcess {
    /// Spawn `binary args...` WITHOUT waiting for HTTP health.
    ///
    /// `binary` is resolved via `PATH` unless it is an absolute path. Used both
    /// for the real `llama-server` and, in tests, for a harmless stand-in
    /// (e.g. `sleep`) so no real binary or GPU is required.
    pub fn spawn(
        model_name: &str,
        port: u16,
        binary: &str,
        args: &[String],
    ) -> Result<Self, WorkerError> {
        info!(
            "spawning llama-server for '{}': {} {}",
            model_name,
            binary,
            args.join(" ")
        );
        let child = Command::new(binary)
            .args(args)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                WorkerError::ModelLoad(format!(
                    "failed to spawn llama-server binary '{binary}' for model '{model_name}': {e} \
                     (install llama.cpp's llama-server on PATH or set LLAMASERVER_BINARY_PATH)"
                ))
            })?;
        Ok(Self {
            model_name: model_name.to_string(),
            port,
            child,
        })
    }

    /// `true` while the child is still running. Reaps the child on exit
    /// (non-blocking), so a self-exited process is detected here and never
    /// lingers as a zombie.
    pub fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Poll `GET http://127.0.0.1:<port>/health` until it returns 200, the
    /// child exits, or `timeout` elapses.
    ///
    /// A minimal loopback HTTP GET via the `reqwest` client already present in
    /// the dependency tree (no new heavy dependency). Fails fast if the child
    /// dies during startup (bad binary / args / port clash).
    pub async fn wait_until_healthy(&mut self, timeout: Duration) -> Result<(), WorkerError> {
        let url = format!("http://127.0.0.1:{}/health", self.port);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| {
                WorkerError::ModelLoad(format!("failed to build health-check http client: {e}"))
            })?;
        let deadline = Instant::now() + timeout;
        loop {
            // Fail fast: if the child already exited it will never be healthy.
            if let Ok(Some(status)) = self.child.try_wait() {
                return Err(WorkerError::ModelLoad(format!(
                    "llama-server for '{}' exited during startup ({status})",
                    self.model_name
                )));
            }
            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().as_u16() == 200 {
                    info!(
                        "llama-server for '{}' healthy on port {}",
                        self.model_name, self.port
                    );
                    return Ok(());
                }
            }
            if Instant::now() >= deadline {
                return Err(WorkerError::ModelLoad(format!(
                    "llama-server for '{}' did not become healthy within {:?} (GET {})",
                    self.model_name, timeout, url
                )));
            }
            sleep(Duration::from_millis(250)).await;
        }
    }

    /// Send the terminate signal to the child (SIGKILL on Unix /
    /// `TerminateProcess` on Windows) without awaiting. Non-blocking and safe to
    /// call on an already-exited child. Cross-platform via tokio's
    /// [`Child::start_kill`] — no unix-only APIs.
    pub fn start_kill(&mut self) -> Result<(), WorkerError> {
        self.child.start_kill().map_err(WorkerError::Io)
    }

    /// Terminate the child and await its reaping. Used by graceful unload and by
    /// tests that need a deterministic "it is gone now" point.
    pub async fn shutdown(&mut self) -> Result<(), WorkerError> {
        if let Err(e) = self.child.start_kill() {
            // Already exited / already killed — nothing to do.
            debug!("start_kill for '{}': {}", self.model_name, e);
        }
        if let Err(e) = self.child.wait().await {
            warn!("wait after kill for '{}': {}", self.model_name, e);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // --- metadata -> spec parsing ------------------------------------------

    #[test]
    fn spec_none_without_metadata_or_for_other_engines() {
        assert_eq!(llamaserver_spec_from_metadata(None).unwrap(), None);
        let empty = meta(&[]);
        assert_eq!(llamaserver_spec_from_metadata(Some(&empty)).unwrap(), None);
        // burn / llamacpp models must fall through untouched (Ok(None)).
        let burn = meta(&[("engine", "burn")]);
        assert_eq!(llamaserver_spec_from_metadata(Some(&burn)).unwrap(), None);
        let llamacpp = meta(&[("engine", "llamacpp"), ("gguf_repo_id", "x/y")]);
        assert_eq!(
            llamaserver_spec_from_metadata(Some(&llamacpp)).unwrap(),
            None
        );
    }

    #[test]
    fn spec_parses_full_happy_path() {
        let m = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "unsloth/Devstral-Small-2-24B-GGUF"),
            ("gguf_file", "devstral-small-2-24b-q4_k_m.gguf"),
            ("llamaserver.port", "8081"),
            ("llamaserver.parallel", "8"),
            ("llamaserver.extra_args", "--flash-attn -np 8"),
            ("n_ctx", "32768"),
            ("cache_type_k", "q8_0"),
            ("cache_type_v", "q4_0"),
        ]);
        let spec = llamaserver_spec_from_metadata(Some(&m)).unwrap().unwrap();
        assert_eq!(spec.repo_id, "unsloth/Devstral-Small-2-24B-GGUF");
        assert_eq!(spec.file, "devstral-small-2-24b-q4_k_m.gguf");
        assert_eq!(spec.port, 8081);
        assert_eq!(spec.parallel, 8);
        assert_eq!(spec.n_ctx, Some(32768));
        assert_eq!(spec.cache_type_k, Some("q8_0".to_string()));
        assert_eq!(spec.cache_type_v, Some("q4_0".to_string()));
        assert_eq!(
            spec.extra_args,
            vec![
                "--flash-attn".to_string(),
                "-np".to_string(),
                "8".to_string()
            ]
        );
    }

    #[test]
    fn spec_defaults_parallel_and_optional_fields() {
        let m = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "9000"),
        ]);
        let spec = llamaserver_spec_from_metadata(Some(&m)).unwrap().unwrap();
        assert_eq!(spec.parallel, DEFAULT_PARALLEL);
        assert_eq!(spec.n_ctx, None);
        assert_eq!(spec.cache_type_k, None);
        assert_eq!(spec.cache_type_v, None);
        assert!(spec.extra_args.is_empty());
    }

    #[test]
    fn spec_rejects_missing_port() {
        let m = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
        ]);
        let err = llamaserver_spec_from_metadata(Some(&m)).unwrap_err();
        assert!(err.to_string().contains("llamaserver.port"));
    }

    #[test]
    fn spec_rejects_bad_port() {
        // non-numeric
        let m = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "not-a-port"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&m)).is_err());
        // out of u16 range
        let m = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "70000"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&m)).is_err());
    }

    #[test]
    fn spec_rejects_missing_gguf_source() {
        let no_repo = meta(&[
            ("engine", "llamaserver"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "8080"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&no_repo)).is_err());
        let no_file = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("llamaserver.port", "8080"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&no_file)).is_err());
    }

    #[test]
    fn spec_rejects_bad_parallel_or_cache_type() {
        let bad_parallel = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "8080"),
            ("llamaserver.parallel", "lots"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&bad_parallel)).is_err());
        let bad_cache = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "8080"),
            ("cache_type_k", "int8"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&bad_cache)).is_err());
    }

    // --- arg building (exact argv) -----------------------------------------

    #[test]
    fn build_args_full_exact_argv() {
        let spec = LlamaServerSpec {
            repo_id: "x/y".to_string(),
            file: "m.gguf".to_string(),
            port: 8081,
            n_ctx: Some(32768),
            parallel: 8,
            cache_type_k: Some("q8_0".to_string()),
            cache_type_v: Some("q4_0".to_string()),
            extra_args: vec!["--flash-attn".to_string(), "--mlock".to_string()],
        };
        let args = spec.build_args(Path::new("/models/m.gguf"), "0.0.0.0");
        assert_eq!(
            args,
            vec![
                "-m",
                "/models/m.gguf",
                "--host",
                "0.0.0.0",
                "--port",
                "8081",
                "-c",
                "32768",
                "-ngl",
                "999",
                "--parallel",
                "8",
                "-ctk",
                "q8_0",
                "-ctv",
                "q4_0",
                "--flash-attn",
                "--mlock",
            ]
        );
    }

    #[test]
    fn build_args_minimal_omits_optional_flags() {
        let spec = LlamaServerSpec {
            repo_id: "x/y".to_string(),
            file: "m.gguf".to_string(),
            port: 8080,
            n_ctx: None,
            parallel: DEFAULT_PARALLEL,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: vec![],
        };
        let args = spec.build_args(Path::new("/m.gguf"), "127.0.0.1");
        // No -c (n_ctx absent), no -ctk/-ctv (cache types absent), no extras.
        assert_eq!(
            args,
            vec![
                "-m",
                "/m.gguf",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "-ngl",
                "999",
                "--parallel",
                "4",
            ]
        );
    }

    // --- supervision (no real llama-server binary / GPU) -------------------

    #[tokio::test]
    async fn spawn_reports_running_then_kill_on_unload_stops_it() {
        // A harmless long-running stand-in for llama-server; health-check is
        // deliberately skipped (no HTTP server is listening).
        let mut proc = LlamaServerProcess::spawn("test-model", 12345, "sleep", &["30".to_string()])
            .expect("`sleep` should spawn on Linux");
        assert!(
            proc.is_running(),
            "child should be running right after spawn"
        );

        // kill-on-unload: shutdown() terminates and reaps the child.
        proc.shutdown().await.unwrap();
        assert!(!proc.is_running(), "child must be gone after shutdown()");
    }

    #[tokio::test]
    async fn exited_child_is_detected() {
        // `true` exits 0 immediately — stands in for a llama-server that dies on
        // its own (bad flags, OOM, port clash). Detection must not panic.
        let mut proc = LlamaServerProcess::spawn("test-model", 12346, "true", &[])
            .expect("`true` should spawn on Linux");
        // Give it a moment to exit and be reaped by is_running()'s try_wait.
        let mut gone = false;
        for _ in 0..100 {
            if !proc.is_running() {
                gone = true;
                break;
            }
            sleep(Duration::from_millis(20)).await;
        }
        assert!(gone, "a self-exited child must report not-running");
    }
}
