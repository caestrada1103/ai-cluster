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

/// C2 — flags [`LlamaServerSpec::extra_args`] (metadata key
/// `llamaserver.extra_args`, coordinator-controlled and, transitively,
/// attacker-influenceable if C3/C4 auth is ever misconfigured) is allowed to
/// contain. `extra_args` is whitespace-split verbatim and appended to the
/// `llama-server` child's argv (`execve`-style — no shell — so this is flag
/// injection, not shell injection, but that is still enough for e.g. `--path
/// <dir>` to turn the child into an arbitrary-file HTTP server, `--log-file
/// <path>` into an arbitrary write primitive, or `--host`/`--port` to
/// silently rebind the process this loader just spawned and is tracking by a
/// DIFFERENT port).
///
/// Every flag `llama-server --help` accepts that takes a filesystem path, a
/// network address, or a credential is deliberately left OFF this list —
/// those already have first-class, validated fields on [`LlamaServerSpec`]
/// (`port`, `n_gpu_layers`, `n_cpu_moe`, `cache_type_k`/`v`) or simply have no
/// legitimate use here. Flags that DUPLICATE one of those typed fields
/// (`-np`/`--parallel`, `-c`/`--ctx-size`, `-m`/`--model`, `-ngl`/
/// `--n-gpu-layers`, `-ncmoe`/`--n-cpu-moe`, `-ctk`/`-ctv`, `--host`,
/// `--port`) are ALSO excluded — allowing them here would let a second,
/// conflicting flag silently override the value this loader already used to
/// size the KV-cache GPU reservation (H2) or to build the argv, which
/// `llama-server`'s "last flag wins" argv parsing would accept without
/// complaint. This is a strict ALLOWLIST: any token that looks like a flag
/// (starts with `-`) and is not in this set is rejected outright — including
/// flags not yet invented when this list was written.
const ALLOWED_EXTRA_FLAGS: &[&str] = &[
    // Sampling / generation tuning — no path or network semantics.
    "--temp",
    "--temperature",
    "--top-k",
    "--top-p",
    "--min-p",
    "--top-nsigma",
    "--top-n-sigma",
    "--repeat-last-n",
    "--repeat-penalty",
    "--presence-penalty",
    "--frequency-penalty",
    "--samplers",
    "--mirostat",
    "--mirostat-lr",
    "--mirostat-ent",
    "--dynatemp-range",
    "--dynatemp-exp",
    "--xtc-probability",
    "--xtc-threshold",
    "--typical",
    "--typical-p",
    "--dry-multiplier",
    "--dry-base",
    "--dry-allowed-length",
    "--dry-penalty-last-n",
    "--ignore-eos",
    // Performance / batching / offload tuning — numbers and enums only.
    "-fa",
    "--flash-attn",
    "--mlock",
    "--mmap",
    "--no-mmap",
    "-t",
    "--threads",
    "-tb",
    "--threads-batch",
    "-b",
    "--batch-size",
    "-ub",
    "--ubatch-size",
    "--warmup",
    "--no-warmup",
    "--numa",
    "-sm",
    "--split-mode",
    "-ts",
    "--tensor-split",
    "-mg",
    "--main-gpu",
    "-dev",
    "--device",
    "-kvo",
    "--kv-offload",
    "-nkvo",
    "--no-kv-offload",
    "-kvu",
    "--kv-unified",
    "-no-kvu",
    "--no-kv-unified",
    "--swa-full",
    "-cb",
    "--cont-batching",
    "-nocb",
    "--no-cont-batching",
    "--context-shift",
    "--no-context-shift",
    "--check-tensors",
    "--op-offload",
    "--no-op-offload",
    "-nr",
    "--repack",
    "--no-repack",
    "-dt",
    "--defrag-thold",
    // RoPE scaling — numeric only.
    "--rope-scaling",
    "--rope-scale",
    "--rope-freq-base",
    "--rope-freq-scale",
    "--yarn-orig-ctx",
    "--yarn-ext-factor",
    "--yarn-attn-factor",
    "--yarn-beta-slow",
    "--yarn-beta-fast",
    // Template engine toggle (not `--chat-template-file`/paths).
    "--jinja",
    "--no-jinja",
];

/// Reject an `extra_args` token that looks like a CLI flag (starts with `-`)
/// but is not on [`ALLOWED_EXTRA_FLAGS`]. Bare values (a flag's argument,
/// e.g. the `20` in `--top-k 20`) are not flags themselves and pass through
/// unchecked — the flag they belong to is what gates whether they can appear
/// at all.
fn validate_extra_arg(token: &str) -> Result<(), WorkerError> {
    if token.starts_with('-') && !ALLOWED_EXTRA_FLAGS.contains(&token) {
        return Err(WorkerError::Configuration(format!(
            "llamaserver.extra_args: flag '{token}' is not on the allowlist \
             (rejected — file/network/credential-affecting flags like --path, \
             --log-file, --host, --api-key-file, --lora, --slot-save-path, \
             --models-dir, --hf-token, etc. are never permitted here; use a \
             typed metadata field or the worker's own config for anything else)"
        )));
    }
    Ok(())
}

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
    /// **Per-conversation/slot** context window (metadata key `n_ctx`; `None`
    /// leaves `llama-server`'s own default, and no `-c` is passed).
    ///
    /// `llama-server` divides its own `-c` value EVENLY across `--parallel`
    /// slots (verified on hardware: `-c 262144` with `--parallel 4` reports
    /// `n_ctx: 65536, total_slots: 4` via `GET /props`) — so passing this
    /// field straight through as `-c` would silently give every conversation
    /// only `n_ctx / parallel` tokens. [`Self::total_ctx`] (used by
    /// [`Self::build_args`]) multiplies by `parallel` before building `-c` so
    /// this field keeps its intuitive registry meaning: the number of tokens
    /// ONE conversation actually gets, honoring the user requirement that a
    /// model's own trained context (`n_ctx_train`) is the ceiling AND the
    /// thing actually delivered, never silently shrunk underneath it.
    pub n_ctx: Option<u32>,
    /// Continuous-batching slots, maps to `--parallel` (metadata key
    /// `llamaserver.parallel`, default [`DEFAULT_PARALLEL`]). Raising this no
    /// longer shrinks per-slot context (see `n_ctx` above) — it costs
    /// `parallel` times the single-slot KV-cache footprint instead, so it
    /// must stay sized to the available VRAM/unified memory (see the
    /// per-entry arithmetic in `config/models.toml`).
    pub parallel: u32,
    /// Transformer layers to offload to the GPU, maps to `-ngl` (metadata key
    /// `n_gpu_layers`; same key/semantics as the in-process `llamacpp`
    /// engine's `GgufLoadSpec::n_gpu_layers` — `< 0` means "all", passed to
    /// `llama-server` as `-ngl 999` since llama.cpp clamps any value >= the
    /// real layer count to "all"). Defaults to the worker's
    /// `llamacpp_default_n_gpu_layers` config (itself `-1` — today's
    /// full-offload behavior — unless overridden), so existing deployments
    /// that never set this key are unaffected. This is the partial-offload
    /// knob consumer GPUs with 8-16 GB VRAM need when a model does not fit
    /// entirely on-GPU.
    pub n_gpu_layers: i32,
    /// MoE expert-tensor CPU offload, maps to `--n-cpu-moe <N>` (metadata key
    /// `n_cpu_moe`) — keeps the MoE weights of the first `N` layers on the
    /// CPU while `-ngl`/`n_gpu_layers` still places every non-expert tensor
    /// (and any non-offloaded experts) on GPU. This is llama.cpp's
    /// purpose-built knob for fitting a big MoE model's expert tensors into
    /// less VRAM than `n_gpu_layers` alone can achieve (see
    /// `qwen3-coder-30b-a3b-instruct-gguf` in `config/models.toml`, which
    /// used a raw `extra_args = "--n-cpu-moe 40"` workaround before this
    /// field existed). `None` omits the flag (llama.cpp's own default: no
    /// forced CPU offload).
    pub n_cpu_moe: Option<u32>,
    /// KV-cache K quantization, maps to `-ctk` (metadata key `cache_type_k`).
    pub cache_type_k: Option<String>,
    /// KV-cache V quantization, maps to `-ctv` (metadata key `cache_type_v`).
    pub cache_type_v: Option<String>,
    /// Extra `llama-server` flags appended verbatim (metadata key
    /// `llamaserver.extra_args`, whitespace-split).
    pub extra_args: Vec<String>,
}

impl LlamaServerSpec {
    /// The actual `-c` value: `n_ctx` (this struct's per-slot/per-conversation
    /// meaning) multiplied by `parallel`, so `llama-server`'s own division of
    /// `-c` across slots reproduces exactly `n_ctx` tokens per slot instead of
    /// silently shrinking it by a factor of `parallel` (see the `n_ctx` field
    /// doc and the "HIGHEST PRIORITY" fix in `config/models.toml`'s
    /// `qwen3.6-35b-a3b-gguf` entry for the full measured KV-cache
    /// arithmetic this implies).
    ///
    /// `Ok(None)` when `n_ctx` is unset (defer to `llama-server`'s own
    /// default; per-slot context is then whatever `llama-server` picks, not
    /// independently verifiable here). `Err` only on `u32` overflow — an
    /// astronomically large `n_ctx * parallel` that cannot be represented,
    /// caught explicitly rather than silently wrapping to a smaller (wrong)
    /// value.
    pub fn total_ctx(&self) -> Result<Option<u32>, WorkerError> {
        match self.n_ctx {
            None => Ok(None),
            Some(per_slot) => per_slot
                .checked_mul(self.parallel)
                .map(Some)
                .ok_or_else(|| {
                    WorkerError::Configuration(format!(
                        "n_ctx ({per_slot}) * parallel ({}) overflows u32 — lower one of them",
                        self.parallel
                    ))
                }),
        }
    }

    /// Build the exact `llama-server` argv (excluding the binary itself).
    ///
    /// Order: `-m <path> --host <bind> --port <port> [-c <total_ctx>] -ngl
    /// <N> [--n-cpu-moe <N>] --parallel <N> [-ctk <t>] [-ctv <t>]
    /// (--slots|--no-slots) <extra_args...>`.
    ///
    /// `--jinja` is intentionally NOT passed — it is default-on in current
    /// llama.cpp.
    ///
    /// `--slots`/`--no-slots` is always passed explicitly so the setting never
    /// depends on the llama-server build's own default. See docs/configuration.md.
    ///
    /// `extra_args` is re-validated against [`ALLOWED_EXTRA_FLAGS`] here as well
    /// as at parse time, for callers that build a spec directly.
    pub fn build_args(
        &self,
        gguf_path: &Path,
        bind_host: &str,
        enable_slots_endpoint: bool,
    ) -> Result<Vec<String>, WorkerError> {
        for token in &self.extra_args {
            validate_extra_arg(token)?;
        }
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
        // See `total_ctx()`: this is `n_ctx * parallel`, NOT the raw
        // per-slot `n_ctx`, so llama-server's internal division reproduces
        // `n_ctx` tokens per slot.
        if let Some(total_ctx) = self.total_ctx()? {
            args.push("-c".to_string());
            args.push(total_ctx.to_string());
        }
        // Offload layers to the GPU. `< 0` == "all" (llama.cpp clamps any
        // value >= the real layer count to "all", so 999 is the conventional
        // sentinel); a non-negative value is the partial-offload knob for
        // cards too small to fit every layer.
        args.push("-ngl".to_string());
        args.push(if self.n_gpu_layers < 0 {
            "999".to_string()
        } else {
            self.n_gpu_layers.to_string()
        });
        if let Some(n) = self.n_cpu_moe {
            args.push("--n-cpu-moe".to_string());
            args.push(n.to_string());
        }
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
        args.push(if enable_slots_endpoint {
            "--slots".to_string()
        } else {
            "--no-slots".to_string()
        });
        args.extend(self.extra_args.iter().cloned());
        Ok(args)
    }
}

/// Parse `engine = "llamaserver"` routing metadata.
///
/// * `Ok(None)` — no metadata, or `engine` is absent / not `"llamaserver"`
///   (burn and in-process llamacpp models fall through untouched).
/// * `Ok(Some(spec))` — a complete llamaserver spec.
/// * `Err(..)` — missing `llamaserver.port` / `gguf_repo_id` / `gguf_file`, or a
///   malformed numeric / cache-type value.
///
/// `default_n_gpu_layers` is used when the `n_gpu_layers` metadata key is
/// absent — callers pass the worker's `llamacpp_default_n_gpu_layers` config
/// (shared with the in-process `llamacpp` engine's identical default, since
/// both ultimately drive the same llama.cpp `-ngl`/`n_gpu_layers` knob).
pub fn llamaserver_spec_from_metadata(
    metadata: Option<&HashMap<String, String>>,
    default_n_gpu_layers: i32,
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

    // Same metadata key/semantics as GgufLoadSpec::n_gpu_layers (in-process
    // llamacpp engine) — `-1`/absent means "all layers", which is today's
    // behavior (`-ngl 999`), so existing deployments are unaffected.
    let n_gpu_layers = match metadata.get("n_gpu_layers") {
        Some(v) => v
            .parse::<i32>()
            .map_err(|e| WorkerError::Configuration(format!("invalid n_gpu_layers '{v}': {e}")))?,
        None => default_n_gpu_layers,
    };

    let n_cpu_moe = match metadata.get("n_cpu_moe") {
        Some(v) => Some(
            v.parse::<u32>()
                .map_err(|e| WorkerError::Configuration(format!("invalid n_cpu_moe '{v}': {e}")))?,
        ),
        None => None,
    };

    let extra_args: Vec<String> = metadata
        .get("llamaserver.extra_args")
        .map(|s| s.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default();
    // C2 — validate every token against the flag allowlist BEFORE it can ever
    // reach `Command::new(binary).args(...)`. See `validate_extra_arg`/
    // `ALLOWED_EXTRA_FLAGS` for what is and is not permitted and why.
    for token in &extra_args {
        validate_extra_arg(token)?;
    }

    Ok(Some(LlamaServerSpec {
        repo_id,
        file,
        port,
        n_ctx,
        parallel,
        n_gpu_layers,
        n_cpu_moe,
        cache_type_k,
        cache_type_v,
        extra_args,
    }))
}

/// Reject a `llamaserver.port` outside the configured allowed range (C2
/// defense-in-depth). The typed `port` field always drives the child's
/// `--port`/`--host` — those flag NAMES can never appear in `extra_args`
/// (see [`ALLOWED_EXTRA_FLAGS`]) — but an unconstrained port number could
/// still be pointed at a privileged port or collide with another service on
/// the host, so the coordinator-assigned value is checked against a
/// configured range (`worker.toml`'s `llamaserver_port_min`/`_max`).
pub fn validate_llamaserver_port(port: u16, min: u16, max: u16) -> Result<(), WorkerError> {
    if port < min || port > max {
        return Err(WorkerError::Configuration(format!(
            "llamaserver.port {port} is outside the configured allowed range {min}-{max} \
             (see worker.toml's llamaserver_port_min/llamaserver_port_max)"
        )));
    }
    Ok(())
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

    /// Best-effort post-health sanity check: `GET /props` and compare the
    /// context `llama-server` actually reports against what this load asked
    /// for (`expected_total_ctx`, i.e. [`LlamaServerSpec::total_ctx`]),
    /// logging a WARNING on any mismatch instead of blindly trusting our own
    /// arg-building arithmetic.
    ///
    /// This is exactly the mechanism that caught the original "`-c` silently
    /// divided by `--parallel`" bug on real hardware — `llama-server /props`
    /// reports `n_ctx`/`total_slots` for the process as actually started, so
    /// checking it here makes a future regression (a llama.cpp version that
    /// changes this behavior again, a bad flag, ...) loud instead of silent.
    /// Never fails the load: `/props` is a diagnostic, not a contract, and a
    /// healthy `/health` response already means the model is servable.
    pub async fn verify_props(&self, expected_total_ctx: Option<u32>) {
        let Some(expected_total_ctx) = expected_total_ctx else {
            return;
        };
        let url = format!("http://127.0.0.1:{}/props", self.port);
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                debug!(
                    "llama-server '{}': skipping /props verification (client build failed: {e})",
                    self.model_name
                );
                return;
            }
        };
        let body = match client.get(&url).send().await {
            Ok(resp) => match resp.text().await {
                Ok(text) => text,
                Err(e) => {
                    warn!(
                        "llama-server '{}': /props verification skipped (failed to read response body: {e})",
                        self.model_name
                    );
                    return;
                }
            },
            Err(e) => {
                warn!(
                    "llama-server '{}': /props verification skipped (request failed: {e}) — \
                     cannot confirm the effective per-slot context matches config/models.toml",
                    self.model_name
                );
                return;
            }
        };
        let parsed: serde_json::Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(e) => {
                warn!("llama-server '{}': /props returned unparseable JSON ({e}) — skipping verification", self.model_name);
                return;
            }
        };
        let reported_ctx = parsed.get("n_ctx").and_then(serde_json::Value::as_u64);
        let reported_slots = parsed
            .get("total_slots")
            .and_then(serde_json::Value::as_u64);
        match (reported_ctx, reported_slots) {
            (Some(ctx), Some(slots)) if slots > 0 => {
                let per_slot = ctx / slots;
                if ctx == u64::from(expected_total_ctx) {
                    info!(
                        "llama-server '{}': /props verified — {per_slot} tokens/slot x {slots} slots = {ctx} total context (matches config)",
                        self.model_name
                    );
                } else {
                    warn!(
                        "llama-server '{}': /props reports n_ctx={ctx} across total_slots={slots} \
                         (effective {per_slot} tokens/slot), but config/models.toml expected a total \
                         of {expected_total_ctx} — the model was NOT loaded with the context this \
                         config believes it has; check for a version/flag mismatch",
                        self.model_name
                    );
                }
            }
            _ => debug!(
                "llama-server '{}': /props response missing n_ctx/total_slots — skipping verification",
                self.model_name
            ),
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
        assert_eq!(llamaserver_spec_from_metadata(None, -1).unwrap(), None);
        let empty = meta(&[]);
        assert_eq!(
            llamaserver_spec_from_metadata(Some(&empty), -1).unwrap(),
            None
        );
        // burn / llamacpp models must fall through untouched (Ok(None)).
        let burn = meta(&[("engine", "burn")]);
        assert_eq!(
            llamaserver_spec_from_metadata(Some(&burn), -1).unwrap(),
            None
        );
        let llamacpp = meta(&[("engine", "llamacpp"), ("gguf_repo_id", "x/y")]);
        assert_eq!(
            llamaserver_spec_from_metadata(Some(&llamacpp), -1).unwrap(),
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
            ("llamaserver.extra_args", "--flash-attn --mlock"),
            ("n_ctx", "32768"),
            ("n_gpu_layers", "20"),
            ("n_cpu_moe", "10"),
            ("cache_type_k", "q8_0"),
            ("cache_type_v", "q4_0"),
        ]);
        let spec = llamaserver_spec_from_metadata(Some(&m), -1)
            .unwrap()
            .unwrap();
        assert_eq!(spec.repo_id, "unsloth/Devstral-Small-2-24B-GGUF");
        assert_eq!(spec.file, "devstral-small-2-24b-q4_k_m.gguf");
        assert_eq!(spec.port, 8081);
        assert_eq!(spec.parallel, 8);
        assert_eq!(spec.n_ctx, Some(32768));
        assert_eq!(spec.n_gpu_layers, 20);
        assert_eq!(spec.n_cpu_moe, Some(10));
        assert_eq!(spec.cache_type_k, Some("q8_0".to_string()));
        assert_eq!(spec.cache_type_v, Some("q4_0".to_string()));
        assert_eq!(
            spec.extra_args,
            vec!["--flash-attn".to_string(), "--mlock".to_string()]
        );
    }

    // --- C2: extra_args flag allowlist --------------------------------------

    #[test]
    fn spec_rejects_disallowed_extra_arg_flags() {
        for dangerous in [
            "--path",
            "--log-file",
            "--host",
            "--port",
            "--api-key-file",
            "--api-key",
            "--lora",
            "--lora-scaled",
            "--slot-save-path",
            "--media-path",
            "--models-dir",
            "--models-preset",
            "--hf-token",
            "--hf-repo",
            "--chat-template-file",
            "--ssl-key-file",
            "--ssl-cert-file",
            "-m",  // --model — a second, conflicting model path
            "-c",  // --ctx-size — bypasses the n_ctx/total_ctx arithmetic
            "-np", // --parallel — bypasses the typed `parallel` field
        ] {
            let m = meta(&[
                ("engine", "llamaserver"),
                ("gguf_repo_id", "x/y"),
                ("gguf_file", "m.gguf"),
                ("llamaserver.port", "8080"),
                ("llamaserver.extra_args", dangerous),
            ]);
            let err = llamaserver_spec_from_metadata(Some(&m), -1)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(dangerous),
                "expected rejection of '{dangerous}', got: {err}"
            );
        }
    }

    #[test]
    fn spec_accepts_allowlisted_extra_arg_flags() {
        let m = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "8080"),
            (
                "llamaserver.extra_args",
                "--flash-attn --mlock --no-mmap --threads 8 --rope-scale 2.0",
            ),
        ]);
        let spec = llamaserver_spec_from_metadata(Some(&m), -1)
            .unwrap()
            .unwrap();
        assert_eq!(
            spec.extra_args,
            vec![
                "--flash-attn".to_string(),
                "--mlock".to_string(),
                "--no-mmap".to_string(),
                "--threads".to_string(),
                "8".to_string(),
                "--rope-scale".to_string(),
                "2.0".to_string(),
            ]
        );
    }

    // --- C2: llamaserver.port range constraint ------------------------------

    #[test]
    fn port_range_accepts_in_range() {
        assert!(validate_llamaserver_port(8081, 1024, 65535).is_ok());
        assert!(validate_llamaserver_port(1024, 1024, 65535).is_ok());
        assert!(validate_llamaserver_port(65535, 1024, 65535).is_ok());
    }

    #[test]
    fn port_range_rejects_out_of_range() {
        assert!(validate_llamaserver_port(80, 1024, 65535).is_err());
        assert!(validate_llamaserver_port(1023, 1024, 65535).is_err());
        assert!(validate_llamaserver_port(100, 1024, 8000).is_err());
    }

    #[test]
    fn spec_defaults_parallel_and_optional_fields() {
        let m = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "9000"),
        ]);
        let spec = llamaserver_spec_from_metadata(Some(&m), -1)
            .unwrap()
            .unwrap();
        assert_eq!(spec.parallel, DEFAULT_PARALLEL);
        assert_eq!(spec.n_ctx, None);
        assert_eq!(spec.n_gpu_layers, -1);
        assert_eq!(spec.n_cpu_moe, None);
        assert_eq!(spec.cache_type_k, None);
        assert_eq!(spec.cache_type_v, None);
        assert!(spec.extra_args.is_empty());
    }

    #[test]
    fn spec_uses_caller_provided_default_n_gpu_layers() {
        let m = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "9001"),
        ]);
        let spec = llamaserver_spec_from_metadata(Some(&m), 24)
            .unwrap()
            .unwrap();
        assert_eq!(spec.n_gpu_layers, 24);
    }

    #[test]
    fn spec_rejects_missing_port() {
        let m = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
        ]);
        let err = llamaserver_spec_from_metadata(Some(&m), -1).unwrap_err();
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
        assert!(llamaserver_spec_from_metadata(Some(&m), -1).is_err());
        // out of u16 range
        let m = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "70000"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&m), -1).is_err());
    }

    #[test]
    fn spec_rejects_missing_gguf_source() {
        let no_repo = meta(&[
            ("engine", "llamaserver"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "8080"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&no_repo), -1).is_err());
        let no_file = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("llamaserver.port", "8080"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&no_file), -1).is_err());
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
        assert!(llamaserver_spec_from_metadata(Some(&bad_parallel), -1).is_err());
        let bad_cache = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "8080"),
            ("cache_type_k", "int8"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&bad_cache), -1).is_err());
    }

    #[test]
    fn spec_rejects_bad_n_gpu_layers_or_n_cpu_moe() {
        let bad_ngl = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "8080"),
            ("n_gpu_layers", "many"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&bad_ngl), -1).is_err());
        let bad_moe = meta(&[
            ("engine", "llamaserver"),
            ("gguf_repo_id", "x/y"),
            ("gguf_file", "m.gguf"),
            ("llamaserver.port", "8080"),
            ("n_cpu_moe", "-5"),
        ]);
        assert!(llamaserver_spec_from_metadata(Some(&bad_moe), -1).is_err());
    }

    // --- total_ctx() (n_ctx * parallel) -------------------------------------

    #[test]
    fn total_ctx_multiplies_per_slot_by_parallel() {
        let spec = LlamaServerSpec {
            repo_id: "x/y".to_string(),
            file: "m.gguf".to_string(),
            port: 8086,
            n_ctx: Some(262144),
            parallel: 4,
            n_gpu_layers: -1,
            n_cpu_moe: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: vec![],
        };
        // Matches the hardware-verified qwen3.6-35b-a3b-gguf sizing: 262144
        // per slot x 4 slots = 1048576 total -c.
        assert_eq!(spec.total_ctx().unwrap(), Some(1_048_576));
    }

    #[test]
    fn total_ctx_none_when_n_ctx_unset() {
        let spec = LlamaServerSpec {
            repo_id: "x/y".to_string(),
            file: "m.gguf".to_string(),
            port: 8080,
            n_ctx: None,
            parallel: DEFAULT_PARALLEL,
            n_gpu_layers: -1,
            n_cpu_moe: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: vec![],
        };
        assert_eq!(spec.total_ctx().unwrap(), None);
    }

    #[test]
    fn total_ctx_rejects_overflow_instead_of_wrapping() {
        let spec = LlamaServerSpec {
            repo_id: "x/y".to_string(),
            file: "m.gguf".to_string(),
            port: 8080,
            n_ctx: Some(u32::MAX),
            parallel: 2,
            n_gpu_layers: -1,
            n_cpu_moe: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: vec![],
        };
        assert!(spec.total_ctx().is_err());
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
            n_gpu_layers: -1,
            n_cpu_moe: Some(20),
            cache_type_k: Some("q8_0".to_string()),
            cache_type_v: Some("q4_0".to_string()),
            extra_args: vec!["--flash-attn".to_string(), "--mlock".to_string()],
        };
        let args = spec
            .build_args(Path::new("/models/m.gguf"), "0.0.0.0", false)
            .unwrap();
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
                // total_ctx() = 32768 * 8, not the raw per-slot n_ctx.
                "262144",
                "-ngl",
                "999",
                "--n-cpu-moe",
                "20",
                "--parallel",
                "8",
                "-ctk",
                "q8_0",
                "-ctv",
                "q4_0",
                "--no-slots",
                "--flash-attn",
                "--mlock",
            ]
        );
    }

    #[test]
    fn build_args_enable_slots_endpoint_passes_slots_flag() {
        let spec = LlamaServerSpec {
            repo_id: "x/y".to_string(),
            file: "m.gguf".to_string(),
            port: 8080,
            n_ctx: None,
            parallel: DEFAULT_PARALLEL,
            n_gpu_layers: -1,
            n_cpu_moe: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: vec![],
        };
        let args = spec
            .build_args(Path::new("/m.gguf"), "127.0.0.1", true)
            .unwrap();
        assert!(args.iter().any(|a| a == "--slots"));
        assert!(!args.iter().any(|a| a == "--no-slots"));
    }

    #[test]
    fn build_args_rejects_disallowed_extra_flag_even_if_constructed_directly() {
        // Defense in depth: build_args re-validates extra_args, not just the
        // metadata parser — see the doc comment on ALLOWED_EXTRA_FLAGS.
        let spec = LlamaServerSpec {
            repo_id: "x/y".to_string(),
            file: "m.gguf".to_string(),
            port: 8080,
            n_ctx: None,
            parallel: DEFAULT_PARALLEL,
            n_gpu_layers: -1,
            n_cpu_moe: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: vec!["--path".to_string(), "/etc".to_string()],
        };
        let err = spec
            .build_args(Path::new("/m.gguf"), "127.0.0.1", false)
            .unwrap_err();
        assert!(err.to_string().contains("--path"));
    }

    #[test]
    fn build_args_partial_offload_uses_explicit_n_gpu_layers() {
        let spec = LlamaServerSpec {
            repo_id: "x/y".to_string(),
            file: "m.gguf".to_string(),
            port: 8081,
            n_ctx: None,
            parallel: 1,
            n_gpu_layers: 20,
            n_cpu_moe: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: vec![],
        };
        let args = spec
            .build_args(Path::new("/m.gguf"), "0.0.0.0", false)
            .unwrap();
        // Consumer-GPU partial offload: -ngl 20, not the "all layers" 999
        // sentinel, and no --n-cpu-moe flag since it was not configured.
        assert_eq!(
            args,
            vec![
                "-m",
                "/m.gguf",
                "--host",
                "0.0.0.0",
                "--port",
                "8081",
                "-ngl",
                "20",
                "--parallel",
                "1",
                "--no-slots",
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
            n_gpu_layers: -1,
            n_cpu_moe: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: vec![],
        };
        let args = spec
            .build_args(Path::new("/m.gguf"), "127.0.0.1", false)
            .unwrap();
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
                "--no-slots",
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
