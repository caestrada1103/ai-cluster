//! Model loading and management
//!
//! This module handles loading models from various formats,
//! managing model instances, and coordinating with the GPU manager.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;

use crate::backend::WorkerBackend;
use crate::error::WorkerError;
use crate::gpu_manager::GPUManager;
use crate::llamaserver_process::{llamaserver_spec_from_metadata, LlamaServerProcess};
use crate::models::{
    common::{RMSNormRecord, RotaryEmbeddingRecord},
    deepseek::{
        DeepSeek, DeepSeekAttentionRecord, DeepSeekConfig, DeepSeekLayerRecord, DeepSeekMoERecord,
        DeepSeekRecord, ExpertRecord,
    },
    llama::{
        Llama, LlamaAttentionRecord, LlamaConfig, LlamaLayerRecord, LlamaMLPRecord, LlamaRecord,
    },
    qwen::{Qwen, QwenAttentionRecord, QwenConfig, QwenLayerRecord, QwenMLPRecord, QwenRecord},
    ModelConfig, ModelInstance, TextGeneration,
};
use burn::backend::wgpu::WgpuDevice;
use burn::{
    module::{ConstantRecord, Module, Param, ParamId},
    nn::{EmbeddingRecord, LinearRecord},
    tensor::Tensor,
};
use dashmap::DashMap;
use half::f16;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use safetensors::SafeTensors;
use tokio::sync::Semaphore;
use tracing::{error, info, warn};

/// Resolve a model's attention `head_dim`: an explicit `config.json` value
/// wins; otherwise `hidden_size / num_attention_heads`.
///
/// H3: the previous call site used `config.head_dim.unwrap_or(hidden_size /
/// num_attention_heads)` — `unwrap_or`'s argument is EAGERLY evaluated even
/// when `head_dim` is `Some` and the division is never used, and
/// `num_attention_heads` comes from `load_model_config`'s
/// `json["num_attention_heads"].as_u64().unwrap_or(0)`, i.e. a value an
/// attacker-influenceable `config.json` (a malicious/corrupt HF repo) can
/// set to `0`. That combination made a division-by-zero PANIC reachable from
/// remote input — not just an `Err`. This function only divides when
/// `head_dim` is actually absent AND the divisor is non-zero, turning the
/// invalid-config case into an ordinary, catchable `Err`.
fn resolve_head_dim(config: &ModelConfig) -> Result<usize, WorkerError> {
    if let Some(head_dim) = config.head_dim {
        return Ok(head_dim);
    }
    if config.num_attention_heads == 0 {
        return Err(WorkerError::ModelLoad(
            "config.json reports num_attention_heads = 0 (and no explicit head_dim) — cannot \
             compute head_dim; this repo's config.json is invalid or was tampered with"
                .to_string(),
        ));
    }
    Ok(config.hidden_size / config.num_attention_heads)
}

/// RAII guard releasing GPU memory reservations tracked via [`Self::track`]
/// if dropped without [`Self::commit`] being called first (H3) — including
/// during a PANIC unwind, which the historical "free on the `Err` arm only"
/// pattern elsewhere in this module does not cover: a panic unwinds straight
/// past that `Err` arm, permanently leaking the reservation.
///
/// `GPUManager::free_memory` does no `.await`-suspending work of its own
/// (`DashMap`/`AtomicU64` bookkeeping only — see `gpu_manager.rs`), so
/// running it from a spawned task on `Drop` is safe and fast; `Drop` itself
/// cannot be `async`, hence the `tokio::spawn`.
struct ReservationGuard {
    gpu_manager: Arc<GPUManager>,
    gpu_ids: Vec<usize>,
    tag: String,
    committed: bool,
}

impl ReservationGuard {
    fn new(gpu_manager: Arc<GPUManager>, tag: &str) -> Self {
        Self {
            gpu_manager,
            gpu_ids: Vec::new(),
            tag: tag.to_string(),
            committed: false,
        }
    }

    /// Record that `gpu_id` now holds a reservation under this guard's tag.
    fn track(&mut self, gpu_id: usize) {
        self.gpu_ids.push(gpu_id);
    }

    /// The tracked reservation(s) now belong to something else (a
    /// successfully loaded model) — do not release them on drop.
    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if self.committed || self.gpu_ids.is_empty() {
            return;
        }
        let gpu_manager = self.gpu_manager.clone();
        let gpu_ids = std::mem::take(&mut self.gpu_ids);
        let tag = std::mem::take(&mut self.tag);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    for gpu_id in gpu_ids {
                        gpu_manager.free_memory(gpu_id, &tag).await;
                    }
                });
            }
            Err(_) => {
                // No runtime available (e.g. dropped during process
                // shutdown) — nothing more can be done; log loudly rather
                // than silently leaking.
                error!(
                    "ReservationGuard for '{}' dropped with no tokio runtime available — \
                     GPU reservation(s) on device(s) {:?} were NOT released",
                    tag, gpu_ids
                );
            }
        }
    }
}

/// Model loader configuration
#[derive(Debug, Clone)]
pub struct ModelLoaderConfig {
    pub cache_dir: PathBuf,
    pub download_dir: PathBuf,
    pub max_concurrent_loads: usize,
    /// HuggingFace token for gated repos. Env var HF_TOKEN wins over this.
    pub hf_token: Option<String>,
    /// Override for the HF hub cache directory (defaults to cache_dir).
    pub hf_cache_dir: Option<PathBuf>,
    /// CPU threads for llama.cpp generation (0 = auto). See config/worker.toml.
    pub llamacpp_n_threads: i32,
    /// Default GPU layer offload for llama.cpp models (-1 = all layers).
    pub llamacpp_default_n_gpu_layers: i32,
    /// Path to the `llama-server` binary spawned for `engine = "llamaserver"`
    /// models (env `LLAMASERVER_BINARY_PATH` wins; default `"llama-server"`).
    pub llamaserver_binary_path: String,
    /// Bind interface passed to `llama-server --host` (default `"0.0.0.0"`).
    pub llamaserver_bind_host: String,
    /// Seconds to poll `llama-server`'s `/health` before declaring the load
    /// failed — reuses the worker's `request_timeout_secs`.
    pub llamaserver_health_timeout_secs: u64,
    /// Maximum number of models kept resident at once (0 = unlimited). See
    /// `WorkerConfig::max_loaded_models` in config/worker.toml.
    pub max_loaded_models: usize,
    /// Ceiling for any caller-supplied per-slot `n_ctx` (H2). See
    /// `WorkerConfig::max_n_ctx`.
    pub max_n_ctx: u32,
    /// Worker's `max_concurrent_requests` — used to size the in-process
    /// `llamacpp` engine's KV-cache reservation (H2).
    pub max_concurrent_requests: usize,
    /// Whether spawned `llama-server` children expose `/slots` (H1). See
    /// `WorkerConfig::llamaserver_enable_slots_endpoint`.
    pub llamaserver_enable_slots_endpoint: bool,
    /// Allowed `llamaserver.port` range, inclusive (C2).
    pub llamaserver_port_min: u16,
    pub llamaserver_port_max: u16,
}

impl Default for ModelLoaderConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from("./models"),
            download_dir: PathBuf::from("./data/downloads"),
            max_concurrent_loads: 2,
            hf_token: None,
            hf_cache_dir: None,
            llamacpp_n_threads: 0,
            llamacpp_default_n_gpu_layers: -1,
            llamaserver_binary_path: "llama-server".to_string(),
            llamaserver_bind_host: "127.0.0.1".to_string(),
            llamaserver_health_timeout_secs: 120,
            max_loaded_models: 0,
            max_n_ctx: 262_144,
            max_concurrent_requests: 32,
            llamaserver_enable_slots_endpoint: false,
            llamaserver_port_min: 1024,
            llamaserver_port_max: 65535,
        }
    }
}

/// Model loader
pub struct ModelLoader {
    gpu_manager: Arc<GPUManager>,
    loaded_models: Arc<DashMap<String, ModelInstance>>,
    /// Serializes loads of the SAME model (the global semaphore only limits cross-model concurrency).
    loading_locks: Arc<DashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    load_semaphore: Arc<Semaphore>,
    hf_api: Option<Api>,
    /// CPU threads for llama.cpp generation (0 = auto).
    #[allow(dead_code)] // read only when built with --features llamacpp
    llamacpp_n_threads: i32,
    /// Default GPU layer offload for llama.cpp models (-1 = all).
    llamacpp_default_n_gpu_layers: i32,
    /// Supervised `llama-server` child processes, keyed by model name. Kept
    /// separate from `loaded_models` because the child handle (not a
    /// `TextGeneration`) must be reachable for kill-on-unload and liveness.
    llamaserver_processes: Arc<DashMap<String, Arc<tokio::sync::Mutex<LlamaServerProcess>>>>,
    /// Models evicted internally by the `max_loaded_models` policy, awaiting
    /// pickup by the gRPC layer.
    ///
    /// `worker.rs` keeps its OWN `loaded_models` map, and only its gRPC
    /// `unload_model` handler clears both. An eviction happens *inside*
    /// `load_model`, so without this queue `worker.rs` would keep a stale
    /// entry — reporting the model as loaded and, worse for the in-process
    /// `llamacpp` engine, holding the last `Arc` to the model so its weights
    /// would never be freed. Drained by [`ModelLoader::take_evicted`].
    /// `tokio::sync::Mutex` (not the `std::sync::Mutex` imported above) because
    /// it is locked from async code across the eviction path.
    evicted_pending: Arc<tokio::sync::Mutex<Vec<String>>>,
    /// Path to the `llama-server` binary (env/config resolved).
    llamaserver_binary_path: String,
    /// `--host` bind interface for spawned `llama-server` processes.
    llamaserver_bind_host: String,
    /// `/health` poll timeout (seconds) for a spawned `llama-server`.
    llamaserver_health_timeout_secs: u64,
    /// Maximum number of models kept resident at once (0 = unlimited). When a
    /// load would exceed this, `load_model` evicts the oldest-loaded model(s)
    /// first via `unload`.
    max_loaded_models: usize,
    /// H7 — loader-wide lock serializing the "evict victim(s) then insert"
    /// decision across CONCURRENT loads of different model names. See the
    /// call sites ([`Self::load_model`]'s pre-load pass and
    /// [`Self::evict_to_fit_and_insert`]) for why two phases are needed.
    eviction_lock: Arc<tokio::sync::Mutex<()>>,
    /// Ceiling applied to any caller-supplied per-slot `n_ctx` before it
    /// sizes a KV-cache reservation or reaches `llama-server -c` (H2).
    max_n_ctx: u32,
    /// Slots used for the in-process `llamacpp` engine's KV-cache
    /// reservation (H2) — the worker's `max_concurrent_requests`, since the
    /// in-process engine builds one full-`n_ctx` context PER concurrent
    /// request, not one total.
    #[allow(dead_code)] // read only when built with --features llamacpp
    max_concurrent_requests: usize,
    /// Whether spawned `llama-server` children expose `/slots` (H1).
    llamaserver_enable_slots_endpoint: bool,
    /// Allowed `llamaserver.port` range, inclusive (C2).
    llamaserver_port_min: u16,
    llamaserver_port_max: u16,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(
        config: ModelLoaderConfig,
        gpu_manager: Arc<GPUManager>,
    ) -> Result<Self, WorkerError> {
        let cache_dir = config
            .hf_cache_dir
            .clone()
            .unwrap_or_else(|| config.cache_dir.clone());
        let mut builder = hf_hub::api::tokio::ApiBuilder::new()
            .with_endpoint("https://huggingface.co".to_string())
            .with_cache_dir(cache_dir);

        // Env var wins; TOML hf_token is the fallback for gated repos.
        let token = std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| config.hf_token.clone());
        if let Some(token) = token {
            builder = builder.with_token(Some(token));
        }

        let hf_api = builder.build().ok();

        std::fs::create_dir_all(&config.cache_dir)?;
        std::fs::create_dir_all(&config.download_dir)?;

        Ok(Self {
            gpu_manager,
            loaded_models: Arc::new(DashMap::new()),
            loading_locks: Arc::new(DashMap::new()),
            load_semaphore: Arc::new(Semaphore::new(config.max_concurrent_loads)),
            hf_api,
            llamacpp_n_threads: config.llamacpp_n_threads,
            llamacpp_default_n_gpu_layers: config.llamacpp_default_n_gpu_layers,
            llamaserver_processes: Arc::new(DashMap::new()),
            evicted_pending: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            llamaserver_binary_path: config.llamaserver_binary_path,
            llamaserver_bind_host: config.llamaserver_bind_host,
            llamaserver_health_timeout_secs: config.llamaserver_health_timeout_secs,
            max_loaded_models: config.max_loaded_models,
            eviction_lock: Arc::new(tokio::sync::Mutex::new(())),
            max_n_ctx: config.max_n_ctx,
            max_concurrent_requests: config.max_concurrent_requests,
            llamaserver_enable_slots_endpoint: config.llamaserver_enable_slots_endpoint,
            llamaserver_port_min: config.llamaserver_port_min,
            llamaserver_port_max: config.llamaserver_port_max,
        })
    }

    /// Evict the oldest-loaded model(s) (excluding `exclude`) until
    /// `loaded_models.len() < max_loaded_models`, or until there is no more
    /// eligible victim. No-op when `max_loaded_models == 0` (unlimited).
    /// Callers MUST hold `eviction_lock` — see [`Self::load_model`]'s
    /// pre-load pass and [`Self::evict_to_fit_and_insert`] (H7).
    async fn evict_to_fit(&self, exclude: &str) {
        while self.max_loaded_models > 0 && self.loaded_models.len() >= self.max_loaded_models {
            // Find the oldest-loaded entry that isn't the model we're about
            // to load. Collect the victim's name into an owned String and
            // drop the DashMap iterator before awaiting `unload` — holding a
            // DashMap ref/iterator across an `.await` on another shard can
            // deadlock.
            let victim = self
                .loaded_models
                .iter()
                .filter(|entry| entry.key() != exclude)
                .min_by_key(|entry| entry.value().loaded_at())
                .map(|entry| entry.key().clone());

            let Some(victim) = victim else {
                // No eviction candidate (e.g. every remaining entry is the
                // target itself) — break rather than spinning forever.
                break;
            };

            info!(
                "Evicting {} (max_loaded_models = {}) to make room for {}",
                victim, self.max_loaded_models, exclude
            );
            self.unload(&victim).await;
            // Hand the name to the gRPC layer so it drops its own copy too.
            self.evicted_pending.lock().await.push(victim);
        }
    }

    /// Atomically make room for (if needed) and register `instance` under
    /// `model_name` (H7).
    ///
    /// This — not the pre-load pass in [`Self::load_model`] — is what
    /// actually guarantees `loaded_models.len() <= max_loaded_models`: it
    /// re-evaluates the cap and evicts again if necessary, then inserts,
    /// all under ONE acquisition of the loader-wide `eviction_lock`. Two
    /// concurrent loads calling this for different model names are fully
    /// serialized against EACH OTHER here, so the second one to run always
    /// sees the first one's insert and evicts accordingly — the exact race
    /// (`both evict the same victim, both insert, cap exceeded`) the
    /// pre-load-only version allowed.
    ///
    /// The heavy download/build work happens BETWEEN the pre-load pass and
    /// this call, deliberately outside any lock, so a resident cap does not
    /// serialize unrelated models' multi-GB downloads against each other —
    /// only the cheap evict+insert bookkeeping is serialized.
    async fn evict_to_fit_and_insert(&self, model_name: &str, instance: ModelInstance) {
        let _evict_guard = self.eviction_lock.lock().await;
        self.evict_to_fit(model_name).await;
        self.loaded_models.insert(model_name.to_string(), instance);
    }

    /// Load a model
    pub async fn load_model(
        &self,
        model_name: &str,
        repo_override: Option<&str>,
        model_config: Option<&crate::cluster::ModelConfig>,
        gpu_ids: &[u32],
        quantization: crate::cluster::Quantization,
        parallelism: crate::cluster::ParallelismStrategy,
    ) -> Result<ModelInstance, WorkerError> {
        // model_name is the registry key (map key everywhere); repo_id is what we download.
        let repo_id: &str = repo_override
            .filter(|s| !s.is_empty())
            .unwrap_or(model_name);

        if let Some(entry) = self.loaded_models.get(model_name) {
            info!("Model {} already loaded", model_name);
            return Ok(entry.value().clone());
        }

        if quantization != crate::cluster::Quantization::None {
            return Err(WorkerError::InvalidRequest(format!(
                "quantization {:?} is not implemented — request NONE (weights load as FP32)",
                quantization
            )));
        }

        // Per-model lock: two concurrent requests for the SAME model serialize here,
        // so exactly one performs the multi-GB load.
        let model_lock = self
            .loading_locks
            .entry(model_name.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone();
        let _model_guard = model_lock.lock().await;

        if let Some(entry) = self.loaded_models.get(model_name) {
            info!("Model {} loaded by concurrent task", model_name);
            return Ok(entry.value().clone());
        }

        // Resident-model cap: evict the oldest-loaded model(s) to make room
        // before starting this load. `max_loaded_models == 0` means
        // unlimited (no eviction, preserves prior behavior).
        //
        // H7: this pre-load pass is guarded by `eviction_lock` — a
        // loader-WIDE lock, unlike `_model_guard` above (which only
        // serializes loads of THIS SAME model name). Without it, two
        // concurrent loads of DIFFERENT model names each pass their own
        // `_model_guard`, both observe the cap as full under their own
        // independent view, both pick (and evict) the SAME victim, and both
        // then insert — exceeding `max_loaded_models`. This pass exists so
        // the victim's GPU memory is freed BEFORE this load's own
        // `allocate_memory` call runs (needed for the load to even fit);
        // the AUTHORITATIVE, race-proof check happens again immediately
        // before every `loaded_models.insert` below, in
        // [`Self::evict_to_fit_and_insert`] — see that method's doc for why
        // this two-phase split (rather than one lock held across the whole,
        // possibly multi-GB, load) is still race-free.
        {
            let _evict_guard = self.eviction_lock.lock().await;
            self.evict_to_fit(model_name).await;
        }

        let _permit =
            self.load_semaphore.acquire().await.map_err(|e| {
                WorkerError::Resource(format!("Failed to acquire load permit: {}", e))
            })?;

        // Level-2 distributed routing: an rpc_server peer reserves VRAM for
        // a lead elsewhere and holds no directly-servable model — checked
        // BEFORE gguf/Burn routing so a peer's metadata never has to carry a
        // (fake) GGUF source. STUB this increment: no subprocess/real
        // ggml-RPC yet (Task 8 in pending-work/11-distributed-multi-node-inference.md).
        // A "lead" spec falls through to the normal routing below unchanged
        // this increment (Task 9 wires the real proxy engine).
        if let Some(dist_spec) = distributed_spec_from_metadata(model_config.map(|c| &c.metadata))?
        {
            if dist_spec.role == DistributedRole::RpcServer {
                return self
                    .load_rpc_server_stub(model_name, gpu_ids, quantization, parallelism)
                    .await;
            }
        }

        // Engine routing: `llamaserver` models supervise a `llama-server`
        // child process (Plan 13). Checked BEFORE the gguf branch below —
        // `gguf_spec_from_metadata` errors on an unknown engine string like
        // "llamaserver", so this must intercept it first.
        if let Some(spec) = llamaserver_spec_from_metadata(
            model_config.map(|c| &c.metadata),
            self.llamacpp_default_n_gpu_layers,
        )? {
            return self
                .load_llamaserver_model(model_name, spec, gpu_ids, quantization, parallelism)
                .await;
        }

        // Engine routing: GGUF/llama.cpp models bypass the entire
        // safetensors + config.json path below (GGUF repos often ship no
        // config.json; llama.cpp reads architecture from GGUF metadata).
        if let Some(mut spec) = gguf_spec_from_metadata(
            model_config.map(|c| &c.metadata),
            self.llamacpp_default_n_gpu_layers,
        )? {
            // H2: clamp any caller-supplied n_ctx before it can size the
            // KV-cache reservation or reach the loaded context.
            spec.n_ctx = spec.n_ctx.map(|n| n.min(self.max_n_ctx));
            return self
                .load_llamacpp_model(model_name, spec, gpu_ids, quantization, parallelism)
                .await;
        }

        info!("Loading model {}...", model_name);
        let model_path = self.get_model_path(repo_id).await?;
        let config = self
            .load_model_config(model_name, model_config, &model_path)
            .await?;

        let memory_used = self.calculate_memory_usage(&config);
        // H3: a `ReservationGuard`, not manual match/rollback bookkeeping —
        // its `Drop` releases every reservation tracked so far if this
        // function returns (via `?`) OR panics before `commit()` runs. That
        // second case matters here specifically: `head_dim` below used to be
        // computed with an EAGERLY-evaluated `unwrap_or(hidden_size /
        // num_attention_heads)`, and `num_attention_heads` comes from
        // `config.json` (`as_u64().unwrap_or(0)` in `load_model_config`) —
        // an attacker-influenceable value a malicious/corrupt repo could set
        // to `0`, causing a division-by-zero PANIC that unwound straight
        // past the old `Err`-arm-only rollback below, permanently leaking
        // the reservation on every occurrence. `resolve_head_dim` below
        // turns that into an ordinary `Err`; this guard is the second half
        // of the fix — it also covers any OTHER panic in the build path
        // (e.g. a Burn tensor shape mismatch), not just this one bug.
        let mut reservation = ReservationGuard::new(self.gpu_manager.clone(), model_name);
        for &gpu_id in gpu_ids {
            self.gpu_manager
                .allocate_memory(gpu_id as usize, memory_used as u64, model_name)
                .await?;
            reservation.track(gpu_id as usize);
        }

        let build_result: Result<Arc<Mutex<dyn TextGeneration + Send>>, WorkerError> = async {
            let device = if let Some(&id) = gpu_ids.first() {
                WgpuDevice::DiscreteGpu(id as usize)
            } else {
                // WgpuDevice::default() is the modern replacement for the deprecated BestAvailable;
                // it selects the best adapter the wgpu backend can find (DX12/Vulkan/Metal).
                WgpuDevice::default()
            };

            // Load weights
            let mut weights = self.load_safetensors(repo_id, &device).await?;

            // Get model directory (tokenizer.json lives here)
            let model_path = self.get_model_path(repo_id).await?;
            // Ensure tokenizer.json is downloaded
            if let Some(api) = &self.hf_api {
                let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
                let _ = repo.get("tokenizer.json").await.map_err(|e| {
                    WorkerError::ModelLoad(format!("Failed to download tokenizer.json: {}", e))
                })?;
                info!("Tokenizer downloaded to: {:?}", model_path);

                // generation_config.json carries the authoritative eos_token_id for many repos.
                // Best-effort: absent for some models, config.json then supplies it.
                let _ = repo.get("generation_config.json").await;
            }

            // Instantiate model
            let model: Arc<Mutex<dyn TextGeneration + Send>> = match config.architecture.as_str() {
                "llama" => {
                    let llama_config = LlamaConfig {
                        hidden_size: config.hidden_size,
                        num_layers: config.num_layers,
                        num_attention_heads: config.num_attention_heads,
                        num_kv_heads: config.num_kv_heads,
                        head_dim: resolve_head_dim(&config)?,
                        intermediate_size: config.intermediate_size,
                        vocab_size: config.vocab_size,
                        max_seq_len: config.max_seq_len,
                        rms_norm_eps: config.rms_norm_eps,
                        rope_theta: config.rope_theta,
                    };

                    info!("Mapping weights to LlamaRecord...");
                    let record = create_llama_record(&mut weights, &llama_config)?;
                    info!("Record created. Initializing Llama...");

                    let model =
                        Llama::new(&llama_config, &device, &model_path)?.load_record(record);
                    Arc::new(Mutex::new(model))
                }
                "qwen" => {
                    let qwen_config = QwenConfig {
                        hidden_size: config.hidden_size,
                        num_layers: config.num_layers,
                        num_attention_heads: config.num_attention_heads,
                        num_kv_heads: config.num_kv_heads,
                        head_dim: resolve_head_dim(&config)?,
                        intermediate_size: config.intermediate_size,
                        vocab_size: config.vocab_size,
                        max_seq_len: config.max_seq_len,
                        rms_norm_eps: config.rms_norm_eps,
                        rope_theta: config.rope_theta,
                        attention_bias: config.attention_bias,
                    };

                    info!("Mapping weights to QwenRecord...");
                    let record = create_qwen_record(&mut weights, &qwen_config)?;
                    info!("Record created. Initializing Qwen...");

                    let model = Qwen::new(&qwen_config, &device, &model_path)?.load_record(record);
                    Arc::new(Mutex::new(model))
                }
                "deepseek" => {
                    let head_dim = resolve_head_dim(&config)?;
                    let has_expert_weights =
                        weights.contains_key("model.layers.0.mlp.experts.0.gate_proj.weight");

                    if !has_expert_weights {
                        // Dense DeepSeek checkpoints (deepseek-llm-7b/67b) use the Llama layout.
                        info!("DeepSeek checkpoint has no expert weights — loading via Llama path");
                        let llama_config = LlamaConfig {
                            hidden_size: config.hidden_size,
                            num_layers: config.num_layers,
                            num_attention_heads: config.num_attention_heads,
                            num_kv_heads: config.num_kv_heads,
                            head_dim,
                            intermediate_size: config.intermediate_size,
                            vocab_size: config.vocab_size,
                            max_seq_len: config.max_seq_len,
                            rms_norm_eps: config.rms_norm_eps,
                            rope_theta: config.rope_theta,
                        };
                        let record = create_llama_record(&mut weights, &llama_config)?;
                        let model =
                            Llama::new(&llama_config, &device, &model_path)?.load_record(record);
                        Arc::new(Mutex::new(model))
                    } else {
                        // MoE checkpoint: build the config from config.json, not name substrings.
                        let ds_config = DeepSeekConfig {
                            hidden_size: config.hidden_size,
                            num_layers: config.num_layers,
                            num_attention_heads: config.num_attention_heads,
                            num_kv_heads: config.num_kv_heads,
                            head_dim,
                            intermediate_size: config.intermediate_size,
                            vocab_size: config.vocab_size,
                            max_seq_len: config.max_seq_len,
                            rms_norm_eps: config.rms_norm_eps,
                            rope_theta: config.rope_theta,
                            num_experts: config.num_experts.unwrap_or(1),
                            num_experts_per_tok: config.num_experts_per_tok.unwrap_or(1),
                        };
                        info!("Mapping weights to DeepSeekRecord...");
                        let record = create_deepseek_record(&mut weights, &ds_config)?;
                        info!("Record created. Initializing DeepSeek...");
                        let model =
                            DeepSeek::new(ds_config, &device, &model_path)?.load_record(record);
                        Arc::new(Mutex::new(model))
                    }
                }
                other => {
                    return Err(WorkerError::ModelLoad(format!(
                        "Unsupported architecture: {}",
                        other
                    )))
                }
            };
            Ok(model)
        }
        .await;

        // `?` here relies on `reservation`'s `Drop` for cleanup on the `Err`
        // path (and, unlike the manual match this replaces, on a panic
        // unwind too) — see the guard's construction above for why that
        // matters.
        let model = build_result?;

        let instance = ModelInstance::new(
            model_name.to_string(),
            memory_used,
            gpu_ids.to_vec(),
            quantization as i32,
            parallelism as i32,
            Some(model),
        );
        // Ownership of the reservation transfers to the now-loaded instance
        // — do not release it on drop.
        reservation.commit();

        self.evict_to_fit_and_insert(model_name, instance.clone())
            .await;
        info!("Model {} loaded successfully", model_name);

        Ok(instance)
    }

    /// Register this node as an `rpc_server` peer for a distributed model —
    /// STUB (Task 5 of pending-work/11-distributed-multi-node-inference.md).
    ///
    /// An rpc_server peer lends its whole local GPU(s) to a lead node
    /// elsewhere, so this reserves everything currently free on each
    /// requested device (an honest "this GPU is spoken for" signal even
    /// though no real `rpc-server` process is running yet) and inserts a
    /// [`ModelInstance`] with no attached model — the existing "no model
    /// attached" placeholder also used elsewhere. No subprocess, no real
    /// ggml-RPC this increment; Task 8 replaces this with the real
    /// `rpc-server` supervisor + a synthetic allocation sized to what that
    /// process actually consumes.
    async fn load_rpc_server_stub(
        &self,
        model_name: &str,
        gpu_ids: &[u32],
        quantization: crate::cluster::Quantization,
        parallelism: crate::cluster::ParallelismStrategy,
    ) -> Result<ModelInstance, WorkerError> {
        info!(
            "Registering {} as an rpc_server peer stub on GPU(s) {:?} (no subprocess yet)",
            model_name, gpu_ids
        );

        let mut reserved_gpus: Vec<usize> = Vec::new();
        let mut total_reserved: u64 = 0;
        for &gpu_id in gpu_ids {
            let gpu_id = gpu_id as usize;
            let available = self.gpu_manager.get_available_memory(gpu_id).await;
            match self
                .gpu_manager
                .allocate_memory(gpu_id, available, model_name)
                .await
            {
                Ok(()) => {
                    reserved_gpus.push(gpu_id);
                    total_reserved += available;
                }
                Err(e) => {
                    // Roll back reservations made so far — no leak on partial failure.
                    for &g in &reserved_gpus {
                        self.gpu_manager.free_memory(g, model_name).await;
                    }
                    return Err(e);
                }
            }
        }

        // `model: None` — the existing "no model attached" placeholder. A
        // stray Infer against this instance hits `ModelInstance::generate`'s
        // `Err(WorkerError::Internal(...))` path, not a panic.
        let instance = ModelInstance::new(
            model_name.to_string(),
            total_reserved as usize,
            gpu_ids.to_vec(),
            quantization as i32,
            parallelism as i32,
            None,
        );

        self.evict_to_fit_and_insert(model_name, instance.clone())
            .await;
        info!(
            "rpc_server peer stub {} registered ({} bytes reserved across {} GPU(s))",
            model_name,
            total_reserved,
            reserved_gpus.len()
        );
        Ok(instance)
    }

    /// Supervise a `llama-server` child process for a `engine = "llamaserver"`
    /// model (Plan 13). Pure process management — no in-process weights, no
    /// `TextGeneration`; agentic inference is proxied to the child over HTTP by
    /// the coordinator, so the [`ModelInstance`] carries `model: None`.
    ///
    /// Flow: resolve/download the GGUF (same hf-hub path as llama.cpp) → reserve
    /// its file size as an honest VRAM estimate → spawn `llama-server` → poll
    /// `/health` until 200 or the load timeout. Any failure kills the child and
    /// rolls back the GPU reservations.
    async fn load_llamaserver_model(
        &self,
        model_name: &str,
        mut spec: crate::llamaserver_process::LlamaServerSpec,
        gpu_ids: &[u32],
        quantization: crate::cluster::Quantization,
        parallelism: crate::cluster::ParallelismStrategy,
    ) -> Result<ModelInstance, WorkerError> {
        // C2: constrain the coordinator-assigned port to the configured
        // range, and H2: clamp any caller-supplied per-slot n_ctx before it
        // drives `-c`/the KV-cache footprint. Both fail fast, before any
        // download/spawn work.
        crate::llamaserver_process::validate_llamaserver_port(
            spec.port,
            self.llamaserver_port_min,
            self.llamaserver_port_max,
        )?;
        spec.n_ctx = spec.n_ctx.map(|n| n.min(self.max_n_ctx));

        info!(
            "Loading model {} via llama-server (port {}, {}/{})",
            model_name, spec.port, spec.repo_id, spec.file
        );

        // Resolve/download the GGUF (plus any sibling shards) — identical
        // hf-hub pattern to llama.cpp.
        let api = self
            .hf_api
            .as_ref()
            .ok_or_else(|| WorkerError::ModelLoad("HF API not initialized".to_string()))?;
        let repo = api.repo(Repo::new(spec.repo_id.clone(), RepoType::Model));
        let (gguf_path, file_size) = download_gguf(&repo, &spec.file).await?;

        // Log the effective per-slot / total context BEFORE spawning, so a
        // misconfiguration is loud in the logs even if `/props` verification
        // below is inconclusive (process not reachable yet, etc.). See the
        // `LlamaServerSpec::n_ctx`/`total_ctx` docs for why this multiply
        // exists: llama-server divides its own `-c` evenly across
        // `--parallel` slots, so the registry's `n_ctx` must be multiplied by
        // `parallel` here to keep meaning "tokens per conversation".
        match spec.total_ctx()? {
            Some(total) => info!(
                "llama-server {}: n_ctx={} tokens/slot x parallel={} slots = {} total context (-c)",
                model_name,
                spec.n_ctx.unwrap_or_default(),
                spec.parallel,
                total
            ),
            None => info!(
                "llama-server {}: no n_ctx configured — using llama-server's own default context \
                 (per-slot value cannot be verified from config alone)",
                model_name
            ),
        }

        let mut reserved_gpus: Vec<usize> = Vec::new();
        for &gpu_id in gpu_ids {
            match self
                .gpu_manager
                .allocate_memory(gpu_id as usize, file_size, model_name)
                .await
            {
                Ok(()) => reserved_gpus.push(gpu_id as usize),
                Err(e) => {
                    for &g in &reserved_gpus {
                        self.gpu_manager.free_memory(g, model_name).await;
                    }
                    return Err(e);
                }
            }
        }

        // Build argv + spawn. On failure roll back the reservations.
        let args = match spec.build_args(
            &gguf_path,
            &self.llamaserver_bind_host,
            self.llamaserver_enable_slots_endpoint,
        ) {
            Ok(args) => args,
            Err(e) => {
                for &g in &reserved_gpus {
                    self.gpu_manager.free_memory(g, model_name).await;
                }
                return Err(e);
            }
        };
        let spawn_res =
            LlamaServerProcess::spawn(model_name, spec.port, &self.llamaserver_binary_path, &args);
        let mut process = match spawn_res {
            Ok(p) => p,
            Err(e) => {
                for &g in &reserved_gpus {
                    self.gpu_manager.free_memory(g, model_name).await;
                }
                return Err(e);
            }
        };

        // Loaded == GET /health returns 200 within the load timeout. On failure
        // kill the child and roll back.
        let health_timeout = std::time::Duration::from_secs(self.llamaserver_health_timeout_secs);
        if let Err(e) = process.wait_until_healthy(health_timeout).await {
            let _ = process.start_kill();
            for &g in &reserved_gpus {
                self.gpu_manager.free_memory(g, model_name).await;
            }
            return Err(e);
        }

        // Best-effort: cross-check the context llama-server actually started
        // with against what this load computed, so a future regression in
        // this arithmetic (or in llama-server's own slot-division behavior)
        // is loud in the logs rather than silently wrong again.
        process.verify_props(spec.total_ctx()?).await;

        // Register the supervised child + a model-less instance (inference goes
        // through the coordinator HTTP proxy, not this worker's gRPC Infer).
        let handle = Arc::new(tokio::sync::Mutex::new(process));
        self.llamaserver_processes
            .insert(model_name.to_string(), handle);

        let instance = ModelInstance::new(
            model_name.to_string(),
            file_size as usize,
            gpu_ids.to_vec(),
            quantization as i32,
            parallelism as i32,
            None,
        );
        self.evict_to_fit_and_insert(model_name, instance.clone())
            .await;
        info!(
            "Model {} loaded via llama-server on port {}",
            model_name, spec.port
        );
        Ok(instance)
    }

    /// Whether `model_name` is served by a supervised `llama-server` child
    /// (so worker gRPC Infer must reject it with a FAILED_PRECONDITION).
    pub fn is_llamaserver(&self, model_name: &str) -> bool {
        self.llamaserver_processes.contains_key(model_name)
    }

    /// Reap `llama-server` children that exited on their own: kill nothing (they
    /// are already gone), drop them from both the process map and
    /// `loaded_models`, and free their GPU reservations. Returns the reaped
    /// model names so callers can prune their own view too. Keeps status honest
    /// ("report as not loaded" once the child dies) without a background task.
    /// Drain the names evicted by the `max_loaded_models` policy since the last
    /// call. The caller (`worker.rs`) must remove each from its own
    /// `loaded_models` map and clear the model's metrics — see
    /// [`ModelLoader::evicted_pending`] for why that second map exists.
    pub async fn take_evicted(&self) -> Vec<String> {
        std::mem::take(&mut *self.evicted_pending.lock().await)
    }

    pub async fn reap_exited_llamaservers(&self) -> Vec<String> {
        let names: Vec<String> = self
            .llamaserver_processes
            .iter()
            .map(|e| e.key().clone())
            .collect();
        let mut reaped = Vec::new();
        for name in names {
            // Clone the Arc out and drop the DashMap ref before awaiting.
            let proc = self
                .llamaserver_processes
                .get(&name)
                .map(|r| r.value().clone());
            let dead = match proc {
                Some(p) => !p.lock().await.is_running(),
                None => continue,
            };
            if dead {
                self.llamaserver_processes.remove(&name);
                if let Some((_, instance)) = self.loaded_models.remove(&name) {
                    for &gpu_id in instance.gpu_ids() {
                        self.gpu_manager.free_memory(gpu_id as usize, &name).await;
                    }
                }
                warn!(
                    "llama-server for {} exited on its own — marked unloaded",
                    name
                );
                reaped.push(name);
            }
        }
        reaped
    }

    /// Remove a model from the registry and release its GPU memory reservations.
    /// Returns true when the model was loaded.
    pub async fn unload(&self, model_name: &str) -> bool {
        // Kill a supervised llama-server child first, if this model is one.
        let killed_server = if let Some((_, proc)) = self.llamaserver_processes.remove(model_name) {
            if let Err(e) = proc.lock().await.shutdown().await {
                warn!("error shutting down llama-server for {}: {}", model_name, e);
            }
            info!("Unload {}: llama-server child terminated", model_name);
            true
        } else {
            false
        };

        let removed = self.loaded_models.remove(model_name);
        if let Some((_, instance)) = &removed {
            for &gpu_id in instance.gpu_ids() {
                let freed = self
                    .gpu_manager
                    .free_memory(gpu_id as usize, model_name)
                    .await;
                info!(
                    "Unload {}: freed {} bytes on GPU {}",
                    model_name, freed, gpu_id
                );
            }
        }
        // instance drops here — last Arc clone (worker.rs removed its copy first) frees the weights.
        removed.is_some() || killed_server
    }

    async fn load_safetensors(
        &self,
        model_name: &str,
        device: &WgpuDevice,
    ) -> Result<HashMap<String, Tensor<WorkerBackend, 1>>, WorkerError> {
        let api = self
            .hf_api
            .as_ref()
            .ok_or(WorkerError::ModelLoad("HF API not initialized".to_string()))?;
        let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));

        let mut weights = HashMap::new();
        let mut files = Vec::new();

        // Check for index (sharded) or single file
        if let Ok(index_path) = repo.get("model.safetensors.index.json").await {
            let index_content = tokio::fs::read_to_string(&index_path)
                .await
                .map_err(|e| WorkerError::ModelLoad(format!("Failed to read index: {}", e)))?;
            let json: serde_json::Value = serde_json::from_str(&index_content)
                .map_err(|e| WorkerError::ModelLoad(format!("Json error: {}", e)))?;

            if let Some(map) = json["weight_map"].as_object() {
                let mut filenames: Vec<String> = map
                    .values()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                filenames.sort();
                filenames.dedup();
                for fname in filenames {
                    files.push(
                        repo.get(&fname)
                            .await
                            .map_err(|e| WorkerError::ModelLoad(e.to_string()))?,
                    );
                }
            }
        } else if let Ok(path) = repo.get("model.safetensors").await {
            files.push(path);
        } else {
            return Err(WorkerError::ModelLoad("No safetensors found".to_string()));
        }

        info!("Loading {} safetensors files...", files.len());

        for file in files {
            // Async read keeps the runtime free during I/O
            let data = tokio::fs::read(&file)
                .await
                .map_err(|e| WorkerError::ModelLoad(e.to_string()))?;

            // CPU-heavy deserialization and dtype conversion run on the blocking thread pool
            let parsed: Vec<(String, Vec<f32>)> = tokio::task::spawn_blocking(move || {
                let safetensors = SafeTensors::deserialize(&data)
                    .map_err(|e| WorkerError::ModelLoad(e.to_string()))?;
                let mut out: Vec<(String, Vec<f32>)> = Vec::new();
                for (name, view) in safetensors.tensors() {
                    let floats: Vec<f32> = match view.dtype() {
                        safetensors::Dtype::F16 => view
                            .data()
                            .chunks(2)
                            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                            .collect(),
                        safetensors::Dtype::BF16 => view
                            .data()
                            .chunks(2)
                            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                            .collect(),
                        safetensors::Dtype::F32 => view
                            .data()
                            .chunks(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect(),
                        _ => continue, // Skip unsupported dtypes
                    };
                    out.push((name.to_string(), floats));
                }
                Ok::<_, WorkerError>(out)
            })
            .await
            .map_err(|e| WorkerError::Internal(format!("spawn_blocking join error: {e}")))??;

            // Tensor creation stays in async context (WgpuDevice reference not moved)
            for (name, floats) in parsed {
                let tensor = Tensor::<WorkerBackend, 1>::from_floats(floats.as_slice(), device);
                weights.insert(name, tensor);
            }
        }

        Ok(weights)
    }

    async fn get_model_path(&self, model_name: &str) -> Result<PathBuf, WorkerError> {
        if let Some(api) = &self.hf_api {
            let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));
            let config = repo
                .get("config.json")
                .await
                .map_err(|e| WorkerError::ModelLoad(e.to_string()))?;
            let parent = config.parent().ok_or_else(|| {
                WorkerError::ModelLoad("config.json path has no parent directory".to_string())
            })?;
            return Ok(parent.to_path_buf());
        }
        Err(WorkerError::ModelLoad("No HF API".to_string()))
    }

    async fn load_model_config(
        &self,
        _name: &str,
        _provided: Option<&crate::cluster::ModelConfig>,
        path: &Path,
    ) -> Result<ModelConfig, WorkerError> {
        let config_path = path.join("config.json");
        let s = tokio::fs::read_to_string(&config_path)
            .await
            .map_err(|e| WorkerError::ModelLoad(format!("Failed to read config: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&s)?;
        let arch_raw = json["architectures"][0]
            .as_str()
            .unwrap_or("llama")
            .to_lowercase();
        if arch_raw.contains("qwen3") {
            return Err(WorkerError::ModelLoad(
                "Qwen3 checkpoints are not supported yet (per-head q_norm/k_norm not implemented). \
                 Use a Qwen2.5 checkpoint instead."
                    .to_string(),
            ));
        }
        let architecture = if arch_raw.contains("llama") {
            "llama".to_string()
        } else if arch_raw.contains("qwen") {
            "qwen".to_string()
        } else if arch_raw.contains("deepseek") {
            "deepseek".to_string()
        } else {
            arch_raw
        };
        let num_experts = json["num_experts"]
            .as_u64()
            .or_else(|| json["n_routed_experts"].as_u64()) // DeepSeek-V2/V3 key
            .map(|v| v as usize);
        let num_experts_per_tok = json["num_experts_per_token"]
            .as_u64()
            .or_else(|| json["num_experts_per_tok"].as_u64())
            .map(|v| v as usize);
        let is_moe = num_experts.is_some();
        Ok(ModelConfig {
            architecture,
            num_layers: json["num_hidden_layers"].as_u64().unwrap_or(0) as usize,
            hidden_size: json["hidden_size"].as_u64().unwrap_or(0) as usize,
            num_attention_heads: json["num_attention_heads"].as_u64().unwrap_or(0) as usize,
            num_kv_heads: json["num_key_value_heads"].as_u64().unwrap_or(0) as usize,
            vocab_size: json["vocab_size"].as_u64().unwrap_or(32000) as usize,
            max_seq_len: json["max_position_embeddings"].as_u64().unwrap_or(2048) as usize,
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(0) as usize,
            rms_norm_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta: json["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            head_dim: json["head_dim"].as_u64().map(|v| v as usize),
            attention_bias: json["attention_bias"].as_bool().unwrap_or(false),
            is_moe,
            num_experts,
            num_experts_per_tok,
        })
    }

    /// Honest accounting: every weight is loaded as FP32 today (4 bytes/param),
    /// and MoE models replicate the FFN per expert.
    ///
    /// M4: `config.*` is ultimately sourced from an attacker-influenceable
    /// `config.json` (`as_u64().unwrap_or(0) as usize` in
    /// `load_model_config`), and release builds use Rust's default
    /// `overflow-checks = false` — a plain `usize` multiply chain here would
    /// silently WRAP on a crafted huge value, producing a small
    /// `memory_used` that then under-reserves GPU memory instead of failing
    /// the load. Every multiply/add below runs in `u128` (wide enough that
    /// overflow is unreachable from any `u64`-sourced input) and the result
    /// saturates to `usize::MAX` rather than wrapping, so a bogus config
    /// makes the subsequent `allocate_memory` call fail loudly (not enough
    /// available memory) instead of silently under-reserving.
    fn calculate_memory_usage(&self, config: &ModelConfig) -> usize {
        let vocab_size = config.vocab_size as u128;
        let hidden_size = config.hidden_size as u128;
        let num_layers = config.num_layers as u128;
        let intermediate_size = config.intermediate_size as u128;
        let expert_factor = if config.is_moe {
            config.num_experts.unwrap_or(1).max(1) as u128
        } else {
            1u128
        };

        let embed = vocab_size.saturating_mul(hidden_size);
        let attn = num_layers
            .saturating_mul(4)
            .saturating_mul(hidden_size)
            .saturating_mul(hidden_size);
        let ffn = num_layers
            .saturating_mul(3)
            .saturating_mul(hidden_size)
            .saturating_mul(intermediate_size)
            .saturating_mul(expert_factor);
        let norm = num_layers
            .saturating_mul(2)
            .saturating_add(1)
            .saturating_mul(hidden_size);
        let params = embed
            .saturating_add(attn)
            .saturating_add(ffn)
            .saturating_add(norm);

        params.saturating_mul(4).min(usize::MAX as u128) as usize
    }

    /// Load a GGUF model via the llama.cpp engine (feature `llamacpp`).
    #[cfg(feature = "llamacpp")]
    async fn load_llamacpp_model(
        &self,
        model_name: &str,
        spec: GgufLoadSpec,
        gpu_ids: &[u32],
        quantization: crate::cluster::Quantization,
        parallelism: crate::cluster::ParallelismStrategy,
    ) -> Result<ModelInstance, WorkerError> {
        use crate::llamacpp_engine::LlamaCppEngine;

        info!(
            "Loading GGUF model {} via llama.cpp ({}/{})",
            model_name, spec.repo_id, spec.file
        );

        // Download the GGUF (plus any sibling shards) with the same hf-hub
        // API pattern as safetensors.
        let api = self
            .hf_api
            .as_ref()
            .ok_or_else(|| WorkerError::ModelLoad("HF API not initialized".to_string()))?;
        let repo = api.repo(Repo::new(spec.repo_id.clone(), RepoType::Model));
        let (gguf_path, file_size) = download_gguf(&repo, &spec.file).await?;

        // Phase 1 — reserve the weights (the total GGUF size across every
        // shard; llama.cpp uses the quantized weights as-is). The KV cache
        // needs the model's dimensions, which are only known once it is
        // loaded, so it is reserved in phase 2 below. Roll back on partial
        // failure, matching every sibling path.
        let mut reserved_gpus: Vec<usize> = Vec::new();
        for &gpu_id in gpu_ids {
            match self
                .gpu_manager
                .allocate_memory(gpu_id as usize, file_size, model_name)
                .await
            {
                Ok(()) => reserved_gpus.push(gpu_id as usize),
                Err(e) => {
                    for &g in &reserved_gpus {
                        self.gpu_manager.free_memory(g, model_name).await;
                    }
                    return Err(e);
                }
            }
        }

        // Model load is blocking (mmap + optional GPU upload) — one
        // spawn_blocking, same as the safetensors dtype-conversion path.
        let n_gpu_layers = spec.n_gpu_layers;
        let n_ctx = spec.n_ctx;
        let n_threads = self.llamacpp_n_threads;
        let cache_type_k = spec.cache_type_k.clone();
        let cache_type_v = spec.cache_type_v.clone();
        let path_for_load = gguf_path.clone();
        let engine = tokio::task::spawn_blocking(move || {
            LlamaCppEngine::load(
                &path_for_load,
                n_gpu_layers,
                n_ctx,
                n_threads,
                cache_type_k,
                cache_type_v,
            )
        })
        .await
        .map_err(|e| WorkerError::Internal(format!("spawn_blocking join error: {e}")));

        // The load itself can fail (bad GGUF, OOM inside llama.cpp) — release
        // the phase-1 weight reservation before propagating.
        let engine = match engine {
            Ok(Ok(engine)) => engine,
            Ok(Err(e)) | Err(e) => {
                for &g in &reserved_gpus {
                    self.gpu_manager.free_memory(g, model_name).await;
                }
                return Err(e);
            }
        };

        // Phase 2 — reserve the KV cache now that the model's dimensions are
        // known. H2: the in-process engine builds a FRESH full-n_ctx context
        // PER REQUEST (see `llamacpp_engine.rs`), and up to
        // `max_concurrent_requests` of those can be in flight at once (the
        // `infer_semaphore` in `worker.rs` is sized to exactly that), so the
        // reservation must cover all of them — reserving for a single slot
        // here (as before) undercounts real usage by up to
        // `max_concurrent_requests`x on unified-memory hosts, where that is
        // the Linux OOM killer, not a catchable CUDA allocation error.
        let kv_bytes = engine.kv_cache_bytes(self.max_concurrent_requests as u32);
        for &gpu_id in gpu_ids {
            if let Err(e) = self
                .gpu_manager
                .allocate_memory(gpu_id as usize, kv_bytes, model_name)
                .await
            {
                for &g in &reserved_gpus {
                    self.gpu_manager.free_memory(g, model_name).await;
                }
                return Err(e);
            }
        }
        info!(
            "GGUF {} reserved {:.2} GiB weights + {:.2} GiB KV cache",
            model_name,
            file_size as f64 / (1024.0 * 1024.0 * 1024.0),
            kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // Explicit trait-object coercion, same style as the Burn branches above.
        let model: Arc<Mutex<dyn TextGeneration + Send>> = Arc::new(Mutex::new(engine));

        // Note: the Quantization proto enum describes the Burn pipeline; the
        // GGUF file carries its real quantization (e.g. Q4_K_M) internally.
        let instance = ModelInstance::new(
            model_name.to_string(),
            file_size as usize,
            gpu_ids.to_vec(),
            quantization as i32,
            parallelism as i32,
            Some(model),
        );

        self.evict_to_fit_and_insert(model_name, instance.clone())
            .await;
        info!(
            "GGUF model {} loaded successfully via llama.cpp",
            model_name
        );
        Ok(instance)
    }

    /// Stub used when the worker binary was built without `llamacpp`.
    #[cfg(not(feature = "llamacpp"))]
    async fn load_llamacpp_model(
        &self,
        model_name: &str,
        _spec: GgufLoadSpec,
        _gpu_ids: &[u32],
        _quantization: crate::cluster::Quantization,
        _parallelism: crate::cluster::ParallelismStrategy,
    ) -> Result<ModelInstance, WorkerError> {
        Err(WorkerError::ModelLoad(format!(
            "Model {} requires the llama.cpp engine, but this worker was built \
             without the 'llamacpp' cargo feature (rebuild with --features llamacpp)",
            model_name
        )))
    }
}

// ---------------------------------------------------------------------------
// Multi-shard GGUF download (always compiled — pure hf-hub + string logic,
// no llama.cpp dependency). HuggingFace splits very large GGUFs across
// several files because of its per-file size cap (naming convention
// `<prefix>-%05d-of-%05d.gguf`, e.g. "...-00001-of-00004.gguf"). llama.cpp
// itself auto-loads every sibling once given the FIRST shard's local path,
// but only if every shard already sits next to it on disk — hf-hub does not
// fetch siblings on its own, so the loader must resolve and download every
// shard before handing the first one's path to llama.cpp / llama-server.
// ---------------------------------------------------------------------------

/// Reject a GGUF filename (or a shard name derived from one) that could
/// escape the HF cache directory once handed to `hf-hub`.
///
/// `hf_hub::CacheRepo::get`/`ApiRepo::download_with_progress` build the local
/// cache path with `PathBuf::push(filename)` and DO NOT sanitize `..`
/// components or absolute paths themselves (verified against hf-hub 0.4.3
/// `src/lib.rs::CacheRepo::get` / `src/api/tokio.rs::download_with_progress`
/// — `pointer_path.push(filename)`, no validation). A `filename` of
/// `"../../../../etc/passwd"` (or similar) would make the resulting pointer
/// path — and, via `symlink_or_rename`, an actual symlink — land outside the
/// cache directory. This is a property of the crate, not of shard
/// derivation specifically, but shard derivation (`gguf_shard_filenames`)
/// generates NEW filenames from `file` at runtime that were never literally
/// present in `config/models.toml`, so every one of them is re-validated
/// here too, not just the configured `file`.
///
/// Rejects: empty, absolute (leading `/` or `\`, or a Windows drive prefix
/// like `C:`), and any path component that is exactly `.` or `..`. Forward-
/// slash directory prefixes are otherwise ALLOWED — that is a legitimate,
/// already-used pattern (e.g. the commented-out `glm-4.5-air-gguf` entry's
/// `file = "Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf"`).
fn is_safe_gguf_relative_path(name: &str) -> bool {
    if name.is_empty() || name.starts_with('/') || name.starts_with('\\') {
        return false;
    }
    if name.as_bytes().get(1) == Some(&b':') {
        return false; // e.g. "C:\..." — hf-hub is cross-platform even if this worker targets Linux today
    }
    name.split(['/', '\\'])
        .all(|part| part != "." && part != "..")
}

/// Parse the `<prefix>-<NNNNN>-of-<MMMMM>.gguf` shard-naming convention.
///
/// Returns `(prefix, shard_index, total_shards, digit_width)` on a match.
/// Both numeric groups must be the same zero-padded width (the real
/// llama.cpp/HuggingFace convention — guards against false positives like
/// "release-2024-of-3.gguf") and `1 <= shard_index <= total_shards`;
/// anything else is treated as "not actually a shard filename" (`None`), so
/// ordinary single-file GGUFs are always left untouched.
fn parse_shard_suffix(file: &str) -> Option<(&str, u32, u32, usize)> {
    let stem = file.strip_suffix(".gguf")?;
    // rsplitn(4, '-') yields, from the right: [total, "of", index, prefix]
    // (the 4th item absorbs everything else, so a prefix containing '-' is
    // preserved verbatim).
    let mut parts = stem.rsplitn(4, '-');
    let total_str = parts.next()?;
    let of_str = parts.next()?;
    let index_str = parts.next()?;
    let prefix = parts.next()?;
    if of_str != "of" {
        return None;
    }
    let digit_width = index_str.len();
    if digit_width == 0
        || total_str.len() != digit_width
        || !index_str.bytes().all(|b| b.is_ascii_digit())
        || !total_str.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let index: u32 = index_str.parse().ok()?;
    let total: u32 = total_str.parse().ok()?;
    if index == 0 || total == 0 || index > total {
        return None;
    }
    Some((prefix, index, total, digit_width))
}

/// Every shard filename a configured GGUF `file` needs, in order
/// (`...-00001-of-N...` first). Returns `vec![file.to_string()]` unchanged
/// when `file` does not match the multi-shard naming convention — the common
/// case, and byte-for-byte the prior single-file behavior.
fn gguf_shard_filenames(file: &str) -> Vec<String> {
    match parse_shard_suffix(file) {
        Some((prefix, _index, total, width)) => (1..=total)
            .map(|i| format!("{prefix}-{i:0width$}-of-{total:0width$}.gguf"))
            .collect(),
        None => vec![file.to_string()],
    }
}

/// Download `file` and, if it is part of a multi-shard GGUF, every sibling
/// shard too (see [`gguf_shard_filenames`]). Returns the local path of
/// `file` itself (what gets handed to llama.cpp / llama-server as `-m`) and
/// the TOTAL size in bytes across every downloaded shard — the honest weight
/// size for GPU-memory reservation/logging, not just the first shard's size.
async fn download_gguf(
    repo: &hf_hub::api::tokio::ApiRepo,
    file: &str,
) -> Result<(PathBuf, u64), WorkerError> {
    // Validate BEFORE any shard derivation or network/filesystem action — see
    // `is_safe_gguf_relative_path` for why this exists.
    if !is_safe_gguf_relative_path(file) {
        return Err(WorkerError::Configuration(format!(
            "invalid GGUF filename '{file}': must be a relative path with no '.'/'..' components"
        )));
    }
    let shards = gguf_shard_filenames(file);
    if shards.len() > 1 {
        info!(
            "GGUF '{}' is a {}-shard model — fetching all shards before loading",
            file,
            shards.len()
        );
    }
    let mut primary: Option<PathBuf> = None;
    let mut total_bytes: u64 = 0;
    for shard in &shards {
        // Defense in depth: every DERIVED shard name is re-checked too, not
        // just the configured `file` above (see the doc comment on
        // `is_safe_gguf_relative_path`).
        if !is_safe_gguf_relative_path(shard) {
            return Err(WorkerError::Configuration(format!(
                "invalid derived GGUF shard name '{shard}': must be a relative path with no '.'/'..' components"
            )));
        }
        let path = repo.get(shard).await.map_err(|e| {
            WorkerError::ModelLoad(format!("Failed to download GGUF shard '{shard}': {e}"))
        })?;
        let size = tokio::fs::metadata(&path)
            .await
            .map_err(|e| {
                WorkerError::ModelLoad(format!("Failed to stat GGUF shard '{shard}': {e}"))
            })?
            .len();
        total_bytes += size;
        if shard == file {
            primary = Some(path);
        }
    }
    let primary = primary.ok_or_else(|| {
        WorkerError::ModelLoad(format!(
            "internal error: shard list for '{file}' did not include the file itself"
        ))
    })?;
    Ok((primary, total_bytes))
}

// ---------------------------------------------------------------------------
// llama.cpp engine routing (always compiled — the parse layer has no
// llama.cpp dependency; only the loader in `load_llamacpp_model` is gated)
// ---------------------------------------------------------------------------

/// GGUF model source parsed from the `ModelConfig.metadata` map the
/// coordinator sends in `LoadModelRequest` (see coordinator/models.py
/// `ModelConfig.grpc_metadata`).
#[derive(Debug, Clone, PartialEq)]
pub struct GgufLoadSpec {
    /// HuggingFace repo containing the GGUF file, e.g. "Qwen/Qwen2.5-0.5B-Instruct-GGUF".
    pub repo_id: String,
    /// Exact .gguf filename inside the repo.
    pub file: String,
    /// Transformer layers to offload to the GPU (-1 = all).
    pub n_gpu_layers: i32,
    /// Optional per-model context window override.
    pub n_ctx: Option<u32>,
    /// Level-1 (local multi-GPU) split weights for THIS node's own
    /// `gpu_ids`, in order — len == len(gpu_ids) when present. Parsed from
    /// the comma-separated `tensor_split` metadata key (see the metadata key
    /// contract in pending-work/11-distributed-multi-node-inference.md).
    #[allow(dead_code)]
    // consumed by the Level-1 load() wiring in a later increment (Task 0-gated)
    pub tensor_split: Option<Vec<f32>>,
    /// KV-cache quantization type for K, e.g. "q8_0"/"q4_0" (Task 2b). `None`
    /// leaves llama.cpp's own default (f16) — byte-for-byte today's behavior.
    /// Validated against [`ALLOWED_KV_CACHE_TYPES`] at parse time; the
    /// string→`KvCacheType` mapping itself lives in `llamacpp_engine.rs`
    /// (feature `llamacpp`) since the crate enum isn't available here.
    pub cache_type_k: Option<String>,
    /// KV-cache quantization type for V. See `cache_type_k`.
    pub cache_type_v: Option<String>,
}

/// ggml type names accepted for `cache_type_k`/`cache_type_v` metadata (Task
/// 2b metadata contract). `f16` is llama.cpp's own default — accepted
/// explicitly too so a config can be self-documenting.
const ALLOWED_KV_CACHE_TYPES: &[&str] = &["f16", "q8_0", "q4_0", "q5_0", "q5_1", "q4_1"];

/// Parse one KV-cache-type metadata key, validating it against
/// [`ALLOWED_KV_CACHE_TYPES`]. `Ok(None)` when the key is absent.
///
/// `pub(crate)` so the `llamaserver` engine ([`crate::llamaserver_process`])
/// validates the SAME `cache_type_k`/`cache_type_v` keys identically.
pub(crate) fn parse_cache_type(
    metadata: &HashMap<String, String>,
    key: &str,
) -> Result<Option<String>, WorkerError> {
    match metadata.get(key) {
        None => Ok(None),
        Some(v) if ALLOWED_KV_CACHE_TYPES.contains(&v.as_str()) => Ok(Some(v.clone())),
        Some(v) => Err(WorkerError::Configuration(format!(
            "invalid {key} '{v}' (expected one of {})",
            ALLOWED_KV_CACHE_TYPES.join(", ")
        ))),
    }
}

/// Parse the shared `tensor_split` metadata key: comma-separated f32 weights.
/// Used by both [`gguf_spec_from_metadata`] (Level-1, local) and
/// [`distributed_spec_from_metadata`] (Level-2, cross-node combined split).
fn parse_tensor_split(metadata: &HashMap<String, String>) -> Result<Option<Vec<f32>>, WorkerError> {
    match metadata.get("tensor_split") {
        None => Ok(None),
        Some(v) => {
            let parsed: Result<Vec<f32>, _> =
                v.split(',').map(|s| s.trim().parse::<f32>()).collect();
            let parsed = parsed.map_err(|e| {
                WorkerError::Configuration(format!("invalid tensor_split '{v}': {e}"))
            })?;
            Ok(Some(parsed))
        }
    }
}

/// Parse engine-routing metadata.
///
/// * `Ok(None)` — no metadata, no `engine` key, or `engine == "burn"`: use the
///   default Burn path (fully backwards compatible).
/// * `Ok(Some(spec))` — `engine == "llamacpp"` with a complete GGUF source.
/// * `Err(..)` — unknown engine name or an incomplete/invalid GGUF spec.
pub fn gguf_spec_from_metadata(
    metadata: Option<&HashMap<String, String>>,
    default_n_gpu_layers: i32,
) -> Result<Option<GgufLoadSpec>, WorkerError> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    match metadata.get("engine").map(String::as_str) {
        None | Some("burn") => Ok(None),
        Some("llamacpp") => {
            let repo_id = metadata
                .get("gguf_repo_id")
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    WorkerError::Configuration(
                        "llamacpp engine requires metadata key 'gguf_repo_id'".to_string(),
                    )
                })?
                .clone();
            let file = metadata
                .get("gguf_file")
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    WorkerError::Configuration(
                        "llamacpp engine requires metadata key 'gguf_file'".to_string(),
                    )
                })?
                .clone();
            let n_gpu_layers = match metadata.get("n_gpu_layers") {
                Some(v) => v.parse::<i32>().map_err(|e| {
                    WorkerError::Configuration(format!("invalid n_gpu_layers '{v}': {e}"))
                })?,
                None => default_n_gpu_layers,
            };
            let n_ctx = match metadata.get("n_ctx") {
                Some(v) => Some(v.parse::<u32>().map_err(|e| {
                    WorkerError::Configuration(format!("invalid n_ctx '{v}': {e}"))
                })?),
                None => None,
            };
            let tensor_split = parse_tensor_split(metadata)?;
            let cache_type_k = parse_cache_type(metadata, "cache_type_k")?;
            let cache_type_v = parse_cache_type(metadata, "cache_type_v")?;
            Ok(Some(GgufLoadSpec {
                repo_id,
                file,
                n_gpu_layers,
                n_ctx,
                tensor_split,
                cache_type_k,
                cache_type_v,
            }))
        }
        Some(other) => Err(WorkerError::Configuration(format!(
            "Unknown inference engine '{other}' (expected 'burn' or 'llamacpp')"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Distributed (Level-2, cross-node ggml-RPC) metadata routing — always
// compiled, no llama.cpp dependency. See the metadata key contract in
// pending-work/11-distributed-multi-node-inference.md. `Ok(None)` when the
// `distributed_role` key is absent means byte-for-byte today's single-node
// behavior; only the STUB `rpc_server` role is wired into a load path this
// increment (Task 5) — the real `lead` engine ships in Task 9.
// ---------------------------------------------------------------------------

/// Which side of a Level-2 (cross-node) ggml-RPC split this node plays for
/// one distributed model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedRole {
    /// Owns the real llama.cpp context for the model and dials out to
    /// `rpc_peers` (Task 9 — not implemented this increment).
    Lead,
    /// Lends this node's local GPU(s) compute+VRAM to a lead; holds no
    /// directly-servable model (Task 5 STUB, this increment).
    RpcServer,
}

/// Level-2 distributed-load metadata parsed from `ModelConfig.metadata`.
#[derive(Debug, Clone, PartialEq)]
pub struct DistributedSpec {
    /// This node's role for the distributed model.
    pub role: DistributedRole,
    /// Lead only: comma-separated "host:port" peers, GPU-granular.
    ///
    /// Ordering invariant (do not break without updating both sides):
    /// `rpc_peers[i]` and `tensor_split[len(lead_gpu_ids) + i]` refer to the
    /// SAME (node, GPU) — the peer-portion of `tensor_split` must be laid out
    /// in the exact same order as `rpc_peers`.
    pub rpc_peers: Vec<String>,
    /// rpc_server only: base port its `rpc-server` process(es) bind to.
    pub rpc_bind_port: Option<u16>,
    /// Lead only: combined flat split weights — [lead's own gpu_ids...,
    /// peer_1's lent GPUs..., peer_2's...].
    pub tensor_split: Option<Vec<f32>>,
}

/// Parse Level-2 distributed-role metadata.
///
/// * `Ok(None)` — no metadata or no `distributed_role` key: today's
///   single-node path, fully backwards compatible.
/// * `Ok(Some(spec))` — `distributed_role` is `"lead"` or `"rpc_server"`.
/// * `Err(..)` — unknown role, or a malformed `rpc_bind_port`/`tensor_split`.
pub fn distributed_spec_from_metadata(
    metadata: Option<&HashMap<String, String>>,
) -> Result<Option<DistributedSpec>, WorkerError> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let role = match metadata.get("distributed_role").map(String::as_str) {
        None => return Ok(None),
        Some("lead") => DistributedRole::Lead,
        Some("rpc_server") => DistributedRole::RpcServer,
        Some(other) => {
            return Err(WorkerError::Configuration(format!(
                "Unknown distributed_role '{other}' (expected 'lead' or 'rpc_server')"
            )))
        }
    };

    let rpc_peers = match metadata.get("rpc_peers") {
        Some(v) if !v.is_empty() => v.split(',').map(|s| s.trim().to_string()).collect(),
        _ => Vec::new(),
    };

    let rpc_bind_port = match metadata.get("rpc_bind_port") {
        Some(v) => Some(v.parse::<u16>().map_err(|e| {
            WorkerError::Configuration(format!("invalid rpc_bind_port '{v}': {e}"))
        })?),
        None => None,
    };

    let tensor_split = parse_tensor_split(metadata)?;

    Ok(Some(DistributedSpec {
        role,
        rpc_peers,
        rpc_bind_port,
        tensor_split,
    }))
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

    // --- path-traversal validation (security hardening on fix 3) ----------

    #[test]
    fn safe_path_accepts_plain_and_directory_prefixed_names() {
        assert!(is_safe_gguf_relative_path("model-q4_k_m.gguf"));
        assert!(is_safe_gguf_relative_path(
            "Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf"
        ));
        // A component that merely STARTS with ".." but isn't exactly ".." is
        // a legitimate (if unusual) filename, not a traversal.
        assert!(is_safe_gguf_relative_path("..hidden-00001-of-00002.gguf"));
    }

    #[test]
    fn safe_path_rejects_traversal_and_absolute_paths() {
        assert!(!is_safe_gguf_relative_path(""));
        assert!(!is_safe_gguf_relative_path("../../../../etc/passwd"));
        assert!(!is_safe_gguf_relative_path("a/../../b.gguf"));
        assert!(!is_safe_gguf_relative_path("./model.gguf"));
        assert!(!is_safe_gguf_relative_path("/etc/passwd"));
        assert!(!is_safe_gguf_relative_path("\\Windows\\System32\\evil"));
        assert!(!is_safe_gguf_relative_path("C:\\evil.gguf"));
    }

    #[test]
    fn download_gguf_rejects_traversal_before_any_shard_expansion() {
        // gguf_shard_filenames is pure string logic and never invoked over
        // the network, so this exercises the same validation `download_gguf`
        // performs without needing a real hf-hub API/repo. A crafted
        // multi-shard-looking traversal filename must never even reach shard
        // expansion/download.
        assert!(!is_safe_gguf_relative_path(
            "../../../../etc/cron.d/evil-00001-of-00002.gguf"
        ));
    }

    // --- multi-shard GGUF filename detection/generation (fix 3) ------------

    #[test]
    fn shard_pattern_single_file_unchanged() {
        assert_eq!(
            gguf_shard_filenames("qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            vec!["qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()]
        );
    }

    #[test]
    fn shard_pattern_detects_and_generates_all_siblings() {
        let shards = gguf_shard_filenames("Qwen3-Coder-Next-00001-of-00004.gguf");
        assert_eq!(
            shards,
            vec![
                "Qwen3-Coder-Next-00001-of-00004.gguf".to_string(),
                "Qwen3-Coder-Next-00002-of-00004.gguf".to_string(),
                "Qwen3-Coder-Next-00003-of-00004.gguf".to_string(),
                "Qwen3-Coder-Next-00004-of-00004.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn shard_pattern_works_from_any_starting_shard() {
        // The configured `file` need not be shard 1 — the sibling list is
        // always the full 1..=total set regardless of which one was given.
        let shards = gguf_shard_filenames("gpt-oss-120b-mxfp4-00003-of-00003.gguf");
        assert_eq!(
            shards,
            vec![
                "gpt-oss-120b-mxfp4-00001-of-00003.gguf".to_string(),
                "gpt-oss-120b-mxfp4-00002-of-00003.gguf".to_string(),
                "gpt-oss-120b-mxfp4-00003-of-00003.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn shard_pattern_preserves_directory_prefix() {
        let shards =
            gguf_shard_filenames("Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-00001-of-00004.gguf");
        assert_eq!(shards.len(), 4);
        assert_eq!(
            shards[3],
            "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-00004-of-00004.gguf"
        );
    }

    #[test]
    fn shard_pattern_two_way_split_with_directory() {
        let shards = gguf_shard_filenames("Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf");
        assert_eq!(
            shards,
            vec![
                "Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf".to_string(),
                "Q4_K_M/GLM-4.5-Air-Q4_K_M-00002-of-00002.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn shard_pattern_rejects_malformed_or_non_shard_names() {
        // No ".gguf" suffix at all.
        assert_eq!(
            gguf_shard_filenames("model-00001-of-00004"),
            vec!["model-00001-of-00004".to_string()]
        );
        // Mismatched zero-padding width between the two numeric groups.
        assert_eq!(
            gguf_shard_filenames("model-1-of-00004.gguf"),
            vec!["model-1-of-00004.gguf".to_string()]
        );
        // Shard index 0 (1-based convention) is not a valid shard set.
        assert_eq!(
            gguf_shard_filenames("model-00000-of-00004.gguf"),
            vec!["model-00000-of-00004.gguf".to_string()]
        );
        // Shard index beyond the declared total.
        assert_eq!(
            gguf_shard_filenames("model-00005-of-00004.gguf"),
            vec!["model-00005-of-00004.gguf".to_string()]
        );
        // A filename that merely CONTAINS "-of-" text but isn't the shard
        // pattern (non-digit segment where the index should be) must fall
        // through unchanged rather than false-positive.
        assert_eq!(
            gguf_shard_filenames("best-of-3-results.gguf"),
            vec!["best-of-3-results.gguf".to_string()]
        );
    }

    #[test]
    fn gguf_spec_none_without_metadata() {
        assert_eq!(gguf_spec_from_metadata(None, -1).unwrap(), None);
    }

    #[test]
    fn gguf_spec_none_for_burn_or_absent_engine() {
        let burn = meta(&[("engine", "burn")]);
        assert_eq!(gguf_spec_from_metadata(Some(&burn), -1).unwrap(), None);
        let empty = meta(&[]);
        assert_eq!(gguf_spec_from_metadata(Some(&empty), -1).unwrap(), None);
    }

    #[test]
    fn gguf_spec_parses_full_llamacpp_metadata() {
        let m = meta(&[
            ("engine", "llamacpp"),
            ("gguf_repo_id", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            ("gguf_file", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            ("n_gpu_layers", "20"),
            ("n_ctx", "4096"),
        ]);
        let spec = gguf_spec_from_metadata(Some(&m), -1).unwrap().unwrap();
        assert_eq!(spec.repo_id, "Qwen/Qwen2.5-0.5B-Instruct-GGUF");
        assert_eq!(spec.file, "qwen2.5-0.5b-instruct-q4_k_m.gguf");
        assert_eq!(spec.n_gpu_layers, 20);
        assert_eq!(spec.n_ctx, Some(4096));
    }

    #[test]
    fn gguf_spec_applies_worker_default_gpu_layers() {
        let m = meta(&[
            ("engine", "llamacpp"),
            ("gguf_repo_id", "some/repo"),
            ("gguf_file", "model.gguf"),
        ]);
        let spec = gguf_spec_from_metadata(Some(&m), -1).unwrap().unwrap();
        assert_eq!(spec.n_gpu_layers, -1);
        assert_eq!(spec.n_ctx, None);
    }

    #[test]
    fn gguf_spec_rejects_incomplete_or_unknown() {
        let missing_file = meta(&[("engine", "llamacpp"), ("gguf_repo_id", "some/repo")]);
        assert!(gguf_spec_from_metadata(Some(&missing_file), -1).is_err());

        let missing_repo = meta(&[("engine", "llamacpp"), ("gguf_file", "model.gguf")]);
        assert!(gguf_spec_from_metadata(Some(&missing_repo), -1).is_err());

        let unknown = meta(&[("engine", "vllm")]);
        assert!(gguf_spec_from_metadata(Some(&unknown), -1).is_err());

        let bad_layers = meta(&[
            ("engine", "llamacpp"),
            ("gguf_repo_id", "some/repo"),
            ("gguf_file", "model.gguf"),
            ("n_gpu_layers", "many"),
        ]);
        assert!(gguf_spec_from_metadata(Some(&bad_layers), -1).is_err());
    }

    #[test]
    fn gguf_spec_parses_tensor_split_when_present() {
        let m = meta(&[
            ("engine", "llamacpp"),
            ("gguf_repo_id", "some/repo"),
            ("gguf_file", "model.gguf"),
            ("tensor_split", "0.6,0.4"),
        ]);
        let spec = gguf_spec_from_metadata(Some(&m), -1).unwrap().unwrap();
        assert_eq!(spec.tensor_split, Some(vec![0.6, 0.4]));
    }

    #[test]
    fn gguf_spec_tensor_split_absent_is_none() {
        let m = meta(&[
            ("engine", "llamacpp"),
            ("gguf_repo_id", "some/repo"),
            ("gguf_file", "model.gguf"),
        ]);
        let spec = gguf_spec_from_metadata(Some(&m), -1).unwrap().unwrap();
        assert_eq!(spec.tensor_split, None);
    }

    #[test]
    fn gguf_spec_rejects_malformed_tensor_split() {
        let m = meta(&[
            ("engine", "llamacpp"),
            ("gguf_repo_id", "some/repo"),
            ("gguf_file", "model.gguf"),
            ("tensor_split", "0.6,not-a-float"),
        ]);
        assert!(gguf_spec_from_metadata(Some(&m), -1).is_err());
    }

    #[test]
    fn gguf_spec_parses_cache_types_when_present() {
        let m = meta(&[
            ("engine", "llamacpp"),
            ("gguf_repo_id", "some/repo"),
            ("gguf_file", "model.gguf"),
            ("cache_type_k", "q8_0"),
            ("cache_type_v", "q4_0"),
        ]);
        let spec = gguf_spec_from_metadata(Some(&m), -1).unwrap().unwrap();
        assert_eq!(spec.cache_type_k, Some("q8_0".to_string()));
        assert_eq!(spec.cache_type_v, Some("q4_0".to_string()));
    }

    #[test]
    fn gguf_spec_cache_types_absent_is_none() {
        let m = meta(&[
            ("engine", "llamacpp"),
            ("gguf_repo_id", "some/repo"),
            ("gguf_file", "model.gguf"),
        ]);
        let spec = gguf_spec_from_metadata(Some(&m), -1).unwrap().unwrap();
        assert_eq!(spec.cache_type_k, None);
        assert_eq!(spec.cache_type_v, None);
    }

    #[test]
    fn gguf_spec_accepts_every_allowed_cache_type() {
        for ty in ALLOWED_KV_CACHE_TYPES {
            let m = meta(&[
                ("engine", "llamacpp"),
                ("gguf_repo_id", "some/repo"),
                ("gguf_file", "model.gguf"),
                ("cache_type_k", ty),
            ]);
            let spec = gguf_spec_from_metadata(Some(&m), -1).unwrap().unwrap();
            assert_eq!(spec.cache_type_k, Some(ty.to_string()));
        }
    }

    #[test]
    fn gguf_spec_rejects_invalid_cache_type() {
        let bad_k = meta(&[
            ("engine", "llamacpp"),
            ("gguf_repo_id", "some/repo"),
            ("gguf_file", "model.gguf"),
            ("cache_type_k", "int8"),
        ]);
        assert!(gguf_spec_from_metadata(Some(&bad_k), -1).is_err());

        let bad_v = meta(&[
            ("engine", "llamacpp"),
            ("gguf_repo_id", "some/repo"),
            ("gguf_file", "model.gguf"),
            ("cache_type_v", "fp16"), // close but not the accepted spelling
        ]);
        assert!(gguf_spec_from_metadata(Some(&bad_v), -1).is_err());
    }

    #[test]
    fn distributed_spec_none_without_metadata_or_role() {
        assert_eq!(distributed_spec_from_metadata(None).unwrap(), None);
        let empty = meta(&[]);
        assert_eq!(distributed_spec_from_metadata(Some(&empty)).unwrap(), None);
        let no_role = meta(&[("engine", "llamacpp")]);
        assert_eq!(
            distributed_spec_from_metadata(Some(&no_role)).unwrap(),
            None
        );
    }

    #[test]
    fn distributed_spec_parses_lead_role() {
        let m = meta(&[
            ("distributed_role", "lead"),
            ("rpc_peers", "10.0.0.2:50151,10.0.0.3:50151"),
            ("tensor_split", "0.5,0.3,0.2"),
        ]);
        let spec = distributed_spec_from_metadata(Some(&m)).unwrap().unwrap();
        assert_eq!(spec.role, DistributedRole::Lead);
        assert_eq!(
            spec.rpc_peers,
            vec!["10.0.0.2:50151".to_string(), "10.0.0.3:50151".to_string()]
        );
        assert_eq!(spec.tensor_split, Some(vec![0.5, 0.3, 0.2]));
        assert_eq!(spec.rpc_bind_port, None);
    }

    #[test]
    fn distributed_spec_parses_rpc_server_role() {
        let m = meta(&[
            ("distributed_role", "rpc_server"),
            ("rpc_bind_port", "50151"),
        ]);
        let spec = distributed_spec_from_metadata(Some(&m)).unwrap().unwrap();
        assert_eq!(spec.role, DistributedRole::RpcServer);
        assert_eq!(spec.rpc_bind_port, Some(50151));
        assert!(spec.rpc_peers.is_empty());
        assert_eq!(spec.tensor_split, None);
    }

    #[test]
    fn distributed_spec_rejects_unknown_role() {
        let unknown = meta(&[("distributed_role", "follower")]);
        assert!(distributed_spec_from_metadata(Some(&unknown)).is_err());
    }

    #[test]
    fn distributed_spec_rejects_malformed_bind_port_or_tensor_split() {
        let bad_port = meta(&[
            ("distributed_role", "rpc_server"),
            ("rpc_bind_port", "not-a-port"),
        ]);
        assert!(distributed_spec_from_metadata(Some(&bad_port)).is_err());

        let bad_split = meta(&[("distributed_role", "lead"), ("tensor_split", "abc")]);
        assert!(distributed_spec_from_metadata(Some(&bad_split)).is_err());
    }

    // -----------------------------------------------------------------
    // rpc_server STUB role (Task 5): exercised through the real
    // `ModelLoader::load_model` entrypoint — the same path the worker.rs
    // `LoadModel` gRPC handler calls — so these double as an integration
    // check of the routing order, without needing a live gRPC harness.
    // -----------------------------------------------------------------

    fn test_loader(gpu_manager: Arc<GPUManager>) -> (ModelLoader, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let loader_config = ModelLoaderConfig {
            cache_dir: dir.path().join("models"),
            download_dir: dir.path().join("downloads"),
            ..ModelLoaderConfig::default()
        };
        let loader = ModelLoader::new(loader_config, gpu_manager).unwrap();
        (loader, dir)
    }

    #[tokio::test]
    async fn rpc_server_stub_reserves_vram_and_leaves_no_model_attached() {
        let gpu_manager = Arc::new(GPUManager::new(&[0]).await.unwrap());
        let (loader, _dir) = test_loader(gpu_manager.clone());

        let metadata = meta(&[
            ("distributed_role", "rpc_server"),
            ("rpc_bind_port", "50151"),
        ]);
        let model_config = crate::cluster::ModelConfig {
            metadata,
            ..Default::default()
        };

        let instance = loader
            .load_model(
                "peer-stub",
                None,
                Some(&model_config),
                &[0],
                crate::cluster::Quantization::None,
                crate::cluster::ParallelismStrategy::Auto,
            )
            .await
            .unwrap();

        assert_eq!(instance.gpu_ids(), &[0]);
        // The stub reserved everything free on GPU 0 — available_memory drops.
        assert_eq!(gpu_manager.get_available_memory(0).await, 0);

        // A stray Infer against the placeholder returns the existing clean
        // "no model attached" error (ModelInstance::generate), not a panic.
        // (TextStream isn't Debug, so match instead of unwrap_err().)
        match instance.generate("hello", 4, 0.7, 0.9, 40, None).await {
            Err(e) => assert!(e.to_string().contains("holds no runnable model")),
            Ok(_) => panic!("expected 'no model attached' error, got a stream"),
        }
    }

    #[tokio::test]
    async fn rpc_server_role_routes_before_gguf_metadata_is_consulted() {
        // Ordering invariant: the distributed rpc_server branch must run
        // BEFORE the gguf/Burn routing. Prove it by attaching `engine`/
        // `gguf_repo_id`/`gguf_file` metadata that would trigger a network
        // download if the gguf branch ran first — the stub must short-circuit
        // before ever looking at those keys.
        let gpu_manager = Arc::new(GPUManager::new(&[0]).await.unwrap());
        let (loader, _dir) = test_loader(gpu_manager.clone());

        let metadata = meta(&[
            ("distributed_role", "rpc_server"),
            ("rpc_bind_port", "50151"),
            ("engine", "llamacpp"),
            ("gguf_repo_id", "does-not/exist"),
            ("gguf_file", "does-not-exist.gguf"),
        ]);
        let model_config = crate::cluster::ModelConfig {
            metadata,
            ..Default::default()
        };

        let instance = loader
            .load_model(
                "peer-stub-2",
                None,
                Some(&model_config),
                &[0],
                crate::cluster::Quantization::None,
                crate::cluster::ParallelismStrategy::Auto,
            )
            .await
            .unwrap();
        assert_eq!(instance.gpu_ids(), &[0]);
    }

    // -----------------------------------------------------------------
    // H3: resolve_head_dim (div-by-zero panic on attacker-influenced
    // config.json) and ReservationGuard (release-on-unwind).
    // -----------------------------------------------------------------

    fn base_model_config() -> ModelConfig {
        ModelConfig {
            architecture: "llama".to_string(),
            num_layers: 2,
            hidden_size: 64,
            num_attention_heads: 8,
            num_kv_heads: 8,
            vocab_size: 100,
            max_seq_len: 128,
            intermediate_size: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: None,
            attention_bias: false,
            is_moe: false,
            num_experts: None,
            num_experts_per_tok: None,
        }
    }

    #[test]
    fn resolve_head_dim_divides_when_absent() {
        let config = base_model_config();
        assert_eq!(resolve_head_dim(&config).unwrap(), 8); // 64 / 8
    }

    #[test]
    fn resolve_head_dim_prefers_explicit_value_without_dividing() {
        let mut config = base_model_config();
        config.num_attention_heads = 0; // would panic on division if this were used
        config.head_dim = Some(16);
        assert_eq!(resolve_head_dim(&config).unwrap(), 16);
    }

    #[test]
    fn resolve_head_dim_errors_instead_of_panicking_on_zero_heads() {
        // The exact H3 scenario: an attacker/corrupt-repo config.json with
        // num_attention_heads = 0 and no explicit head_dim. Must return an
        // Err, never panic (the crate is compiled with overflow/div checks
        // that would abort the process on a raw `/0` here).
        let mut config = base_model_config();
        config.num_attention_heads = 0;
        config.head_dim = None;
        let err = resolve_head_dim(&config).unwrap_err();
        assert!(err.to_string().contains("num_attention_heads"));
    }

    #[tokio::test]
    async fn reservation_guard_releases_memory_on_drop_without_commit() {
        let gpu_manager = Arc::new(GPUManager::new(&[0]).await.unwrap());
        let before = gpu_manager.get_available_memory(0).await;

        {
            let mut guard = ReservationGuard::new(gpu_manager.clone(), "leak-test");
            gpu_manager
                .allocate_memory(0, 1_000, "leak-test")
                .await
                .unwrap();
            guard.track(0);
            assert_eq!(gpu_manager.get_available_memory(0).await, before - 1_000);
            // Dropped here WITHOUT calling commit() — simulates an early
            // return or panic between reservation and a successful load.
        }

        // Drop spawns the release onto the runtime rather than blocking;
        // give it a chance to run.
        for _ in 0..50 {
            if gpu_manager.get_available_memory(0).await == before {
                break;
            }
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(
            gpu_manager.get_available_memory(0).await,
            before,
            "ReservationGuard must release its tracked reservation on drop"
        );
    }

    #[tokio::test]
    async fn reservation_guard_commit_prevents_release_on_drop() {
        let gpu_manager = Arc::new(GPUManager::new(&[0]).await.unwrap());
        let before = gpu_manager.get_available_memory(0).await;

        gpu_manager
            .allocate_memory(0, 1_000, "committed-test")
            .await
            .unwrap();
        let mut guard = ReservationGuard::new(gpu_manager.clone(), "committed-test");
        guard.track(0);
        guard.commit(); // ownership transferred — must NOT free on drop

        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(
            gpu_manager.get_available_memory(0).await,
            before - 1_000,
            "commit() must prevent the guard from releasing the reservation"
        );
        // Clean up for hygiene (not asserted).
        gpu_manager.free_memory(0, "committed-test").await;
    }

    // -----------------------------------------------------------------
    // H7: max_loaded_models eviction race — two concurrent loads racing
    // through evict_to_fit_and_insert must never both succeed in exceeding
    // the cap, even though the pre-load eviction pass (only guarded by the
    // PER-MODEL-NAME lock) can't see each other's decisions.
    // -----------------------------------------------------------------

    fn dummy_instance(name: &str) -> ModelInstance {
        ModelInstance::new(name.to_string(), 0, vec![0], 0, 0, None)
    }

    #[tokio::test]
    async fn evict_to_fit_and_insert_never_exceeds_cap_under_concurrency() {
        let gpu_manager = Arc::new(GPUManager::new(&[0]).await.unwrap());
        let dir = tempfile::tempdir().unwrap();
        let loader_config = ModelLoaderConfig {
            cache_dir: dir.path().join("models"),
            download_dir: dir.path().join("downloads"),
            max_loaded_models: 1,
            ..ModelLoaderConfig::default()
        };
        let loader = Arc::new(ModelLoader::new(loader_config, gpu_manager).unwrap());

        // Pre-populate with one already-resident model, matching the race
        // scenario in the audit: model "a" is loaded, and TWO different new
        // models ("b" and "c") each try to load concurrently against a cap
        // of 1 — only one can end up resident.
        loader
            .loaded_models
            .insert("a".to_string(), dummy_instance("a"));

        let loader_b = loader.clone();
        let loader_c = loader.clone();
        let (_, _) = tokio::join!(
            loader_b.evict_to_fit_and_insert("b", dummy_instance("b")),
            loader_c.evict_to_fit_and_insert("c", dummy_instance("c")),
        );

        assert_eq!(
            loader.loaded_models.len(),
            1,
            "max_loaded_models=1 must never be exceeded, even under concurrent loads \
             of different model names"
        );
    }

    // -----------------------------------------------------------------
    // M4: calculate_memory_usage must saturate, never overflow/wrap, on a
    // maximally-hostile config.json. `cargo test` runs in debug (overflow
    // checks ON), so the pre-fix `usize *` chain would have PANICKED here,
    // not just silently wrapped as it would in a release build.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn calculate_memory_usage_saturates_instead_of_overflowing() {
        let gpu_manager = Arc::new(GPUManager::new(&[0]).await.unwrap());
        let (loader, _dir) = test_loader(gpu_manager);
        let mut config = base_model_config();
        // Every field maxed out — the worst case an attacker-controlled
        // `config.json` (`as_u64().unwrap_or(0) as usize`) could supply.
        config.vocab_size = usize::MAX;
        config.hidden_size = usize::MAX;
        config.num_layers = usize::MAX;
        config.intermediate_size = usize::MAX;
        config.is_moe = true;
        config.num_experts = Some(usize::MAX);
        let usage = loader.calculate_memory_usage(&config);
        assert_eq!(usage, usize::MAX, "must saturate, not wrap, on overflow");
    }

    #[tokio::test]
    async fn calculate_memory_usage_matches_hand_computation_for_sane_config() {
        let gpu_manager = Arc::new(GPUManager::new(&[0]).await.unwrap());
        let (loader, _dir) = test_loader(gpu_manager);
        let config = base_model_config(); // hidden=64, layers=2, heads=8, vocab=100, ffn=128
        let usage = loader.calculate_memory_usage(&config);
        let embed = 100 * 64;
        let attn = 2 * 4 * 64 * 64;
        let ffn = 2 * 3 * 64 * 128;
        let norm = (2 * 2 + 1) * 64;
        let expected = (embed + attn + ffn + norm) * 4;
        assert_eq!(usage, expected);
    }
}

/// Helper to transpose Linear weights (HF [out, in] -> Burn [in, out])
fn load_linear(
    weights: &mut HashMap<String, Tensor<WorkerBackend, 1>>,
    name: &str,
    in_features: usize,
    out_features: usize,
    bias: bool,
) -> Result<LinearRecord<WorkerBackend>, WorkerError> {
    let w_name = format!("{}.weight", name);
    let w_flat = weights
        .remove(&w_name)
        .ok_or(WorkerError::ModelLoad(format!("Missing {}", w_name)))?;

    // HF: [out, in]
    // Burn expects: [in, out]
    // So we reshape to HF shape, transpose, then into Burn shape.
    // Actually, simply reshaping to [out, in] and transposing gives [in, out].
    let w = w_flat.reshape([out_features, in_features]).transpose();

    let b = if bias {
        let b_name = format!("{}.bias", name);
        weights.remove(&b_name)
    } else {
        None
    };

    Ok(LinearRecord {
        weight: Param::initialized(ParamId::new(), w),
        bias: b.map(|t| Param::initialized(ParamId::new(), t)),
    })
}

fn load_embedding(
    weights: &mut HashMap<String, Tensor<WorkerBackend, 1>>,
    name: &str,
    num_embeddings: usize,
    embedding_dim: usize,
) -> Result<EmbeddingRecord<WorkerBackend>, WorkerError> {
    let w_name = format!("{}.weight", name);
    let w_flat = weights
        .remove(&w_name)
        .ok_or(WorkerError::ModelLoad(format!("Missing {}", w_name)))?;
    let w = w_flat.reshape([num_embeddings, embedding_dim]);
    Ok(EmbeddingRecord {
        weight: Param::initialized(ParamId::new(), w),
    })
}

fn load_norm(
    weights: &mut HashMap<String, Tensor<WorkerBackend, 1>>,
    name: &str,
    _dim: usize,
) -> Result<RMSNormRecord<WorkerBackend>, WorkerError> {
    let w_name = format!("{}.weight", name);
    let w_flat = weights
        .remove(&w_name)
        .ok_or(WorkerError::ModelLoad(format!("Missing {}", w_name)))?;
    let w = w_flat; // 1D
    Ok(RMSNormRecord {
        weight: Param::initialized(ParamId::new(), w),
        eps: ConstantRecord,
    })
}

fn create_qwen_record(
    weights: &mut HashMap<String, Tensor<WorkerBackend, 1>>,
    config: &QwenConfig,
) -> Result<QwenRecord<WorkerBackend>, WorkerError> {
    let embed = load_embedding(
        weights,
        "model.embed_tokens",
        config.vocab_size,
        config.hidden_size,
    )?;
    let norm = load_norm(weights, "model.norm", config.hidden_size)?;
    let lm_head = load_linear(
        weights,
        "lm_head",
        config.hidden_size,
        config.vocab_size,
        false,
    )?;

    let mut layers = Vec::new();
    for i in 0..config.num_layers {
        let prefix = format!("model.layers.{}", i);

        let bias = config.attention_bias;
        let q = load_linear(
            weights,
            &format!("{}.self_attn.q_proj", prefix),
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias,
        )?;
        let k = load_linear(
            weights,
            &format!("{}.self_attn.k_proj", prefix),
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            bias,
        )?;
        let v = load_linear(
            weights,
            &format!("{}.self_attn.v_proj", prefix),
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            bias,
        )?;
        let o = load_linear(
            weights,
            &format!("{}.self_attn.o_proj", prefix),
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            false,
        )?;

        let attention = QwenAttentionRecord {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            num_heads: ConstantRecord,
            num_kv_heads: ConstantRecord,
            head_dim: ConstantRecord,
        };

        let gate = load_linear(
            weights,
            &format!("{}.mlp.gate_proj", prefix),
            config.hidden_size,
            config.intermediate_size,
            false,
        )?;
        let up = load_linear(
            weights,
            &format!("{}.mlp.up_proj", prefix),
            config.hidden_size,
            config.intermediate_size,
            false,
        )?;
        let down = load_linear(
            weights,
            &format!("{}.mlp.down_proj", prefix),
            config.intermediate_size,
            config.hidden_size,
            false,
        )?;

        let mlp = QwenMLPRecord {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        };

        let in_norm = load_norm(
            weights,
            &format!("{}.input_layernorm", prefix),
            config.hidden_size,
        )?;
        let post_norm = load_norm(
            weights,
            &format!("{}.post_attention_layernorm", prefix),
            config.hidden_size,
        )?;

        layers.push(QwenLayerRecord {
            attention,
            mlp,
            input_layernorm: in_norm,
            post_attention_layernorm: post_norm,
        });
    }

    let rope = RotaryEmbeddingRecord {
        cos: ConstantRecord,
        sin: ConstantRecord,
    };

    Ok(QwenRecord {
        embed_tokens: embed,
        layers,
        norm,
        lm_head,
        config: ConstantRecord,
        rope,
        tokenizer: ConstantRecord,
        device: ConstantRecord,
        eos_token_ids: ConstantRecord,
    })
}

fn create_deepseek_record(
    weights: &mut HashMap<String, Tensor<WorkerBackend, 1>>,
    config: &DeepSeekConfig,
) -> Result<DeepSeekRecord<WorkerBackend>, WorkerError> {
    let embed = load_embedding(
        weights,
        "model.embed_tokens",
        config.vocab_size,
        config.hidden_size,
    )?;
    let norm = load_norm(weights, "model.norm", config.hidden_size)?;
    let lm_head = load_linear(
        weights,
        "lm_head",
        config.hidden_size,
        config.vocab_size,
        false,
    )?;

    let q_out = config.num_attention_heads * config.head_dim;
    let kv_out = config.num_kv_heads * config.head_dim;

    let mut layers = Vec::new();
    for i in 0..config.num_layers {
        let prefix = format!("model.layers.{}", i);

        let q = load_linear(
            weights,
            &format!("{}.self_attn.q_proj", prefix),
            config.hidden_size,
            q_out,
            false,
        )?;
        let k = load_linear(
            weights,
            &format!("{}.self_attn.k_proj", prefix),
            config.hidden_size,
            kv_out,
            false,
        )?;
        let v = load_linear(
            weights,
            &format!("{}.self_attn.v_proj", prefix),
            config.hidden_size,
            kv_out,
            false,
        )?;
        let o = load_linear(
            weights,
            &format!("{}.self_attn.o_proj", prefix),
            q_out,
            config.hidden_size,
            false,
        )?;

        let attention = DeepSeekAttentionRecord {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            num_heads: ConstantRecord,
            num_kv_heads: ConstantRecord,
            head_dim: ConstantRecord,
        };

        // Routing gate: [hidden_size → num_experts]
        let gate_w = load_linear(
            weights,
            &format!("{}.mlp.gate", prefix),
            config.hidden_size,
            config.num_experts,
            false,
        )?;

        // Load all routed experts
        let mut experts = Vec::with_capacity(config.num_experts);
        for j in 0..config.num_experts {
            let ep = format!("{}.mlp.experts.{}", prefix, j);
            let eg = load_linear(
                weights,
                &format!("{}.gate_proj", ep),
                config.hidden_size,
                config.intermediate_size,
                false,
            )?;
            let eu = load_linear(
                weights,
                &format!("{}.up_proj", ep),
                config.hidden_size,
                config.intermediate_size,
                false,
            )?;
            let ed = load_linear(
                weights,
                &format!("{}.down_proj", ep),
                config.intermediate_size,
                config.hidden_size,
                false,
            )?;
            experts.push(ExpertRecord {
                gate_proj: eg,
                up_proj: eu,
                down_proj: ed,
            });
        }

        let moe = DeepSeekMoERecord {
            experts,
            gate: gate_w,
            num_experts_per_tok: ConstantRecord,
        };

        let in_norm = load_norm(
            weights,
            &format!("{}.input_layernorm", prefix),
            config.hidden_size,
        )?;
        let post_norm = load_norm(
            weights,
            &format!("{}.post_attention_layernorm", prefix),
            config.hidden_size,
        )?;

        layers.push(DeepSeekLayerRecord {
            attention,
            moe,
            input_layernorm: in_norm,
            post_attention_layernorm: post_norm,
        });
    }

    let rope = RotaryEmbeddingRecord {
        cos: ConstantRecord,
        sin: ConstantRecord,
    };

    Ok(DeepSeekRecord {
        embed_tokens: embed,
        layers,
        norm,
        lm_head,
        config: ConstantRecord,
        context_device: ConstantRecord,
        rope,
        tokenizer: ConstantRecord,
        eos_token_ids: ConstantRecord,
    })
}

fn create_llama_record(
    weights: &mut HashMap<String, Tensor<WorkerBackend, 1>>,
    config: &LlamaConfig,
) -> Result<LlamaRecord<WorkerBackend>, WorkerError> {
    let embed = load_embedding(
        weights,
        "model.embed_tokens",
        config.vocab_size,
        config.hidden_size,
    )?;
    let norm = load_norm(weights, "model.norm", config.hidden_size)?;
    let lm_head = load_linear(
        weights,
        "lm_head",
        config.hidden_size,
        config.vocab_size,
        false,
    )?;

    let mut layers = Vec::new();
    for i in 0..config.num_layers {
        let prefix = format!("model.layers.{}", i);

        let q = load_linear(
            weights,
            &format!("{}.self_attn.q_proj", prefix),
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            false,
        )?;
        let k = load_linear(
            weights,
            &format!("{}.self_attn.k_proj", prefix),
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            false,
        )?;
        let v = load_linear(
            weights,
            &format!("{}.self_attn.v_proj", prefix),
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            false,
        )?;
        let o = load_linear(
            weights,
            &format!("{}.self_attn.o_proj", prefix),
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            false,
        )?;

        // LlamaAttentionRecord
        // Attention has q,k,v,o
        let attention = LlamaAttentionRecord {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            num_heads: ConstantRecord,
            num_kv_heads: ConstantRecord,
            head_dim: ConstantRecord,
        };

        let gate = load_linear(
            weights,
            &format!("{}.mlp.gate_proj", prefix),
            config.hidden_size,
            config.intermediate_size,
            false,
        )?;
        let up = load_linear(
            weights,
            &format!("{}.mlp.up_proj", prefix),
            config.hidden_size,
            config.intermediate_size,
            false,
        )?;
        let down = load_linear(
            weights,
            &format!("{}.mlp.down_proj", prefix),
            config.intermediate_size,
            config.hidden_size,
            false,
        )?;

        let mlp = LlamaMLPRecord {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        };

        let in_norm = load_norm(
            weights,
            &format!("{}.input_layernorm", prefix),
            config.hidden_size,
        )?;
        let post_norm = load_norm(
            weights,
            &format!("{}.post_attention_layernorm", prefix),
            config.hidden_size,
        )?;

        layers.push(LlamaLayerRecord {
            attention,
            mlp,
            input_layernorm: in_norm,
            post_attention_layernorm: post_norm,
        });
    }

    let rope = RotaryEmbeddingRecord {
        cos: ConstantRecord,
        sin: ConstantRecord,
    };

    Ok(LlamaRecord {
        embed_tokens: embed,
        layers,
        norm,
        lm_head,
        config: ConstantRecord,
        rope,
        tokenizer: ConstantRecord,
        device: ConstantRecord,
        eos_token_ids: ConstantRecord,
    })
}
