//! llama.cpp inference engine (cargo feature `llamacpp`).
//!
//! Wraps the `llama-cpp-2` bindings behind [`TextGeneration`] so GGUF models
//! plug in like the Burn engines: one [`LlamaModel`] shared via `Arc`, one
//! [`LlamaContext`] per generation call inside a `spawn_blocking` (not
//! `Send`), streamed to async via mpsc.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use async_stream::stream;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use once_cell::sync::Lazy;
use tracing::{debug, info};

use crate::error::WorkerError;
use crate::models::{TextGeneration, TextStream};

/// Context window used when the registry supplies no `n_ctx` (clamped to the
/// model's trained context below).
const DEFAULT_N_CTX: u32 = 4096;

/// Process-wide llama.cpp backend. `llama_backend_init` may only run once per
/// process (llama-cpp-2 enforces this with an AtomicBool), so memoize it.
static LLAMA_BACKEND: Lazy<Result<Arc<LlamaBackend>, String>> = Lazy::new(|| {
    LlamaBackend::init()
        .map(Arc::new)
        .map_err(|e| format!("llama.cpp backend init failed: {e}"))
});

fn backend() -> Result<Arc<LlamaBackend>, WorkerError> {
    LLAMA_BACKEND.clone().map_err(WorkerError::ModelLoad)
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Sampling decisions derived from an inference request. Pure data so the
/// request→sampler mapping is unit-testable without llama.cpp state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerParams {
    /// Argmax sampling (temperature ≈ 0), mirroring the Burn path's cutoff.
    pub greedy: bool,
    /// Top-k filter; `None` disables it (request sent k = 0).
    pub top_k: Option<i32>,
    /// Top-p (nucleus) filter; `None` disables it (p ≤ 0 or p ≥ 1).
    pub top_p: Option<f32>,
    /// Softmax temperature (only meaningful when `greedy` is false).
    pub temperature: f32,
}

/// Map the request's sampling fields to [`SamplerParams`].
///
/// Mirrors `worker/models/llama.rs`: temperature below 0.01 means greedy.
pub fn sampler_params(temperature: f32, top_p: f32, top_k: usize) -> SamplerParams {
    if temperature < 0.01 {
        return SamplerParams {
            greedy: true,
            top_k: None,
            top_p: None,
            temperature: 0.0,
        };
    }
    SamplerParams {
        greedy: false,
        top_k: (top_k > 0).then_some(top_k as i32),
        top_p: (top_p > 0.0 && top_p < 1.0).then_some(top_p),
        temperature,
    }
}

/// Build the llama.cpp sampler chain for one generation call.
fn build_sampler(params: SamplerParams, seed: u32) -> LlamaSampler {
    if params.greedy {
        return LlamaSampler::greedy();
    }
    let mut chain: Vec<LlamaSampler> = Vec::with_capacity(4);
    if let Some(k) = params.top_k {
        chain.push(LlamaSampler::top_k(k));
    }
    if let Some(p) = params.top_p {
        chain.push(LlamaSampler::top_p(p, 1));
    }
    chain.push(LlamaSampler::temp(params.temperature));
    chain.push(LlamaSampler::dist(seed));
    LlamaSampler::chain_simple(chain)
}

/// Map a ggml type name to the crate's [`KvCacheType`]. Mirrors the allowed
/// set validated upstream in `model_loader::ALLOWED_KV_CACHE_TYPES`, which
/// has no llama.cpp dependency.
fn parse_kv_cache_type(name: &str) -> Result<KvCacheType, WorkerError> {
    match name {
        "f16" => Ok(KvCacheType::F16),
        "q8_0" => Ok(KvCacheType::Q8_0),
        "q4_0" => Ok(KvCacheType::Q4_0),
        "q5_0" => Ok(KvCacheType::Q5_0),
        "q5_1" => Ok(KvCacheType::Q5_1),
        "q4_1" => Ok(KvCacheType::Q4_1),
        other => Err(WorkerError::Configuration(format!(
            "unknown KV cache type '{other}' (expected one of f16, q8_0, q4_0, q5_0, q5_1, q4_1)"
        ))),
    }
}

/// Reverse of [`parse_kv_cache_type`] — needed to call the shared,
/// string-keyed [`crate::kv_estimate::kv_cache_type_bytes`] from here.
fn kv_cache_type_name(t: KvCacheType) -> &'static str {
    match t {
        KvCacheType::Q8_0 => "q8_0",
        KvCacheType::Q5_1 => "q5_1",
        KvCacheType::Q5_0 => "q5_0",
        KvCacheType::Q4_1 => "q4_1",
        KvCacheType::Q4_0 => "q4_0",
        _ => "f16",
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// A GGUF model served by llama.cpp.
pub struct LlamaCppEngine {
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
    n_ctx: Option<u32>,
    n_threads: i32,
    /// KV-cache quantization for K/V. `None` leaves llama.cpp's own default
    /// (f16) untouched.
    cache_type_k: Option<KvCacheType>,
    cache_type_v: Option<KvCacheType>,
}

impl LlamaCppEngine {
    /// Load a GGUF file. Blocking (mmap + optional GPU upload) — callers must
    /// wrap this in `tokio::task::spawn_blocking`.
    ///
    /// * `n_gpu_layers` — layers to offload to the GPU; `-1` offloads all
    ///   (mapped to `u32::MAX`, which llama.cpp clamps to "all layers").
    /// * `n_ctx` — per-generation context window (`None` = min(trained, 4096)).
    /// * `n_threads` — CPU threads for generation (`0` = llama.cpp default).
    /// * `cache_type_k`/`cache_type_v` — KV-cache quantization type names,
    ///   already validated by `model_loader::gguf_spec_from_metadata`.
    ///   `None` leaves llama.cpp's own default (f16) in effect.
    pub fn load(
        gguf_path: &Path,
        n_gpu_layers: i32,
        n_ctx: Option<u32>,
        n_threads: i32,
        cache_type_k: Option<String>,
        cache_type_v: Option<String>,
    ) -> Result<Self, WorkerError> {
        let cache_type_k = cache_type_k
            .as_deref()
            .map(parse_kv_cache_type)
            .transpose()?;
        let cache_type_v = cache_type_v
            .as_deref()
            .map(parse_kv_cache_type)
            .transpose()?;
        let backend = backend()?;
        let offload = if n_gpu_layers < 0 {
            u32::MAX
        } else {
            n_gpu_layers as u32
        };
        let model_params = LlamaModelParams::default().with_n_gpu_layers(offload);
        let model =
            LlamaModel::load_from_file(&backend, gguf_path, &model_params).map_err(|e| {
                WorkerError::ModelLoad(format!(
                    "llama.cpp failed to load {}: {e}",
                    gguf_path.display()
                ))
            })?;
        info!(
            "llama.cpp model loaded: {} (trained ctx {}, eos token {:?})",
            gguf_path.display(),
            model.n_ctx_train(),
            model.token_eos(),
        );
        Ok(Self {
            model: Arc::new(model),
            backend,
            n_ctx,
            n_threads,
            cache_type_k,
            cache_type_v,
        })
    }

    /// Estimated KV-cache + compute-buffer footprint in bytes for `slots`
    /// concurrent contexts (each generation call opens a fresh
    /// `LlamaContext`, so pass worst-case concurrency, not `1`). See
    /// [`crate::kv_estimate`] for the shared math.
    pub fn kv_cache_bytes(&self, slots: u32) -> u64 {
        let n_ctx_train = self.model.n_ctx_train().max(1);
        let n_ctx = u64::from(self.n_ctx.unwrap_or(DEFAULT_N_CTX).min(n_ctx_train));
        let n_layer = u64::from(self.model.n_layer());
        let n_head_kv = u64::from(self.model.n_head_kv());
        // head_dim = n_embd / n_head; guard n_head to avoid a divide-by-zero on
        // a malformed GGUF header.
        let n_head = u64::from(self.model.n_head()).max(1);
        let head_dim = (self.model.n_embd().max(0) as u64) / n_head;

        crate::kv_estimate::kv_cache_bytes_raw(
            n_ctx,
            n_layer,
            n_head_kv,
            head_dim,
            self.model.is_hybrid(),
            self.cache_type_k.map(kv_cache_type_name),
            self.cache_type_v.map(kv_cache_type_name),
            slots,
        )
    }
}

/// Everything one blocking generation needs (bundled to keep
/// `run_generation` clippy-clean on argument count).
struct GenArgs {
    prompt: String,
    max_tokens: usize,
    params: SamplerParams,
    seed: u32,
    n_ctx: Option<u32>,
    n_threads: i32,
    cache_type_k: Option<KvCacheType>,
    cache_type_v: Option<KvCacheType>,
}

/// Prefill + decode loop. Runs entirely on one blocking thread and streams
/// token pieces through `tx` — the exact shape of the Burn loop in
/// `worker/models/llama.rs::generate`.
fn run_generation(
    model: &LlamaModel,
    backend: &LlamaBackend,
    args: &GenArgs,
    tx: &tokio::sync::mpsc::Sender<Result<String, WorkerError>>,
) -> Result<(), WorkerError> {
    // ── Tokenize (BOS handling comes from GGUF metadata) ──────────────────
    let tokens = model
        .str_to_token(&args.prompt, AddBos::Always)
        .map_err(|e| WorkerError::Internal(format!("llama.cpp tokenize error: {e}")))?;

    // ── Context sizing: registry n_ctx, else default, clamped to trained ──
    let n_ctx_train = model.n_ctx_train().max(1);
    let n_ctx = args.n_ctx.unwrap_or(DEFAULT_N_CTX).min(n_ctx_train);
    if tokens.len() + 1 > n_ctx as usize {
        return Err(WorkerError::Internal(format!(
            "prompt is {} tokens but the llama.cpp context window is {n_ctx}",
            tokens.len()
        )));
    }

    let mut ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx); // whole prompt prefills in one decode call
    if let Some(type_k) = args.cache_type_k {
        ctx_params = ctx_params.with_type_k(type_k);
    }
    if let Some(type_v) = args.cache_type_v {
        ctx_params = ctx_params.with_type_v(type_v);
    }
    if args.n_threads > 0 {
        ctx_params = ctx_params
            .with_n_threads(args.n_threads)
            .with_n_threads_batch(args.n_threads);
    }
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| WorkerError::Internal(format!("llama.cpp context error: {e}")))?;

    // ── PREFILL ────────────────────────────────────────────────────────────
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let last_index = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        // Logits are only needed at the last prompt position.
        batch
            .add(token, i, &[0], i == last_index)
            .map_err(|e| WorkerError::Internal(format!("llama.cpp batch error: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| WorkerError::Internal(format!("llama.cpp prefill error: {e}")))?;

    // ── DECODE LOOP ────────────────────────────────────────────────────────
    let mut sampler = build_sampler(args.params, args.seed);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = batch.n_tokens();
    let mut generated = 0usize;

    while generated < args.max_tokens && (n_cur as u32) < n_ctx {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        // EOS/EOG comes from GGUF metadata — never a hardcoded string.
        if model.is_eog_token(token) {
            debug!("llama.cpp generation hit EOG after {generated} tokens");
            break;
        }

        // Incremental detokenization. `special = false` mirrors the Burn
        // path's skip_special_tokens decode; the stateful UTF-8 decoder holds
        // back incomplete multi-byte sequences between pieces.
        let piece = model
            .token_to_piece(token, &mut decoder, false, None)
            .map_err(|e| WorkerError::Internal(format!("llama.cpp detokenize error: {e}")))?;
        if !piece.is_empty() && tx.blocking_send(Ok(piece)).is_err() {
            // Receiver dropped — client disconnected; stop generating.
            break;
        }
        generated += 1;

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| WorkerError::Internal(format!("llama.cpp batch error: {e}")))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| WorkerError::Internal(format!("llama.cpp decode error: {e}")))?;
    }
    Ok(())
}

impl TextGeneration for LlamaCppEngine {
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: Option<u64>,
    ) -> Result<TextStream, WorkerError> {
        // llama-cpp-2's sampler seed is u32; the trait's is u64/Option.
        // Truncate a real seed, or derive one from the clock for "random".
        let seed: u32 = seed.map(|s| s as u32).unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0xDEAD_BEEF)
        });

        let args = GenArgs {
            prompt: prompt.to_string(),
            max_tokens,
            params: sampler_params(temperature, top_p, top_k),
            seed,
            n_ctx: self.n_ctx,
            n_threads: self.n_threads,
            cache_type_k: self.cache_type_k,
            cache_type_v: self.cache_type_v,
        };
        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);

        // Single spawn_blocking for the entire prefill + decode loop, bridged
        // to the async gRPC layer via an mpsc channel (same as models/llama.rs).
        let (tx, mut rx) =
            tokio::sync::mpsc::channel::<Result<String, WorkerError>>(max_tokens + 2);

        tokio::task::spawn_blocking(move || {
            if let Err(e) = run_generation(&model, &backend, &args, &tx) {
                let _ = tx.blocking_send(Err(e));
            }
        });

        let stream = stream! {
            while let Some(item) = rx.recv().await {
                yield item;
            }
        };
        Ok(Box::pin(stream))
    }

    fn count_prompt_tokens(&self, prompt: &str) -> Option<u32> {
        match self.model.str_to_token(prompt, AddBos::Always) {
            Ok(tokens) => Some(tokens.len() as u32),
            Err(e) => {
                debug!("llama.cpp count_prompt_tokens tokenize error: {e}");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // KV-cache/compute-buffer math itself now lives in, and is tested by,
    // `crate::kv_estimate` (shared with the llamaserver admission check).

    #[test]
    fn kv_cache_type_name_round_trips_through_the_shared_estimator() {
        for (t, name) in [
            (KvCacheType::Q8_0, "q8_0"),
            (KvCacheType::Q5_1, "q5_1"),
            (KvCacheType::Q5_0, "q5_0"),
            (KvCacheType::Q4_1, "q4_1"),
            (KvCacheType::Q4_0, "q4_0"),
            (KvCacheType::F16, "f16"),
        ] {
            assert_eq!(kv_cache_type_name(t), name);
            assert_eq!(
                crate::kv_estimate::kv_cache_type_bytes(Some(kv_cache_type_name(t))),
                crate::kv_estimate::kv_cache_type_bytes(Some(name))
            );
        }
    }

    #[test]
    fn sampler_params_greedy_when_temperature_near_zero() {
        // Mirrors the Burn path's cutoff in worker/models/llama.rs: < 0.01 = argmax.
        let p = sampler_params(0.0, 0.95, 40);
        assert_eq!(
            p,
            SamplerParams {
                greedy: true,
                top_k: None,
                top_p: None,
                temperature: 0.0
            }
        );
        assert!(sampler_params(0.009, 0.95, 40).greedy);
        assert!(!sampler_params(0.011, 0.95, 40).greedy);
    }

    #[test]
    fn sampler_params_full_chain() {
        let p = sampler_params(0.7, 0.95, 40);
        assert_eq!(
            p,
            SamplerParams {
                greedy: false,
                top_k: Some(40),
                top_p: Some(0.95),
                temperature: 0.7
            }
        );
    }

    #[test]
    fn sampler_params_disables_degenerate_filters() {
        // top_k = 0 means "no top-k"; top_p >= 1.0 means "no nucleus filter".
        let p = sampler_params(1.0, 1.0, 0);
        assert_eq!(
            p,
            SamplerParams {
                greedy: false,
                top_k: None,
                top_p: None,
                temperature: 1.0
            }
        );
    }

    #[test]
    fn build_sampler_constructs_for_all_param_shapes() {
        // Smoke test: chain construction must not panic in either mode.
        let _greedy = build_sampler(sampler_params(0.0, 0.95, 40), 42);
        let _chain = build_sampler(sampler_params(0.8, 0.9, 50), 42);
    }

    #[test]
    fn parse_kv_cache_type_accepts_every_allowed_name() {
        assert_eq!(parse_kv_cache_type("f16").unwrap(), KvCacheType::F16);
        assert_eq!(parse_kv_cache_type("q8_0").unwrap(), KvCacheType::Q8_0);
        assert_eq!(parse_kv_cache_type("q4_0").unwrap(), KvCacheType::Q4_0);
        assert_eq!(parse_kv_cache_type("q5_0").unwrap(), KvCacheType::Q5_0);
        assert_eq!(parse_kv_cache_type("q5_1").unwrap(), KvCacheType::Q5_1);
        assert_eq!(parse_kv_cache_type("q4_1").unwrap(), KvCacheType::Q4_1);
    }

    #[test]
    fn parse_kv_cache_type_rejects_unknown_name() {
        assert!(parse_kv_cache_type("int8").is_err());
        assert!(parse_kv_cache_type("fp16").is_err());
    }

    /// End-to-end smoke test: downloads a ~1.1 MiB GGUF and generates text on
    /// CPU. Requires network; excluded from default runs. Execute with:
    ///   cd worker && cargo test --features llamacpp -- --ignored llamacpp
    #[test]
    #[ignore]
    fn llamacpp_e2e_tiny_gguf_generation() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime");
        rt.block_on(async {
            let api = hf_hub::api::tokio::ApiBuilder::new()
                .build()
                .expect("build hf-hub api");
            let repo = api.repo(hf_hub::Repo::new(
                "ggml-org/models".to_string(),
                hf_hub::RepoType::Model,
            ));
            let gguf_path = repo
                .get("tinyllamas/stories260K.gguf")
                .await
                .expect("download tinyllamas/stories260K.gguf (~1.1 MiB, needs network)");

            // n_gpu_layers = 0 → pure CPU, so this runs on any machine.
            // stories260K was trained with a 512-token context. No KV-cache
            // quantization override (None, None) — today's f16 default.
            let engine = LlamaCppEngine::load(&gguf_path, 0, Some(512), 0, None, None)
                .expect("load tiny GGUF via llama.cpp");

            // Greedy (temperature 0.0) for a deterministic, non-empty result.
            let mut stream = engine
                .generate("Once upon a time", 16, 0.0, 0.95, 40, None)
                .expect("start generation");

            use futures::StreamExt;
            let mut text = String::new();
            while let Some(chunk) = stream.next().await {
                text.push_str(&chunk.expect("stream chunk is Ok"));
            }
            assert!(
                !text.trim().is_empty(),
                "expected non-empty generated text, got: {text:?}"
            );

            // Real tokenizer count: positive, grows with prompt length, deterministic.
            let short = engine
                .count_prompt_tokens("Once upon a time")
                .expect("tokenizer count for short prompt");
            assert!(short > 0);
            let long = engine
                .count_prompt_tokens("Once upon a time, in a land far, far away, there lived a")
                .expect("tokenizer count for long prompt");
            assert!(long > short, "longer prompt should yield more tokens");
            let short_again = engine.count_prompt_tokens("Once upon a time").unwrap();
            assert_eq!(
                short, short_again,
                "same prompt should count deterministically"
            );
        });
    }
}
