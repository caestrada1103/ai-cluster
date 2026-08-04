//! llama.cpp inference engine (cargo feature `llamacpp`).
//!
//! Wraps the `llama-cpp-2` bindings (llama.cpp compiled from source) behind
//! the existing [`TextGeneration`] trait so GGUF models plug into
//! `ModelInstance` exactly like the Burn engines.
//!
//! Concurrency model (matches llama-cpp-2 reality):
//! * one [`LlamaModel`] per loaded model, shared via `Arc` (the crate marks it
//!   `Send + Sync`);
//! * one [`LlamaContext`] (own KV cache) created per generation call, entirely
//!   inside a single `spawn_blocking` closure — the context is not `Send` and
//!   never crosses threads;
//! * token pieces stream to the async side through an mpsc channel, mirroring
//!   the Burn pattern in `worker/models/llama.rs::generate`.

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

// ---------------------------------------------------------------------------
// KV-cache quantization (Task 2b)
// ---------------------------------------------------------------------------

/// Map a ggml type name to the crate's [`KvCacheType`].
///
/// The allowed set is validated once upstream, in
/// `model_loader::gguf_spec_from_metadata` (`ALLOWED_KV_CACHE_TYPES`), which
/// has no llama.cpp dependency; this mirrors that same set so the two stay in
/// lockstep. `f16` maps explicitly even though it's llama.cpp's own default —
/// see `KvCacheType`'s doc examples in the vendored crate for the exact enum
/// shape (`with_type_k(KvCacheType::Q4_0)`).
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

/// Bytes per KV element for a ggml cache type.
///
/// Block-quantized types store 32 elements plus per-block scale/min metadata,
/// so the per-element cost is fractional (e.g. `q8_0` = 34 bytes / 32 elements).
/// `None` means llama.cpp's own default, which is `f16`. Any type outside the
/// validated set falls back to the `f16` cost — deliberately the most
/// pessimistic of the supported set, so an unknown type over-reserves rather
/// than under-reserves.
fn kv_cache_type_bytes(t: Option<KvCacheType>) -> f64 {
    match t {
        Some(KvCacheType::Q8_0) => 34.0 / 32.0,
        Some(KvCacheType::Q5_1) => 24.0 / 32.0,
        Some(KvCacheType::Q5_0) => 22.0 / 32.0,
        Some(KvCacheType::Q4_1) => 20.0 / 32.0,
        Some(KvCacheType::Q4_0) => 18.0 / 32.0,
        _ => 2.0,
    }
}

// ---------------------------------------------------------------------------
// KV-cache / compute-buffer sizing (Task: CLAUDE.md fix 3)
// ---------------------------------------------------------------------------

/// Discount applied to the attention-layer KV-cache estimate for
/// hybrid-attention models (`LlamaModel::is_hybrid()` — Qwen3.5/3.6,
/// Qwen3-Next "Gated DeltaNet", and similarly-shaped Jamba/Falcon-H1/
/// Nemotron-H-style architectures). These interleave a minority of real
/// full-attention layers with recurrent/SSM layers that hold a small
/// **fixed-size** state buffer which does NOT scale with `n_ctx` — treating
/// every layer as a full attention layer (the un-discounted formula) badly
/// over-counts for them.
///
/// `llama-cpp-2 = 0.1.150` exposes no per-layer attention-type query and no
/// recurrent-state-size query through its public API: `llama_model_n_head_kv`
/// and `llama_model_n_swa` take no layer index, and this version's
/// `llama.cpp` vendors no `llama_model_n_embd_k_s`/`_v_s`-style getter either
/// (checked directly against the vendored C headers/source under
/// `~/.cargo/registry/src/*/llama-cpp-sys-2-0.1.150/llama.cpp/`, specifically
/// `include/llama.h` and `src/llama-model.cpp`'s `llm_arch_is_hybrid`/
/// `hparams.is_recr(il)` machinery, which is internal and not exported). The
/// only signal available through the crate is the boolean
/// `LlamaModel::is_hybrid()` (`llama_model_is_hybrid`), derived from the
/// GGUF's architecture id.
///
/// Measured on hardware for one hybrid model (Qwen3-family): treating every
/// layer as a full attention layer estimated 2.66 GiB where llama.cpp
/// actually allocated 1360 MiB — real:estimated ≈ 0.51. Different hybrid
/// families interleave attention/recurrent layers at different ratios (per
/// `llama-model.cpp`, Falcon-H1 layers carry BOTH an attention AND a
/// recurrent component per layer; Nemotron-H's attention-layer fraction is
/// much smaller than Qwen3.5/3.6's), so 0.51 is a data point, not a universal
/// constant. We deliberately apply a discount ABOVE the one measured ratio
/// (0.6 rather than 0.51) so this stays a safe OVER-estimate — under-counting
/// KV-cache memory risks OOM-killing the process, which is the failure mode
/// worth avoiding, whereas over-reserving only costs some admission headroom.
///
/// Residual risk: this is a single documented heuristic applied uniformly to
/// every `is_hybrid()` model, not a per-architecture computation — it may
/// still be too generous for a hybrid family whose real attention-layer
/// fraction is far below 60% of the naive estimate (e.g. Nemotron-H-shaped
/// models with very few attention layers), or (rarely) too tight for one
/// where recurrent layers are unusually memory-heavy. Revisit if/when
/// `llama-cpp-2` exposes a per-layer or per-architecture breakdown.
const HYBRID_ATTENTION_DISCOUNT: f64 = 0.6;

/// Linear fit for llama.cpp's own COMPUTE buffers (distinct from — and
/// previously entirely ignored by — the KV-cache estimate). Measured on
/// hardware for a 35B dense model: 493 MiB at `n_ctx=131072`, 820 MiB at
/// `n_ctx=262144`. This is two data points' worth of slope/intercept for one
/// model, not a per-architecture computation (compute-buffer size also
/// depends on batch size, model width, and llama.cpp's own internal
/// scheduling, none of which this function has access to) — treat it as a
/// documented, better-than-zero approximation rather than an exact figure.
fn compute_buffer_bytes(n_ctx: u64) -> u64 {
    const N_CTX_LOW: f64 = 131_072.0;
    const BYTES_LOW: f64 = 493.0 * 1024.0 * 1024.0;
    const N_CTX_HIGH: f64 = 262_144.0;
    const BYTES_HIGH: f64 = 820.0 * 1024.0 * 1024.0;
    const SLOPE: f64 = (BYTES_HIGH - BYTES_LOW) / (N_CTX_HIGH - N_CTX_LOW);
    const INTERCEPT: f64 = BYTES_LOW - N_CTX_LOW * SLOPE;

    (INTERCEPT + n_ctx as f64 * SLOPE).max(0.0) as u64
}

/// Pure, weights-independent KV-cache + compute-buffer estimate for
/// `slots` concurrent llama.cpp contexts. Factored out of
/// [`LlamaCppEngine::kv_cache_bytes`] so it's unit-testable without a loaded
/// `LlamaModel`.
///
/// `2 * n_ctx * n_layer * n_head_kv * head_dim * bytes_per_element` is the KV
/// term (factor 2 covers K and V, budgeted separately since they can be
/// quantized independently via `-ctk`/`-ctv`); `is_hybrid` applies
/// [`HYBRID_ATTENTION_DISCOUNT`]; [`compute_buffer_bytes`] adds llama.cpp's
/// compute buffers; the sum is multiplied by `slots` because the in-process
/// engine builds one full `LlamaContext` (its own KV cache AND its own
/// compute buffers) per generation call — see [`LlamaCppEngine::kv_cache_bytes`]'s
/// doc comment for why `slots` must reflect worst-case concurrency, not `1`.
#[allow(clippy::too_many_arguments)]
fn kv_cache_bytes_raw(
    n_ctx: u64,
    n_layer: u64,
    n_head_kv: u64,
    head_dim: u64,
    is_hybrid: bool,
    cache_type_k: Option<KvCacheType>,
    cache_type_v: Option<KvCacheType>,
    slots: u32,
) -> u64 {
    let kv_elements = n_ctx
        .saturating_mul(n_layer)
        .saturating_mul(n_head_kv)
        .saturating_mul(head_dim);
    let per_element = kv_cache_type_bytes(cache_type_k) + kv_cache_type_bytes(cache_type_v);
    let mut kv_bytes = (kv_elements as f64 * per_element) as u64;

    if is_hybrid {
        kv_bytes = (kv_bytes as f64 * HYBRID_ATTENTION_DISCOUNT) as u64;
    }

    let per_slot_bytes = kv_bytes.saturating_add(compute_buffer_bytes(n_ctx));
    per_slot_bytes.saturating_mul(u64::from(slots.max(1)))
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
    /// KV-cache quantization for K/V (Task 2b). `None` leaves llama.cpp's own
    /// default (f16) untouched — byte-for-byte today's behavior.
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
    /// * `cache_type_k`/`cache_type_v` — KV-cache quantization type names
    ///   (Task 2b metadata contract: `f16`/`q8_0`/`q4_0`/`q5_0`/`q5_1`/`q4_1`),
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
    /// concurrent contexts at this engine's configured window.
    ///
    /// # `slots` must be worst-case concurrency, not `1`
    ///
    /// The in-process engine (see this module's top-level doc comment) builds
    /// a **fresh `LlamaContext` — its own KV cache AND its own compute
    /// buffers — per generation call**, inside `spawn_blocking`; contexts are
    /// never pooled or shared across requests. `slots` must therefore be the
    /// number of generation calls that can be in flight simultaneously for
    /// this model, not the number of models loaded. As of this writing the
    /// only caller (`model_loader.rs::load_model`, which this module does not
    /// own) passes a hardcoded `1`, reserving only a single context's worth
    /// of memory even though the worker admits up to `max_concurrent_requests`
    /// (32 by default, `worker.toml`) simultaneous in-flight requests — i.e.
    /// the true worst case for one loaded model can be ~32x what gets
    /// reserved today. Fixing that requires changing the call site to pass
    /// the worker's real concurrency bound (or a per-model cap); see the repo
    /// task notes for the precise call site and the alternative of bounding
    /// admission by available memory instead of by a fixed slot count.
    ///
    /// # What changed vs. the naive formula
    ///
    /// * KV term: `2 * n_ctx * n_layer * n_head_kv * head_dim *
    ///   bytes_per_element` (factor 2 covers K and V, budgeted separately
    ///   since they can be quantized independently via `-ctk`/`-ctv`), now
    ///   discounted by [`HYBRID_ATTENTION_DISCOUNT`] for hybrid-attention
    ///   models (`is_hybrid()`) — see that constant's doc comment for why
    ///   treating every layer as a full attention layer over-counts them by
    ///   roughly 2x, and why a fixed discount rather than an exact
    ///   computation is the best available option today.
    /// * Compute buffers: previously ignored entirely; see
    ///   [`compute_buffer_bytes`].
    /// * Both terms scale with `slots`, matching the fresh-context-per-request
    ///   reality above.
    ///
    /// The context window mirrors [`run_generation`]'s resolution exactly
    /// (`n_ctx.unwrap_or(DEFAULT_N_CTX).min(n_ctx_train)`), so this predicts the
    /// window that will actually be allocated rather than the requested one.
    ///
    /// This remains a weights-independent estimate (excludes allocator
    /// fragmentation and any llama.cpp overhead beyond KV + compute buffers),
    /// so treat it as an approximation, not an exact figure.
    pub fn kv_cache_bytes(&self, slots: u32) -> u64 {
        let n_ctx_train = self.model.n_ctx_train().max(1);
        let n_ctx = u64::from(self.n_ctx.unwrap_or(DEFAULT_N_CTX).min(n_ctx_train));
        let n_layer = u64::from(self.model.n_layer());
        let n_head_kv = u64::from(self.model.n_head_kv());
        // head_dim = n_embd / n_head; guard n_head to avoid a divide-by-zero on
        // a malformed GGUF header.
        let n_head = u64::from(self.model.n_head()).max(1);
        let head_dim = (self.model.n_embd().max(0) as u64) / n_head;

        kv_cache_bytes_raw(
            n_ctx,
            n_layer,
            n_head_kv,
            head_dim,
            self.model.is_hybrid(),
            self.cache_type_k,
            self.cache_type_v,
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
        // llama-cpp-2's sampler seed is u32, but the shared TextGeneration
        // trait (and the proto InferenceRequest.seed it derives from) uses
        // u64/uint32==0-means-random semantics; worker/src/worker.rs maps
        // proto seed 0 -> None. Truncate a real seed to u32, or derive one
        // from the clock when the caller asked for random sampling.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- fix 3b: compute_buffer_bytes ---------------------------------------

    #[test]
    fn compute_buffer_bytes_reproduces_the_two_measured_points() {
        // Measured on hardware for a 35B model: 493 MiB @ n_ctx=131072,
        // 820 MiB @ n_ctx=262144. The linear fit must reproduce both exactly
        // (they're the two points that define it) modulo integer rounding.
        let low = 493u64 * 1024 * 1024;
        let high = 820u64 * 1024 * 1024;
        assert!(compute_buffer_bytes(131_072).abs_diff(low) <= 1);
        assert!(compute_buffer_bytes(262_144).abs_diff(high) <= 1);
    }

    #[test]
    fn compute_buffer_bytes_grows_with_context() {
        // Was ignored entirely before this fix (always 0) — must now scale
        // with n_ctx, matching the observed growth from 493 MiB to 820 MiB.
        assert!(compute_buffer_bytes(4_096) < compute_buffer_bytes(131_072));
        assert!(compute_buffer_bytes(131_072) < compute_buffer_bytes(262_144));
        assert!(compute_buffer_bytes(0) > 0, "fixed intercept must remain");
    }

    // --- fix 3a/3c: kv_cache_bytes_raw --------------------------------------

    #[test]
    fn hybrid_discount_strictly_reduces_the_kv_term() {
        let non_hybrid = kv_cache_bytes_raw(131_072, 48, 8, 128, false, None, None, 1);
        let hybrid = kv_cache_bytes_raw(131_072, 48, 8, 128, true, None, None, 1);
        assert!(
            hybrid < non_hybrid,
            "hybrid models must not be estimated as if every layer were full attention"
        );
        // The discount only touches the KV term, not the (identical) compute
        // buffer term, so it must land near HYBRID_ATTENTION_DISCOUNT of the
        // KV-only delta rather than being unboundedly smaller.
        let compute = compute_buffer_bytes(131_072);
        let kv_non_hybrid = non_hybrid - compute;
        let kv_hybrid = hybrid - compute;
        let ratio = kv_hybrid as f64 / kv_non_hybrid as f64;
        assert!(
            (ratio - HYBRID_ATTENTION_DISCOUNT).abs() < 0.01,
            "ratio {ratio} should match the documented discount"
        );
    }

    #[test]
    fn non_hybrid_models_are_unaffected_by_the_discount() {
        // Guards against accidentally applying HYBRID_ATTENTION_DISCOUNT to
        // ordinary dense models (the vast majority of GGUF checkpoints).
        let bytes = kv_cache_bytes_raw(4_096, 32, 8, 128, false, None, None, 1);
        let elements = 4_096u64 * 32 * 8 * 128;
        let expected_kv = (elements as f64 * 4.0) as u64; // f16 K + f16 V = 2.0 + 2.0
        assert_eq!(bytes, expected_kv + compute_buffer_bytes(4_096));
    }

    #[test]
    fn slots_scales_both_kv_and_compute_terms() {
        // Fix 3c: the in-process engine builds one full context (KV +
        // compute buffers) PER REQUEST, so both terms must scale with
        // `slots`, not just the KV term.
        let one = kv_cache_bytes_raw(131_072, 48, 8, 128, false, None, None, 1);
        let eight = kv_cache_bytes_raw(131_072, 48, 8, 128, false, None, None, 8);
        assert_eq!(eight, one * 8);
    }

    #[test]
    fn slots_zero_is_treated_as_one() {
        let zero = kv_cache_bytes_raw(4_096, 32, 8, 128, false, None, None, 0);
        let one = kv_cache_bytes_raw(4_096, 32, 8, 128, false, None, None, 1);
        assert_eq!(zero, one);
    }

    #[test]
    fn kv_cache_type_bytes_matches_ggml_block_layouts() {
        // f16 is 2 bytes/element; the block types store 32 elements plus
        // scale/min metadata, so they are strictly cheaper than f16.
        assert_eq!(kv_cache_type_bytes(None), 2.0);
        assert_eq!(kv_cache_type_bytes(Some(KvCacheType::F16)), 2.0);
        assert_eq!(kv_cache_type_bytes(Some(KvCacheType::Q8_0)), 34.0 / 32.0);
        assert_eq!(kv_cache_type_bytes(Some(KvCacheType::Q4_0)), 18.0 / 32.0);
        for t in [
            KvCacheType::Q8_0,
            KvCacheType::Q5_1,
            KvCacheType::Q5_0,
            KvCacheType::Q4_1,
            KvCacheType::Q4_0,
        ] {
            assert!(kv_cache_type_bytes(Some(t)) < kv_cache_type_bytes(None));
        }
    }

    #[test]
    fn kv_cache_type_bytes_orders_by_precision() {
        // Fewer bits must never cost more memory.
        assert!(
            kv_cache_type_bytes(Some(KvCacheType::Q8_0))
                > kv_cache_type_bytes(Some(KvCacheType::Q5_1))
        );
        assert!(
            kv_cache_type_bytes(Some(KvCacheType::Q5_1))
                > kv_cache_type_bytes(Some(KvCacheType::Q5_0))
        );
        assert!(
            kv_cache_type_bytes(Some(KvCacheType::Q5_0))
                > kv_cache_type_bytes(Some(KvCacheType::Q4_1))
        );
        assert!(
            kv_cache_type_bytes(Some(KvCacheType::Q4_1))
                > kv_cache_type_bytes(Some(KvCacheType::Q4_0))
        );
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
        });
    }
}
