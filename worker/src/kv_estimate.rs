//! Weights-independent KV-cache + compute-buffer byte estimate.
//!
//! Always compiled (no `llamacpp` feature) so the in-process llama.cpp engine
//! and the `llamaserver` admission check share identical arithmetic.

/// Context window used when a caller supplies no `n_ctx`.
pub const DEFAULT_N_CTX: u32 = 4096;

/// Discount for hybrid-attention architectures (Qwen3.5/3.6, Qwen3-Next),
/// whose recurrent layers hold a small fixed-size state, not one that scales
/// with `n_ctx`. 0.6 stays a safe over-estimate vs. the ~0.51 measured ratio.
pub const HYBRID_ATTENTION_DISCOUNT: f64 = 0.6;

/// Bytes per KV element for a ggml cache type name. Unknown/`None` defaults
/// to `f16` (2 bytes) — llama.cpp's own default, and the safe over-estimate.
pub fn kv_cache_type_bytes(name: Option<&str>) -> f64 {
    match name {
        Some("q8_0") => 34.0 / 32.0,
        Some("q5_1") => 24.0 / 32.0,
        Some("q5_0") => 22.0 / 32.0,
        Some("q4_1") => 20.0 / 32.0,
        Some("q4_0") => 18.0 / 32.0,
        _ => 2.0,
    }
}

/// Linear fit for llama.cpp's compute buffers, measured on a 35B dense
/// model: 493 MiB at n_ctx=131072, 820 MiB at n_ctx=262144. Approximation.
pub fn compute_buffer_bytes(n_ctx: u64) -> u64 {
    const N_CTX_LOW: f64 = 131_072.0;
    const BYTES_LOW: f64 = 493.0 * 1024.0 * 1024.0;
    const N_CTX_HIGH: f64 = 262_144.0;
    const BYTES_HIGH: f64 = 820.0 * 1024.0 * 1024.0;
    const SLOPE: f64 = (BYTES_HIGH - BYTES_LOW) / (N_CTX_HIGH - N_CTX_LOW);
    const INTERCEPT: f64 = BYTES_LOW - N_CTX_LOW * SLOPE;

    (INTERCEPT + n_ctx as f64 * SLOPE).max(0.0) as u64
}

/// KV-cache + compute-buffer estimate for `slots` concurrent contexts. K/V
/// are sized separately (independently quantizable); the per-slot total is
/// multiplied by `slots` since each slot carries its own cache + buffer.
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_bytes_raw(
    n_ctx: u64,
    n_layer: u64,
    n_head_kv: u64,
    head_dim: u64,
    is_hybrid: bool,
    cache_type_k: Option<&str>,
    cache_type_v: Option<&str>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_buffer_bytes_reproduces_the_two_measured_points() {
        let low = 493 * 1024 * 1024;
        let high = 820 * 1024 * 1024;
        assert!(compute_buffer_bytes(131_072).abs_diff(low) <= 1);
        assert!(compute_buffer_bytes(262_144).abs_diff(high) <= 1);
    }

    #[test]
    fn compute_buffer_bytes_grows_with_context() {
        assert!(compute_buffer_bytes(4_096) < compute_buffer_bytes(131_072));
        assert!(compute_buffer_bytes(131_072) < compute_buffer_bytes(262_144));
        assert!(compute_buffer_bytes(0) > 0, "fixed intercept must remain");
    }

    #[test]
    fn kv_cache_bytes_hybrid_discount_reduces_the_kv_term() {
        let non_hybrid = kv_cache_bytes_raw(131_072, 48, 8, 128, false, None, None, 1);
        let hybrid = kv_cache_bytes_raw(131_072, 48, 8, 128, true, None, None, 1);
        assert!(hybrid < non_hybrid);
        // Only the KV term is discounted, not the compute buffer.
        let compute = compute_buffer_bytes(131_072);
        assert!(hybrid > compute);
    }

    #[test]
    fn kv_cache_bytes_quantized_cache_type_is_smaller_than_f16_default() {
        let f16 = kv_cache_bytes_raw(4_096, 32, 8, 128, false, None, None, 1);
        let q8 = kv_cache_bytes_raw(4_096, 32, 8, 128, false, Some("q8_0"), Some("q8_0"), 1);
        assert!(q8 < f16);
    }

    #[test]
    fn kv_cache_bytes_scales_linearly_with_slots() {
        let one = kv_cache_bytes_raw(131_072, 48, 8, 128, false, None, None, 1);
        let eight = kv_cache_bytes_raw(131_072, 48, 8, 128, false, None, None, 8);
        assert_eq!(eight, one * 8);
    }

    #[test]
    fn kv_cache_bytes_zero_slots_treated_as_one() {
        let zero = kv_cache_bytes_raw(4_096, 32, 8, 128, false, None, None, 0);
        let one = kv_cache_bytes_raw(4_096, 32, 8, 128, false, None, None, 1);
        assert_eq!(zero, one);
    }
}
