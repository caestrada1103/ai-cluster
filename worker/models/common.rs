//! Common model utilities shared across architectures.
//!
//! Re-exported via `pub use common::*` in `mod.rs`.

use burn::module::Module;
use burn::nn;
use burn::tensor::{backend::Backend, Tensor};

// ---------------------------------------------------------------------------
// RMS Normalization
// ---------------------------------------------------------------------------

/// RMS Normalization layer (used by Llama, DeepSeek, Mistral, etc.).
#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    /// Learned scale parameter.
    pub weight: Tensor<B, 1>,
    /// Small epsilon for numerical stability.
    #[module(skip)]
    pub eps: f64,
}

impl<B: Backend> RMSNorm<B> {
    /// Create a new RMSNorm layer.
    pub fn new(hidden_size: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Tensor::ones([hidden_size], device),
            eps,
        }
    }

    /// Apply RMS normalization.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // variance = mean(x^2, dim=-1, keepdim=True)
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        let x_normed = x * (variance + self.eps).sqrt().recip();
        x_normed * self.weight.clone().unsqueeze()
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embeddings (RoPE)
// ---------------------------------------------------------------------------

/// Precomputed rotary embeddings for efficient position encoding.
#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    /// Cosine component: shape [max_seq_len, head_dim]
    pub cos: Tensor<B, 2>,
    /// Sine component: shape [max_seq_len, head_dim]
    pub sin: Tensor<B, 2>,
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Precompute rotary embeddings up to `max_seq_len`.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32, device: &B::Device) -> Self {
        let half_dim = head_dim / 2;

        // inv_freq = 1.0 / (theta ^ (2i / dim))  for i in 0..half_dim
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        let inv_freq_tensor = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);

        // positions = [0, 1, 2, ..., max_seq_len - 1]
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let pos_tensor = Tensor::<B, 1>::from_floats(positions.as_slice(), device);

        // freqs = outer(positions, inv_freq)  -> [max_seq_len, half_dim]
        let freqs = pos_tensor.unsqueeze::<2>().transpose()
            .matmul(inv_freq_tensor.unsqueeze::<2>());

        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        Self { cos, sin }
    }

    /// Apply rotary embedding to query and key tensors.
    ///
    /// Expects `q` and `k` of shape `[batch, heads, seq_len, head_dim]`.
    pub fn apply(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        start_pos: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let seq_len = q.dims()[2];

        // Slice the precomputed cos/sin to the relevant positions
        let cos = self.cos.clone().slice([start_pos..start_pos + seq_len]);
        let sin = self.sin.clone().slice([start_pos..start_pos + seq_len]);

        let q_rotated = Self::rotate_half(q, cos.clone(), sin.clone());
        let k_rotated = Self::rotate_half(k, cos, sin);

        (q_rotated, k_rotated)
    }

    fn rotate_half(x: Tensor<B, 4>, cos: Tensor<B, 2>, sin: Tensor<B, 2>) -> Tensor<B, 4> {
        let dims = x.dims();
        let head_dim = dims[3];
        let half = head_dim / 2;

        let x1 = x.clone().slice([0..dims[0], 0..dims[1], 0..dims[2], 0..half]);
        let x2 = x.slice([0..dims[0], 0..dims[1], 0..dims[2], half..head_dim]);

        let cos = cos.unsqueeze::<4>();
        let sin = sin.unsqueeze::<4>();

        let rotated_x1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let rotated_x2 = x2 * cos + x1 * sin;

        Tensor::cat(vec![rotated_x1, rotated_x2], 3)
    }
}

// ---------------------------------------------------------------------------
// GQA helpers
// ---------------------------------------------------------------------------

/// Repeat KV heads to match the number of query heads in Grouped-Query Attention.
///
/// Input: `[batch, kv_heads, seq_len, head_dim]`
/// Output: `[batch, num_heads, seq_len, head_dim]`
pub fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }

    let [batch, kv_heads, seq_len, head_dim] = x.dims();

    // Expand kv_heads dimension by repeating
    let expanded: Tensor<B, 5> = x
        .unsqueeze_dim(2) // [batch, kv_heads, 1, seq_len, head_dim]
        .repeat_dim(2, n_rep); // [batch, kv_heads, n_rep, seq_len, head_dim]

    expanded.reshape([batch, kv_heads * n_rep, seq_len, head_dim])
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Apply top-k and top-p (nucleus) filtering to logits, then sample.
///
/// Returns the sampled token index.
pub fn top_k_top_p_sample(logits: &[f32], temperature: f32, top_p: f32, top_k: usize) -> usize {
    if logits.is_empty() {
        return 0;
    }

    // Apply temperature
    let scaled: Vec<f32> = logits
        .iter()
        .map(|&l| l / temperature.max(1e-8))
        .collect();

    // Softmax
    let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let mut probs: Vec<(usize, f32)> = exps
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e / sum_exp))
        .collect();

    // Sort descending by probability
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k: keep only the k highest probability tokens
    if top_k > 0 && top_k < probs.len() {
        probs.truncate(top_k);
    }

    // Top-p: keep tokens whose cumulative probability <= top_p
    if top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut cutoff = probs.len();
        for (i, &(_, p)) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    // Re-normalise
    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    for item in probs.iter_mut() {
        item.1 /= total;
    }

    // Sample (deterministic fallback: argmax)
    // In production this would use a proper RNG; for now pick the most likely.
    probs[0].0
}

// ---------------------------------------------------------------------------
// SwiGLU activation
// ---------------------------------------------------------------------------

/// SwiGLU activation: SiLU(gate) * up
pub fn swiglu<B: Backend>(gate: Tensor<B, 3>, up: Tensor<B, 3>) -> Tensor<B, 3> {
    // SiLU(x) = x * sigmoid(x)
    let sigmoid = gate.clone().neg().exp().add_scalar(1.0).recip();
    let silu = gate * sigmoid;
    silu * up
}
