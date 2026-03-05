//! Model parallelism strategies
//!
//! Implements three parallelism modes for Llama-family models:
//!
//! * **Tensor parallelism** (Megatron-LM style) — column/row-parallel projections

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
//!   with KV-cache-aware autoregressive generation.
//! * **Pipeline parallelism** — layer partitioning across stages with sequential
//!   activation passing.
//! * **Expert parallelism** (MoE) — stub for future Mixture-of-Experts routing.
//!
//! ## Tensor parallelism overview
//!
//! Each transformer layer contains two kinds of linear projections:
//!
//! * **Column-parallel** (Q, K, V, gate, up): the weight matrix is split along
//!   its *output* dimension.  Shard *i* stores `W[:, i·s : (i+1)·s]` and
//!   produces a partial output that is later concatenated (attention context) or
//!   summed (row-parallel step) with the other shards' outputs.
//!
//! * **Row-parallel** (O, down): the weight matrix is split along its *input*
//!   dimension.  Shard *i* stores `W[i·s : (i+1)·s, :]` and contributes a
//!   partial `[batch, seq, hidden]` tensor.  An **all-reduce** (element-wise
//!   sum across shards) gives the correct result because matrix–vector products
//!   distribute over addition:
//!
//!   ```text
//!   [x₀, x₁] · [W₀; W₁] = x₀·W₀ + x₁·W₁
//!   ```
//!
//! When `num_shards == 1` the implementation is mathematically identical to
//! `Llama::forward_pass`.  With `num_shards > 1` each shard would ideally run
//! on a separate GPU; the all-reduce would use NCCL or a custom wgpu compute
//! shader.  The current code simulates all shards on the same device so the
//! result is numerically exact while demonstrating the correct data-flow.
//!
//! ## GQA constraint
//!
//! Grouped-Query Attention requires `num_shards` to evenly divide `num_kv_heads`
//! so that each shard carries the same KV-head / Q-head ratio.  If the requested
//! shard count is not a divisor, it is silently clamped down to the nearest
//! valid value.
//!
//! ## All-reduce abstraction
//!
//! The [`AllReduce`] trait provides a pluggable summation backend.  The default
//! [`LocalAllReduce`] sums partials on the same device.  A real multi-GPU
//! deployment would implement an `NcclAllReduce` or `WgpuAllReduce` variant.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::models::common::repeat_kv;
use crate::models::llama::{Llama, KvEntry};

// ---------------------------------------------------------------------------
// TP-distributed KV cache
// ---------------------------------------------------------------------------

/// Tensor-parallel KV cache: one [`KvEntry`] per (layer, shard).
///
/// Indexed as `tp_kv_cache[layer_idx][shard_idx]`.
/// Each shard stores its own `[1, kv_per_shard, seq_so_far, head_dim]` K/V pair.
pub type TpKvCache<B> = Vec<Vec<KvEntry<B>>>;

// ---------------------------------------------------------------------------
// All-reduce abstraction
// ---------------------------------------------------------------------------

/// Trait for reducing (summing) tensor partials across shards.
///
/// In a single-device simulation, all partials live on the same GPU and are
/// summed directly.  In a real multi-GPU setup, this would wrap NCCL
/// `allReduce(SUM)` or a custom wgpu compute shader.
pub trait AllReduce<B: Backend> {
    /// Element-wise sum of all partials.
    fn sum(&self, partials: Vec<Tensor<B, 2>>) -> Tensor<B, 2>;
}

/// Local (single-device) all-reduce: element-wise sum on the same GPU.
///
/// This is the default used when all shards run on one device.
pub struct LocalAllReduce;

impl<B: Backend> AllReduce<B> for LocalAllReduce {
    fn sum(&self, partials: Vec<Tensor<B, 2>>) -> Tensor<B, 2> {
        assert!(!partials.is_empty(), "LocalAllReduce::sum: called with zero partials");
        partials.into_iter().reduce(|a, b| a + b).unwrap()
    }
}

// ---------------------------------------------------------------------------
// ParallelStrategy
// ---------------------------------------------------------------------------

/// Parallel execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStrategy {
    /// Single GPU (no parallelism)
    Single,
    /// Data parallelism (replicate model on multiple GPUs)
    DataParallel,
    /// Tensor parallelism (Megatron-style column/row-parallel layers)
    TensorParallel,
    /// Pipeline parallelism (split layers across GPUs)
    PipelineParallel,
    /// Expert parallelism (Mixture-of-Experts routing across devices)
    ExpertParallel,
}


// Helpers
// ---------------------------------------------------------------------------

/// Clamp `num_shards` to the largest divisor of `num_kv_heads` that is
/// ≤ the requested value (GQA correctness).
fn clamp_shards(num_shards: usize, num_kv_heads: usize) -> usize {
    let mut n = num_shards.min(num_kv_heads).max(1);
    while n > 1 && !num_kv_heads.is_multiple_of(n) {
        n -= 1;
    }
    if n != num_shards {
        tracing::warn!(
            "clamp_shards: requested {num_shards} shards but num_kv_heads={num_kv_heads} \
             is not evenly divisible; clamped to {n}"
        );
    }
    n
}

/// Build an additive causal bias `[1, 1, seq, seq]`:
/// `bias[i,j] = 0.0` when `j ≤ i`, `-1e9` otherwise.
/// Returns `None` for `seq_len ≤ 1` (no masking needed).
fn build_causal_bias<B: Backend>(seq_len: usize, device: &B::Device) -> Option<Tensor<B, 4>> {
    if seq_len <= 1 {
        return None;
    }
    let mut data = vec![-1e9_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            data[i * seq_len + j] = 0.0;
        }
    }
    Some(
        Tensor::<B, 1>::from_floats(data.as_slice(), device)
            .reshape([1, 1, seq_len, seq_len]),
    )
}

/// Run the sharded MLP (SwiGLU) for one layer, returning the all-reduced
/// `[B·S, hidden]` output.  Used by both TP forward and TP prefill to avoid
/// code duplication.
fn tp_mlp_forward<B: Backend>(
    h2_flat: Tensor<B, 2>,
    gate_w: Tensor<B, 2>,
    up_w: Tensor<B, 2>,
    down_w: Tensor<B, 2>,
    hidden: usize,
    inter: usize,
    num_shards: usize,
    inter_per_shard: usize,
    device: &B::Device,
    flat_len: usize,
) -> Tensor<B, 2> {
    let mut mlp_acc: Option<Tensor<B, 2>> = None;

    for shard in 0..num_shards {
        let start = shard * inter_per_shard;
        let end = ((shard + 1) * inter_per_shard).min(inter);
        if start >= inter { break; }

        let gate = h2_flat.clone().matmul(gate_w.clone().slice([0..hidden, start..end]));
        let up = h2_flat.clone().matmul(up_w.clone().slice([0..hidden, start..end]));

        let silu = gate.clone() * (gate.neg().exp().add_scalar(1.0).recip());
        let activated = silu * up;

        let partial = activated.matmul(down_w.clone().slice([start..end, 0..hidden]));
        mlp_acc = Some(match mlp_acc {
            None => partial,
            Some(acc) => acc + partial,
        });
    }

    mlp_acc.unwrap_or_else(|| Tensor::zeros([flat_len, hidden], device))
}

// ---------------------------------------------------------------------------
// tensor_parallel_llama_forward (prompt / multi-token, NO KV cache)
// ---------------------------------------------------------------------------

/// Megatron-style tensor-parallel forward pass for Llama-family models.
///
/// This function runs a full forward pass over the input sequence without
/// maintaining a KV cache.  For autoregressive generation with a KV cache,
/// use [`tensor_parallel_llama_prefill`] followed by repeated calls to
/// [`tensor_parallel_llama_decode_step`].
///
/// See module-level documentation for the algorithm details.
pub fn tensor_parallel_llama_forward<B: Backend>(
    model: &Llama<B>,
    input_ids: Tensor<B, 2>,
    start_pos: usize,
    num_shards: usize,
) -> Tensor<B, 3> {
    let config = &*model.config;
    let num_shards = clamp_shards(num_shards, config.num_kv_heads);

    let [batch, seq_len] = input_ids.dims();
    let device = input_ids.device();
    let hidden = config.hidden_size;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;

    let q_per_shard = config.num_attention_heads / num_shards;
    let kv_per_shard = config.num_kv_heads / num_shards;
    let inter_per_shard = inter.div_ceil(num_shards);
    let global_n_rep = (config.num_attention_heads / config.num_kv_heads).max(1);

    let causal_bias = build_causal_bias::<B>(seq_len, &device);

    let mut x = model.embed_tokens.forward(input_ids.int());

    for layer in &model.layers {
        let h = layer.input_layernorm.forward(x.clone());
        let h_flat = h.reshape([batch * seq_len, hidden]);

        let q_w = layer.attention.q_proj.weight.val();
        let k_w = layer.attention.k_proj.weight.val();
        let v_w = layer.attention.v_proj.weight.val();
        let o_w = layer.attention.o_proj.weight.val();

        let mut attn_acc: Option<Tensor<B, 2>> = None;

        for shard in 0..num_shards {
            let q_s = shard * q_per_shard * head_dim;
            let q_e = q_s + q_per_shard * head_dim;
            let kv_s = shard * kv_per_shard * head_dim;
            let kv_e = kv_s + kv_per_shard * head_dim;

            let q = h_flat.clone()
                .matmul(q_w.clone().slice([0..hidden, q_s..q_e]))
                .reshape([batch, seq_len, q_per_shard, head_dim])
                .swap_dims(1, 2);

            let k = h_flat.clone()
                .matmul(k_w.clone().slice([0..hidden, kv_s..kv_e]))
                .reshape([batch, seq_len, kv_per_shard, head_dim])
                .swap_dims(1, 2);

            let v = h_flat.clone()
                .matmul(v_w.clone().slice([0..hidden, kv_s..kv_e]))
                .reshape([batch, seq_len, kv_per_shard, head_dim])
                .swap_dims(1, 2);

            let (q, k) = model.rope.apply(q, k, start_pos);
            let k = repeat_kv(k, global_n_rep);
            let v = repeat_kv(v, global_n_rep);

            let scale = (head_dim as f64).sqrt();
            let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
            let scores = match &causal_bias {
                Some(bias) => scores + bias.clone(),
                None => scores,
            };
            let attn = burn::tensor::activation::softmax(scores, 3);

            let ctx = attn.matmul(v)
                .swap_dims(1, 2)
                .reshape([batch * seq_len, q_per_shard * head_dim]);

            let partial = ctx.matmul(o_w.clone().slice([q_s..q_e, 0..hidden]));
            attn_acc = Some(match attn_acc {
                None => partial,
                Some(acc) => acc + partial,
            });
        }

        let attn_out = attn_acc
            .unwrap_or_else(|| Tensor::zeros([batch * seq_len, hidden], &device))
            .reshape([batch, seq_len, hidden])
            + x.clone();

        let h2 = layer.post_attention_layernorm.forward(attn_out.clone());
        let h2_flat = h2.reshape([batch * seq_len, hidden]);

        let mlp_out = tp_mlp_forward(
            h2_flat,
            layer.mlp.gate_proj.weight.val(),
            layer.mlp.up_proj.weight.val(),
            layer.mlp.down_proj.weight.val(),
            hidden, inter, num_shards, inter_per_shard, &device, batch * seq_len,
        ).reshape([batch, seq_len, hidden]);

        x = mlp_out + attn_out;
    }

    let x = model.norm.forward(x);
    model.lm_head.forward(x)
}

// ---------------------------------------------------------------------------
// tensor_parallel_llama_prefill (WITH KV cache capture)
// ---------------------------------------------------------------------------

/// TP prefill: run a full-sequence forward pass and populate a [`TpKvCache`].
///
/// Each shard stores its own `[1, kv_per_shard, seq, head_dim]` K/V pair
/// *before* GQA expansion, keeping memory proportional to `n_kv_heads` rather
/// than `n_heads`.
///
/// Returns `(logits_vec, tp_kv_cache)` where `logits_vec` is the last-position
/// logit vector already on CPU.
pub fn tensor_parallel_llama_prefill<B: Backend>(
    model: &Llama<B>,
    input_ids: Tensor<B, 2>,
    num_shards: usize,
) -> (Vec<f32>, TpKvCache<B>) {
    let config = &*model.config;
    let num_shards = clamp_shards(num_shards, config.num_kv_heads);

    let [batch, seq_len] = input_ids.dims();
    let device = input_ids.device();
    let hidden = config.hidden_size;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;

    let q_per_shard = config.num_attention_heads / num_shards;
    let kv_per_shard = config.num_kv_heads / num_shards;
    let inter_per_shard = inter.div_ceil(num_shards);
    let global_n_rep = (config.num_attention_heads / config.num_kv_heads).max(1);

    let causal_bias = build_causal_bias::<B>(seq_len, &device);

    let mut x = model.embed_tokens.forward(input_ids.int());
    let mut tp_kv_cache: TpKvCache<B> = Vec::with_capacity(model.layers.len());

    for layer in &model.layers {
        let h = layer.input_layernorm.forward(x.clone());
        let h_flat = h.reshape([batch * seq_len, hidden]);

        let q_w = layer.attention.q_proj.weight.val();
        let k_w = layer.attention.k_proj.weight.val();
        let v_w = layer.attention.v_proj.weight.val();
        let o_w = layer.attention.o_proj.weight.val();

        let mut attn_acc: Option<Tensor<B, 2>> = None;
        let mut layer_kv: Vec<KvEntry<B>> = Vec::with_capacity(num_shards);

        for shard in 0..num_shards {
            let q_s = shard * q_per_shard * head_dim;
            let q_e = q_s + q_per_shard * head_dim;
            let kv_s = shard * kv_per_shard * head_dim;
            let kv_e = kv_s + kv_per_shard * head_dim;

            let q = h_flat.clone()
                .matmul(q_w.clone().slice([0..hidden, q_s..q_e]))
                .reshape([batch, seq_len, q_per_shard, head_dim])
                .swap_dims(1, 2);

            let k = h_flat.clone()
                .matmul(k_w.clone().slice([0..hidden, kv_s..kv_e]))
                .reshape([batch, seq_len, kv_per_shard, head_dim])
                .swap_dims(1, 2);

            let v = h_flat.clone()
                .matmul(v_w.clone().slice([0..hidden, kv_s..kv_e]))
                .reshape([batch, seq_len, kv_per_shard, head_dim])
                .swap_dims(1, 2);

            let (q, k) = model.rope.apply(q, k, 0);

            // Store pre-GQA K/V for this shard
            layer_kv.push((k.clone(), v.clone()));

            let k = repeat_kv(k, global_n_rep);
            let v = repeat_kv(v, global_n_rep);

            let scale = (head_dim as f64).sqrt();
            let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
            let scores = match &causal_bias {
                Some(bias) => scores + bias.clone(),
                None => scores,
            };
            let attn = burn::tensor::activation::softmax(scores, 3);

            let ctx = attn.matmul(v)
                .swap_dims(1, 2)
                .reshape([batch * seq_len, q_per_shard * head_dim]);

            let partial = ctx.matmul(o_w.clone().slice([q_s..q_e, 0..hidden]));
            attn_acc = Some(match attn_acc {
                None => partial,
                Some(acc) => acc + partial,
            });
        }

        tp_kv_cache.push(layer_kv);

        let attn_out = attn_acc
            .unwrap_or_else(|| Tensor::zeros([batch * seq_len, hidden], &device))
            .reshape([batch, seq_len, hidden])
            + x.clone();

        let h2 = layer.post_attention_layernorm.forward(attn_out.clone());
        let h2_flat = h2.reshape([batch * seq_len, hidden]);

        let mlp_out = tp_mlp_forward(
            h2_flat,
            layer.mlp.gate_proj.weight.val(),
            layer.mlp.up_proj.weight.val(),
            layer.mlp.down_proj.weight.val(),
            hidden, inter, num_shards, inter_per_shard, &device, batch * seq_len,
        ).reshape([batch, seq_len, hidden]);

        x = mlp_out + attn_out;
    }

    let x = model.norm.forward(x);
    let logits = model.lm_head.forward(x);
    let [_, s, vocab] = logits.dims();
    let last = logits.slice([0..1, s - 1..s, 0..vocab]).reshape([vocab]);
    let logits_vec: Vec<f32> = last.into_data().to_vec().unwrap_or_default();

    (logits_vec, tp_kv_cache)
}

// ---------------------------------------------------------------------------
// tensor_parallel_llama_decode_step (single token, uses TpKvCache)
// ---------------------------------------------------------------------------

/// TP decode step: process a single new token using the distributed KV cache.
///
/// Each shard extends its own `[1, kv_per_shard, seq+1, head_dim]` K/V pair,
/// then the attention outputs and MLP outputs are all-reduced across shards.
///
/// Returns the logit vector for the new position (already on CPU).
pub fn tensor_parallel_llama_decode_step<B: Backend>(
    model: &Llama<B>,
    token_id: u32,
    start_pos: usize,
    num_shards: usize,
    kv_cache: &mut TpKvCache<B>,
) -> Vec<f32> {
    let config = &*model.config;
    let num_shards = clamp_shards(num_shards, config.num_kv_heads);

    let device = &*model.device;
    let hidden = config.hidden_size;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;

    let q_per_shard = config.num_attention_heads / num_shards;
    let kv_per_shard = config.num_kv_heads / num_shards;
    let inter_per_shard = inter.div_ceil(num_shards);
    let global_n_rep = (config.num_attention_heads / config.num_kv_heads).max(1);

    let input = Tensor::<B, 1>::from_floats([token_id as f32], device)
        .reshape([1, 1]);
    let mut x = model.embed_tokens.forward(input.int()); // [1, 1, hidden]

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let h = layer.input_layernorm.forward(x.clone());
        let h_flat = h.reshape([1, hidden]); // [1, hidden] — single token

        let q_w = layer.attention.q_proj.weight.val();
        let k_w = layer.attention.k_proj.weight.val();
        let v_w = layer.attention.v_proj.weight.val();
        let o_w = layer.attention.o_proj.weight.val();

        let mut attn_acc: Option<Tensor<B, 2>> = None;

        for shard in 0..num_shards {
            let q_s = shard * q_per_shard * head_dim;
            let q_e = q_s + q_per_shard * head_dim;
            let kv_s = shard * kv_per_shard * head_dim;
            let kv_e = kv_s + kv_per_shard * head_dim;

            // Column-parallel Q/K/V for single token
            let q = h_flat.clone()
                .matmul(q_w.clone().slice([0..hidden, q_s..q_e]))
                .reshape([1, 1, q_per_shard, head_dim])
                .swap_dims(1, 2); // [1, q_per_shard, 1, head_dim]

            let new_k = h_flat.clone()
                .matmul(k_w.clone().slice([0..hidden, kv_s..kv_e]))
                .reshape([1, 1, kv_per_shard, head_dim])
                .swap_dims(1, 2);

            let new_v = h_flat.clone()
                .matmul(v_w.clone().slice([0..hidden, kv_s..kv_e]))
                .reshape([1, 1, kv_per_shard, head_dim])
                .swap_dims(1, 2);

            // Apply RoPE at the correct absolute position
            let (q, new_k) = model.rope.apply(q, new_k, start_pos);

            // Extend this shard's KV cache
            let kv = &mut kv_cache[layer_idx][shard];
            let k = Tensor::cat(vec![kv.0.clone(), new_k], 2);
            let v = Tensor::cat(vec![kv.1.clone(), new_v], 2);
            *kv = (k.clone(), v.clone());

            // GQA expand
            let k = repeat_kv(k, global_n_rep);
            let v = repeat_kv(v, global_n_rep);

            // Attention — single query, no causal mask needed
            let scale = (head_dim as f64).sqrt();
            let attn = burn::tensor::activation::softmax(
                q.matmul(k.swap_dims(2, 3)).div_scalar(scale), 3,
            );
            let ctx = attn.matmul(v)
                .swap_dims(1, 2)
                .reshape([1, q_per_shard * head_dim]); // [1, q_shard_dim]

            // Row-parallel O
            let partial = ctx.matmul(o_w.clone().slice([q_s..q_e, 0..hidden]));
            attn_acc = Some(match attn_acc {
                None => partial,
                Some(acc) => acc + partial,
            });
        }

        let attn_out = attn_acc
            .unwrap_or_else(|| Tensor::zeros([1, hidden], device))
            .reshape([1, 1, hidden])
            + x.clone();

        // MLP
        let h2 = layer.post_attention_layernorm.forward(attn_out.clone());
        let h2_flat = h2.reshape([1, hidden]);

        let mlp_out = tp_mlp_forward(
            h2_flat,
            layer.mlp.gate_proj.weight.val(),
            layer.mlp.up_proj.weight.val(),
            layer.mlp.down_proj.weight.val(),
            hidden, inter, num_shards, inter_per_shard, device, 1,
        ).reshape([1, 1, hidden]);

        x = mlp_out + attn_out;
    }

    let x = model.norm.forward(x);
    let logits = model.lm_head.forward(x);
    logits.reshape([config.vocab_size]).into_data().to_vec().unwrap_or_default()
}

// ---------------------------------------------------------------------------
// pipeline_parallel_llama_forward
// ---------------------------------------------------------------------------

/// Pipeline-parallel forward pass for Llama-family models.
///
/// Partitions the transformer layers into `num_stages` sequential stages.
/// Each stage processes its subset of layers, then passes the activation
/// tensor to the next stage.
///
/// On a real multi-GPU system, each stage would run on a separate device and
/// the activation tensor would be transferred via PCIe/NVLink between stages.
/// The current implementation runs all stages on the same device.
///
/// # Micro-batching (future)
///
/// Pipeline parallelism achieves high utilization through micro-batching:
/// while stage *i* processes micro-batch *k*, stage *i-1* processes
/// micro-batch *k+1*.  This overlap is not yet implemented; all stages
/// run sequentially on the full batch.
pub fn pipeline_parallel_llama_forward<B: Backend>(
    model: &Llama<B>,
    input_ids: Tensor<B, 2>,
    num_stages: usize,
) -> Tensor<B, 3> {
    let num_layers = model.layers.len();
    let num_stages = num_stages.max(1).min(num_layers);
    let layers_per_stage = num_layers.div_ceil(num_stages);

    let [_, seq_len] = input_ids.dims();
    let causal_bias = crate::models::common::build_causal_bias::<B>(seq_len, &model.device);
    let mut x = model.embed_tokens.forward(input_ids.int());

    for chunk in model.layers.chunks(layers_per_stage) {
        for layer in chunk {
            x = layer.forward(x, &model.rope, 0, causal_bias.as_ref());
        }
        // In multi-GPU: transfer activation to next stage's device here.
        // Currently a no-op since all stages share the same device.
    }

    let x = model.norm.forward(x);
    model.lm_head.forward(x)
}


