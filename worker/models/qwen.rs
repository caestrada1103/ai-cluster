//! Qwen3-Coder model implementation
//!
//! Qwen3-Coder is architecturally identical to Llama3 (GQA + RoPE + SwiGLU),
//! reusing all common.rs primitives.  The only differences are chat-format
//! special tokens and the 32B-specific hyperparameters.

#![allow(dead_code)]

use burn::{
    module::{Module, Ignored},
    nn::{Linear, LinearConfig, Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor},
};
use super::TextGeneration;
use super::common::{RMSNorm, RotaryEmbedding, swiglu, repeat_kv};
use super::llama::{KvEntry, KvCache};
use crate::error::WorkerError;
use tokenizers::Tokenizer;
use async_stream::stream;
use std::path::Path;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Qwen3 model configuration
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl QwenConfig {
    /// Default configuration for Qwen3-Coder-32B (Qwen/Qwen3-Coder-32B).
    pub fn qwen3_coder_32b() -> Self {
        Self {
            hidden_size: 5120,
            num_layers: 64,
            num_attention_heads: 40,
            num_kv_heads: 8,
            head_dim: 128, // 5120 / 40
            intermediate_size: 27648,
            vocab_size: 151936,
            max_seq_len: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Qwen attention with Grouped-Query Attention (GQA) — identical to LlamaAttention.
#[derive(Module, Debug)]
pub struct QwenAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    num_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl<B: Backend> QwenAttention<B> {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let q_out = num_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;
        Self {
            q_proj: LinearConfig::new(hidden_size, q_out).with_bias(false).init(device),
            k_proj: LinearConfig::new(hidden_size, kv_out).with_bias(false).init(device),
            v_proj: LinearConfig::new(hidden_size, kv_out).with_bias(false).init(device),
            o_proj: LinearConfig::new(q_out, hidden_size).with_bias(false).init(device),
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// Full-sequence prefill forward.  Returns output and a pre-GQA KV entry
    /// so the caller can populate the KV cache without a second projection pass.
    pub fn forward_prefill(
        &self,
        hidden: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        causal_bias: Option<&Tensor<B, 4>>,
    ) -> (Tensor<B, 3>, KvEntry<B>) {
        let [batch, seq_len, _] = hidden.dims();

        let q = self.q_proj.forward(hidden.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self.k_proj.forward(hidden.clone())
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self.v_proj.forward(hidden)
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = rope.apply(q, k, start_pos);
        let kv_entry = (k.clone(), v.clone());

        let n_rep = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k, n_rep);
        let v_full = repeat_kv(v, n_rep);

        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(k_full.swap_dims(2, 3)).div_scalar(scale);
        let scores = match causal_bias {
            Some(bias) => scores + bias.clone(),
            None => scores,
        };

        let attn = burn::tensor::activation::softmax(scores, 3);
        let output = attn.matmul(v_full)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        (self.o_proj.forward(output), kv_entry)
    }

    /// Single-token decode using the KV cache.
    pub fn forward_decode(
        &self,
        hidden: Tensor<B, 3>,          // [1, 1, hidden]
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        kv: &mut KvEntry<B>,
    ) -> Tensor<B, 3> {
        let q = self.q_proj.forward(hidden.clone())
            .reshape([1, 1, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let new_k = self.k_proj.forward(hidden.clone())
            .reshape([1, 1, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let new_v = self.v_proj.forward(hidden)
            .reshape([1, 1, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, new_k) = rope.apply(q, new_k, start_pos);

        let k = Tensor::cat(vec![kv.0.clone(), new_k], 2);
        let v = Tensor::cat(vec![kv.1.clone(), new_v], 2);
        *kv = (k.clone(), v.clone());

        let n_rep = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k, n_rep);
        let v_full = repeat_kv(v, n_rep);

        let scale = (self.head_dim as f64).sqrt();
        let attn = burn::tensor::activation::softmax(
            q.matmul(k_full.swap_dims(2, 3)).div_scalar(scale),
            3,
        );
        self.o_proj.forward(
            attn.matmul(v_full)
                .swap_dims(1, 2)
                .reshape([1, 1, self.num_heads * self.head_dim]),
        )
    }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// Qwen MLP with SwiGLU activation (identical to LlamaMLP).
#[derive(Module, Debug)]
pub struct QwenMLP<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> QwenMLP<B> {
    pub fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(hidden_size, intermediate_size).with_bias(false).init(device),
            up_proj: LinearConfig::new(hidden_size, intermediate_size).with_bias(false).init(device),
            down_proj: LinearConfig::new(intermediate_size, hidden_size).with_bias(false).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.forward(input.clone());
        let up = self.up_proj.forward(input);
        self.down_proj.forward(swiglu(gate, up))
    }
}

// ---------------------------------------------------------------------------
// Transformer layer
// ---------------------------------------------------------------------------

/// Qwen transformer layer (pre-norm, residual).
#[derive(Module, Debug)]
pub struct QwenLayer<B: Backend> {
    pub attention: QwenAttention<B>,
    pub mlp: QwenMLP<B>,
    pub input_layernorm: RMSNorm<B>,
    pub post_attention_layernorm: RMSNorm<B>,
}

impl<B: Backend> QwenLayer<B> {
    pub fn new(config: &QwenConfig, device: &B::Device) -> Self {
        Self {
            attention: QwenAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_kv_heads,
                config.head_dim,
                device,
            ),
            mlp: QwenMLP::new(config.hidden_size, config.intermediate_size, device),
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps as f64, device),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps as f64, device),
        }
    }

    /// Prefill forward: returns hidden state output and the layer's KV entry.
    pub fn forward_prefill(
        &self,
        input: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        causal_bias: Option<&Tensor<B, 4>>,
    ) -> (Tensor<B, 3>, KvEntry<B>) {
        let residual = input.clone();
        let x = self.input_layernorm.forward(input);
        let (attn_out, kv) = self.attention.forward_prefill(x, rope, start_pos, causal_bias);
        let x = attn_out + residual;

        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(h);
        (x + residual, kv)
    }

    /// Decode-step forward: single token, extends KV cache in-place.
    pub fn forward_decode(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        kv: &mut KvEntry<B>,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let h = self.attention.forward_decode(h, rope, start_pos, kv);
        let h = h + residual;

        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(h);
        let h = self.mlp.forward(h);
        h + residual
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// Complete Qwen3-Coder model.
#[derive(Module, Debug)]
pub struct Qwen<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<QwenLayer<B>>,
    pub norm: RMSNorm<B>,
    pub lm_head: Linear<B>,
    pub config: Ignored<QwenConfig>,
    pub rope: RotaryEmbedding<B>,
    #[module(ignore)]
    pub tokenizer: Ignored<Tokenizer>,
    pub device: Ignored<B::Device>,
}

impl<B: Backend> Qwen<B> {
    pub fn new(config: &QwenConfig, device: &B::Device, tokenizer_path: &Path) -> Result<Self, WorkerError> {
        let layers = (0..config.num_layers)
            .map(|_| QwenLayer::new(config, device))
            .collect();

        let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_theta, device);

        let tok_file = tokenizer_path.join("tokenizer.json");
        eprintln!("[INFO] Loading Qwen tokenizer from: {:?}", tok_file);
        let tokenizer = Tokenizer::from_file(&tok_file)
            .map_err(|e| {
                eprintln!("[WARN] Failed to load tokenizer from {:?}: {}. Trying HF pretrained...", tok_file, e);
                e
            })
            .or_else(|_| Tokenizer::from_pretrained("Qwen/Qwen2.5-0.5B", None))
            .map_err(|e| WorkerError::ModelLoad(format!("Failed to load Qwen tokenizer: {}", e)))?;

        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps as f64, device),
            lm_head: LinearConfig::new(config.hidden_size, config.vocab_size).with_bias(false).init(device),
            config: Ignored(config.clone()),
            rope,
            tokenizer: Ignored(tokenizer),
            device: Ignored(device.clone()),
        })
    }

    /// Full-sequence prefill: returns last-position logits (CPU) and populated KV cache.
    pub fn prefill(&self, input_ids: Tensor<B, 2>) -> (Vec<f32>, KvCache<B>) {
        let [_, seq_len] = input_ids.dims();
        let device = input_ids.device();
        let config = &*self.config;

        let causal_bias: Option<Tensor<B, 4>> = if seq_len > 1 {
            let mut data = vec![-1e9_f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..=i {
                    data[i * seq_len + j] = 0.0;
                }
            }
            Some(Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([1, 1, seq_len, seq_len]))
        } else {
            None
        };

        let mut x = self.embed_tokens.forward(input_ids.int());
        let mut kv_cache: KvCache<B> = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (out, kv) = layer.forward_prefill(x, &self.rope, 0, causal_bias.as_ref());
            kv_cache.push(kv);
            x = out;
        }

        let x = self.norm.forward(x);
        let logits = self.lm_head.forward(x);
        let [_, s, vocab] = logits.dims();
        let last = logits.slice([0..1, s - 1..s, 0..vocab]).reshape([vocab]);
        let logits_vec: Vec<f32> = last.into_data().to_vec().unwrap_or_else(|e| {
            tracing::error!("qwen prefill: failed to pull logits from GPU: {e:?}");
            vec![0.0; config.vocab_size]
        });

        (logits_vec, kv_cache)
    }

    /// Single-token decode step using the KV cache.
    pub fn decode_step(&self, token_id: u32, start_pos: usize, kv_cache: &mut KvCache<B>) -> Vec<f32> {
        let device = &*self.device;
        let vocab = self.config.vocab_size;

        let input = Tensor::<B, 1>::from_floats([token_id as f32], device).reshape([1, 1]);
        let mut x = self.embed_tokens.forward(input.int());

        for (layer, kv) in self.layers.iter().zip(kv_cache.iter_mut()) {
            x = layer.forward_decode(x, &self.rope, start_pos, kv);
        }

        let x = self.norm.forward(x);
        let logits = self.lm_head.forward(x);
        logits.reshape([vocab]).into_data().to_vec().unwrap_or_else(|e| {
            tracing::error!("qwen decode_step: failed to pull logits from GPU: {e:?}");
            vec![0.0; vocab]
        })
    }

    /// Tokenize a prompt, handling Qwen chat-format special tokens.
    pub fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<u32>, WorkerError> {
        let special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"];
        let mut tokens = Vec::new();
        let mut current_pos = 0;

        while current_pos < prompt.len() {
            let mut best_match: Option<&str> = None;
            let mut min_idx = prompt.len();

            for st in &special_tokens {
                if let Some(idx) = prompt[current_pos..].find(st) {
                    let abs_idx = current_pos + idx;
                    if abs_idx < min_idx {
                        min_idx = abs_idx;
                        best_match = Some(st);
                    }
                }
            }

            if min_idx > current_pos {
                let text_segment = &prompt[current_pos..min_idx];
                let add_special = current_pos == 0;
                let encoding = self.tokenizer.encode(text_segment, add_special)
                    .map_err(|e| WorkerError::Internal(format!("Qwen tokenizer error: {}", e)))?;
                tokens.extend_from_slice(encoding.get_ids());
            }

            if let Some(st) = best_match {
                if let Some(id) = self.tokenizer.token_to_id(st) {
                    tokens.push(id);
                } else {
                    let encoding = self.tokenizer.encode(st, false)
                        .map_err(|e| WorkerError::Internal(format!("Qwen tokenizer error (special): {}", e)))?;
                    tokens.extend_from_slice(encoding.get_ids());
                }
                current_pos = min_idx + st.len();
            } else {
                break;
            }
        }
        Ok(tokens)
    }
}

impl<B: Backend> TextGeneration for Qwen<B> {
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<super::TextStream, WorkerError> {
        let tokens = self.tokenize_prompt(prompt)?;
        let prompt_len = tokens.len();
        let model = self.clone();

        let (tx, mut rx) =
            tokio::sync::mpsc::channel::<Result<String, WorkerError>>(max_tokens + 2);

        tokio::task::spawn_blocking(move || {
            // ── PREFILL ──────────────────────────────────────────────────────
            let input_f32: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
            let device: &<B as burn::tensor::backend::Backend>::Device = &model.device;
            let input = Tensor::<B, 1>::from_floats(input_f32.as_slice(), device)
                .reshape([1, prompt_len]);
            let (logits_vec, mut kv_cache) = model.prefill(input);

            let sample = |logits: &[f32]| -> u32 {
                if temperature < 0.01 {
                    logits.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                } else {
                    super::common::top_k_top_p_sample(logits, temperature, top_p, top_k) as u32
                }
            };

            let delta_text = |tok_ids: &[u32], prev_len: &mut usize, tok: &Tokenizer| -> Result<String, WorkerError> {
                let text = tok.decode(tok_ids, true)
                    .map_err(|e| WorkerError::Internal(format!("Decode error: {}", e)))?;
                let delta = if text.len() > *prev_len {
                    let mut start = *prev_len;
                    while start < text.len() && !text.is_char_boundary(start) {
                        start += 1;
                    }
                    if start < text.len() { text[start..].to_string() } else { String::new() }
                } else {
                    String::new()
                };
                *prev_len = text.len();
                Ok(delta)
            };

            // Prefer <|im_end|> as EOS; fall back to <|endoftext|>
            let eos_id: Option<u32> = model.tokenizer.token_to_id("<|im_end|>")
                .or_else(|| model.tokenizer.token_to_id("<|endoftext|>"));

            // ── FIRST TOKEN (from prefill logits) ────────────────────────────
            let first_tok = sample(&logits_vec);
            let mut all_tokens = tokens;
            all_tokens.push(first_tok);
            let mut prev_text_len = 0usize;
            let _ = tx.blocking_send(delta_text(
                &all_tokens[prompt_len..], &mut prev_text_len, &model.tokenizer,
            ));
            if eos_id == Some(first_tok) { return; }

            // ── DECODE LOOP ──────────────────────────────────────────────────
            for _step in 1..max_tokens {
                let cur_tok  = *all_tokens.last().unwrap();
                let start    = all_tokens.len() - 1;
                let logits   = model.decode_step(cur_tok, start, &mut kv_cache);
                let next_tok = sample(&logits);
                all_tokens.push(next_tok);
                let _ = tx.blocking_send(delta_text(
                    &all_tokens[prompt_len..], &mut prev_text_len, &model.tokenizer,
                ));
                if eos_id == Some(next_tok) { break; }
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
