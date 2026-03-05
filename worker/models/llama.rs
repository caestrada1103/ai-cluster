//! Llama model implementation
//!
//! Implements Meta's Llama architecture with grouped-query attention,
//! RMS normalization, and SwiGLU activation.

use burn::{
    module::{Module, Ignored},
    nn::{Linear, LinearConfig, Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor},
};
use super::{Model, ModelConfig, ModelOutput, ModelInput, TokenStream, TextGeneration};
use super::common::{RMSNorm, RotaryEmbedding, swiglu, repeat_kv};
use crate::error::WorkerError;
use tracing;
use tokenizers::Tokenizer;
use async_stream::stream;
use futures::Stream;
use std::pin::Pin;
use std::path::Path;

/// Per-layer KV cache entry: (keys, values) shaped [1, n_kv_heads, seq_so_far, head_dim]
pub type KvEntry<B> = (Tensor<B, 4>, Tensor<B, 4>);
/// Full model KV cache — one entry per transformer layer
pub type KvCache<B> = Vec<KvEntry<B>>;

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Llama attention mechanism with Grouped-Query Attention (GQA).
#[derive(Module, Debug)]
pub struct LlamaAttention<B: Backend> {
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

impl<B: Backend> LlamaAttention<B> {
    /// Create new Llama attention layer
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

    /// Forward pass.
    ///
    /// `causal_bias` is an additive `[1, 1, seq, seq]` mask built once at the
    /// `Llama` level and reused across all layers — avoids an O(seq²) allocation
    /// per layer. Pass `None` for single-token inputs (decode step).
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        causal_bias: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
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

        // Apply RoPE
        let (q, k) = rope.apply(q, k, start_pos);

        // Repeat KV heads for GQA
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_rep);
        let v = repeat_kv(v, n_rep);

        // Scaled dot-product attention; apply the pre-built additive causal bias if provided
        let scale = (self.head_dim as f64).sqrt();
        let attn_scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
        let attn_scores = match causal_bias {
            Some(bias) => attn_scores + bias.clone(),
            None => attn_scores,
        };

        let attn = burn::tensor::activation::softmax(attn_scores, 3);
        let output = attn.matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(output)
    }

    /// Decode-step forward: processes a single new token using the KV cache.
    ///
    /// Appends the new K/V to `kv` and attends over the full cached sequence.
    /// No causal mask is needed — the single query token can attend to all
    /// cached positions by construction.
    pub fn forward_decode(
        &self,
        hidden: Tensor<B, 3>,          // [1, 1, hidden]
        rope: &RotaryEmbedding<B>,
        start_pos: usize,              // absolute position of the new token in the sequence
        kv: &mut KvEntry<B>,
    ) -> Tensor<B, 3> {
        let q = self.q_proj.forward(hidden.clone())
            .reshape([1, 1, self.num_heads, self.head_dim])
            .swap_dims(1, 2);                           // [1, n_heads, 1, head_dim]
        let new_k = self.k_proj.forward(hidden.clone())
            .reshape([1, 1, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let new_v = self.v_proj.forward(hidden)
            .reshape([1, 1, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE at the correct absolute position
        let (q, new_k) = rope.apply(q, new_k, start_pos);

        // Extend cache along the sequence dimension
        let k = Tensor::cat(vec![kv.0.clone(), new_k], 2); // [1, n_kv_heads, seq+1, head_dim]
        let v = Tensor::cat(vec![kv.1.clone(), new_v], 2);
        *kv = (k.clone(), v.clone());

        // GQA expand then attend (single query — no causal mask needed)
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

/// Llama MLP with SwiGLU activation
#[derive(Module, Debug)]
pub struct LlamaMLP<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> LlamaMLP<B> {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        device: &B::Device,
    ) -> Self {
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
// Transformer Layer
// ---------------------------------------------------------------------------

/// Llama transformer layer
#[derive(Module, Debug)]
pub struct LlamaLayer<B: Backend> {
    pub attention: LlamaAttention<B>,
    pub mlp: LlamaMLP<B>,
    pub input_layernorm: RMSNorm<B>,
    pub post_attention_layernorm: RMSNorm<B>,
}

impl<B: Backend> LlamaLayer<B> {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f32,
        device: &B::Device,
    ) -> Self {
        Self {
            attention: LlamaAttention::new(hidden_size, num_heads, num_kv_heads, head_dim, device),
            mlp: LlamaMLP::new(hidden_size, intermediate_size, device),
            input_layernorm: RMSNorm::new(hidden_size, rms_norm_eps as f64, device),
            post_attention_layernorm: RMSNorm::new(hidden_size, rms_norm_eps as f64, device),
        }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
        causal_bias: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let residual = input.clone();
        let x = self.input_layernorm.forward(input);
        let x = self.attention.forward(x, rope, start_pos, causal_bias);
        let x = x + residual;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        x + residual
    }

    /// Decode-step forward: processes a single token using the KV cache.
    pub fn forward_decode(
        &self,
        x: Tensor<B, 3>,               // [1, 1, hidden]
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
// Full Model
// ---------------------------------------------------------------------------

/// Complete Llama model
#[derive(Module, Debug)]
pub struct Llama<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<LlamaLayer<B>>,
    pub norm: RMSNorm<B>,
    pub lm_head: Linear<B>,
    pub config: Ignored<LlamaConfig>,
    pub rope: RotaryEmbedding<B>,
    
    #[module(ignore)]
    pub tokenizer: Ignored<Tokenizer>,
    pub device: Ignored<B::Device>,
}

/// Llama configuration
#[derive(Debug, Clone)]
pub struct LlamaConfig {
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

impl LlamaConfig {
    /// Create configuration for Llama 3 8B
    pub fn llama3_8b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 14336,
            vocab_size: 128256,
            max_seq_len: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
        }
    }

    /// Create configuration for Llama 3 70B
    pub fn llama3_70b() -> Self {
        Self {
            hidden_size: 8192,
            num_layers: 80,
            num_attention_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 28672,
            vocab_size: 128256,
            max_seq_len: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
        }
    }

    /// Create configuration for Llama 2 7B
    pub fn llama2_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            vocab_size: 32000,
            max_seq_len: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }

    /// Convert to generic ModelConfig
    pub fn to_model_config(&self) -> ModelConfig {
        ModelConfig {
            architecture: "llama".to_string(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
            intermediate_size: self.intermediate_size,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            is_moe: false,
            num_experts: None,
            num_experts_per_tok: None,
        }
    }
}

impl<B: Backend> Llama<B> {
    /// Create a new Llama model
    pub fn new(config: &LlamaConfig, device: &B::Device, tokenizer_path: &Path) -> Result<Self, WorkerError> {
        let layers = (0..config.num_layers)
            .map(|_| LlamaLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_kv_heads,
                config.head_dim,
                config.intermediate_size,
                config.rms_norm_eps,
                device,
            ))
            .collect();

        let rope = RotaryEmbedding::new(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            device,
        );

        // Load tokenizer from the model directory
        let tok_file = tokenizer_path.join("tokenizer.json");
        eprintln!("[INFO] Loading tokenizer from: {:?}", tok_file);
        let tokenizer = Tokenizer::from_file(&tok_file)
            .map_err(|e| {
                eprintln!("[WARN] Failed to load tokenizer from {:?}: {}. Trying HF pretrained...", tok_file, e);
                e
            })
            .or_else(|_| {
                Tokenizer::from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", None)
            })
            .map_err(|e| WorkerError::ModelLoad(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .init(device),
            layers,
            norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps as f64, device),
            lm_head: LinearConfig::new(config.hidden_size, config.vocab_size)
                .with_bias(false)
                .init(device),
            config: Ignored(config.clone()),
            rope,
            tokenizer: Ignored(tokenizer),
            device: Ignored(device.clone()),
        })
    }

    /// Forward pass through the model.
    ///
    /// Builds the causal bias once and threads it through all layers,
    /// avoiding an O(seq²) allocation per layer.
    pub fn forward_pass(
        &self,
        input_ids: Tensor<B, 2>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let [_, seq_len] = input_ids.dims();
        let causal_bias = super::common::build_causal_bias::<B>(seq_len, &*self.device);
        let mut x = self.embed_tokens.forward(input_ids.int());

        for layer in &self.layers {
            x = layer.forward(x, &self.rope, start_pos, causal_bias.as_ref());
        }

        let x = self.norm.forward(x);
        self.lm_head.forward(x)
    }

    /// Estimate memory usage in bytes (FP16)
    pub fn memory_usage(&self) -> usize {
        let c = &self.config;
        let embed = c.vocab_size * c.hidden_size;
        let attn = c.num_layers * 4 * c.hidden_size * c.hidden_size;
        let ffn = c.num_layers * 3 * c.hidden_size * c.intermediate_size;
        let norm = (c.num_layers * 2 + 1) * c.hidden_size;
        (embed + attn + ffn) * 2 + norm * 2
    }

    /// Prefill: run the full forward pass on the prompt, return last-position logits
    /// (as `Vec<f32>`, already on CPU) and the populated KV cache.
    ///
    /// The causal bias is computed once at this level and reused across all layers,
    /// eliminating the per-layer O(seq²) allocation that existed in the old approach.
    pub fn prefill(&self, input_ids: Tensor<B, 2>) -> (Vec<f32>, KvCache<B>) {
        let [_, seq_len] = input_ids.dims();
        let device = input_ids.device();
        let config   = &*self.config;
        let n_heads  = config.num_attention_heads;
        let n_kv     = config.num_kv_heads;
        let head_dim = config.head_dim;
        let n_rep    = (n_heads / n_kv).max(1);

        // Build additive causal bias once — [1, 1, seq, seq]
        // bias[i,j] = 0.0 when j ≤ i (allowed), -1e9 when j > i (masked)
        let causal_bias: Option<Tensor<B, 4>> = if seq_len > 1 {
            let mut data = vec![-1e9_f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..=i {
                    data[i * seq_len + j] = 0.0;
                }
            }
            Some(
                Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                    .reshape([1, 1, seq_len, seq_len]),
            )
        } else {
            None
        };

        let mut x = self.embed_tokens.forward(input_ids.int());
        let mut kv_cache: KvCache<B> = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let h = layer.input_layernorm.forward(x.clone());
            let [b, s, _] = h.dims();

            // Project Q, K, V
            let q = layer.attention.q_proj.forward(h.clone())
                .reshape([b, s, n_heads, head_dim])
                .swap_dims(1, 2);                       // [b, n_heads, s, head_dim]
            let k = layer.attention.k_proj.forward(h.clone())
                .reshape([b, s, n_kv, head_dim])
                .swap_dims(1, 2);
            let v = layer.attention.v_proj.forward(h)
                .reshape([b, s, n_kv, head_dim])
                .swap_dims(1, 2);

            // Apply RoPE starting at position 0
            let (q, k) = self.rope.apply(q, k, 0);

            // Store compact (pre-GQA) K, V in the cache
            kv_cache.push((k.clone(), v.clone()));

            // GQA expand for multi-head attention
            let k_full = repeat_kv(k, n_rep);
            let v_full = repeat_kv(v, n_rep);

            // Scaled dot-product attention with causal bias
            let scale = (head_dim as f64).sqrt();
            let scores = q.matmul(k_full.swap_dims(2, 3)).div_scalar(scale);
            let scores = match &causal_bias {
                Some(bias) => scores + bias.clone(),
                None => scores,
            };
            let attn = burn::tensor::activation::softmax(scores, 3);
            let ctx = attn.matmul(v_full)
                .swap_dims(1, 2)
                .reshape([b, s, n_heads * head_dim]);
            let attn_out = layer.attention.o_proj.forward(ctx);

            // Residual + MLP
            let x_attn = attn_out + x.clone();
            let h2 = layer.post_attention_layernorm.forward(x_attn.clone());
            x = layer.mlp.forward(h2) + x_attn;
        }

        // Final norm + lm_head
        let x = self.norm.forward(x);
        let logits = self.lm_head.forward(x);           // [1, seq, vocab]
        let [_, s, vocab] = logits.dims();
        let last = logits.slice([0..1, s - 1..s, 0..vocab]).reshape([vocab]);
        let logits_vec: Vec<f32> = last.into_data().to_vec().unwrap_or_else(|e| {
            tracing::error!("prefill: failed to pull logits from GPU: {e:?}");
            vec![0.0; vocab]
        });

        (logits_vec, kv_cache)
    }

    /// Decode-step: process a single new token using the KV cache.
    ///
    /// Returns the logit vector for the new position (already pulled to CPU).
    /// Complexity is O(seq_cached) — linear in accumulated sequence length.
    pub fn decode_step(
        &self,
        token_id: u32,
        start_pos: usize,
        kv_cache: &mut KvCache<B>,
    ) -> Vec<f32> {
        let device = &*self.device;
        let vocab = self.config.vocab_size;

        let input = Tensor::<B, 1>::from_floats([token_id as f32], device)
            .reshape([1, 1]);                           // [1, 1]
        let mut x = self.embed_tokens.forward(input.int()); // [1, 1, hidden]

        for (layer, kv) in self.layers.iter().zip(kv_cache.iter_mut()) {
            x = layer.forward_decode(x, &self.rope, start_pos, kv);
        }

        let x = self.norm.forward(x);                  // [1, 1, hidden]
        let logits = self.lm_head.forward(x);          // [1, 1, vocab]
        logits.reshape([vocab]).into_data().to_vec().unwrap_or_else(|e| {
            tracing::error!("decode_step: failed to pull logits from GPU: {e:?}");
            vec![0.0; vocab]
        })
    }

    /// Tokenize a prompt, correctly handling known special tokens.
    ///
    /// Special tokens like `</s>`, `<s>`, `<|user|>`, `<|assistant|>` are
    /// looked up in the vocabulary as single tokens rather than being split
    /// into sub-word pieces by the BPE tokenizer.
    pub fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<u32>, WorkerError> {
        let special_tokens = ["</s>", "<s>", "<|user|>", "<|assistant|>"];
        let mut tokens = Vec::new();
        let mut current_pos = 0;

        while current_pos < prompt.len() {
            let mut best_match = None;
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
                    .map_err(|e| WorkerError::Internal(format!("Tokenizer error: {}", e)))?;
                tokens.extend_from_slice(encoding.get_ids());
            }

            if let Some(st) = best_match {
                if let Some(id) = self.tokenizer.token_to_id(st) {
                    tokens.push(id);
                } else {
                    let encoding = self.tokenizer.encode(*st, false)
                        .map_err(|e| WorkerError::Internal(format!("Tokenizer error (special): {}", e)))?;
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

impl<B: Backend> Model<B> for Llama<B> {
    fn name(&self) -> &str {
        "llama"
    }

    fn config(&self) -> ModelConfig {
        self.config.to_model_config()
    }

    fn forward(&self, input: ModelInput<B>) -> Result<ModelOutput<B>, WorkerError> {
        Ok(self.forward_pass(input, 0))
    }

    fn generate(
        &self,
        _prompt: &str,
        max_tokens: usize,
        _temperature: f32,
        _top_p: f32,
        _top_k: usize,
    ) -> Result<TokenStream, WorkerError> {
        Ok(TokenStream::new(max_tokens))
    }

    fn memory_used(&self) -> usize {
        self.memory_usage()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}
impl<B: Backend> TextGeneration for Llama<B> {
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, WorkerError>> + Send>>, WorkerError> {
        let tokens = self.tokenize_prompt(prompt)?;
        let prompt_len = tokens.len();

        // Single model clone — moved outside the loop so it is NOT repeated per token.
        let model = self.clone();

        // Channel to stream results from the blocking inference thread to async consumers.
        let (tx, mut rx) =
            tokio::sync::mpsc::channel::<Result<String, WorkerError>>(max_tokens + 2);

        // Run the entire prefill + decode loop in ONE blocking thread.
        // This eliminates per-token spawn_blocking overhead and per-token Tokio context switches.
        tokio::task::spawn_blocking(move || {
            // ── PREFILL ──────────────────────────────────────────────────────────────
            let input_f32: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
            let device: &<B as burn::tensor::backend::Backend>::Device = &*model.device;
            let input = Tensor::<B, 1>::from_floats(input_f32.as_slice(), device)
                .reshape([1, prompt_len]);
            let (logits_vec, mut kv_cache) = model.prefill(input);

            // Inline helper: sample a token from a logit vector
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

            // Inline helper: compute delta text from newly generated token ids
            let delta_text = |tok_ids: &[u32], prev_len: &mut usize, tokenizer: &Tokenizer| -> Result<String, WorkerError> {
                let text = tokenizer
                    .decode(tok_ids, true)
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

            // Look up EOS id via O(1) token_to_id (avoids rebuilding the full vocab HashMap)
            let eos_id: Option<u32> = model.tokenizer.token_to_id("</s>");

            // ── FIRST GENERATED TOKEN (from prefill logits) ───────────────────────
            let first_tok = sample(&logits_vec);
            let mut all_tokens = tokens;
            all_tokens.push(first_tok);
            let mut prev_text_len = 0usize;
            let _ = tx.blocking_send(delta_text(
                &all_tokens[prompt_len..], &mut prev_text_len, &model.tokenizer,
            ));
            if eos_id == Some(first_tok) { return; }

            // ── DECODE LOOP (one GPU dispatch per step, no Tokio context switch) ──
            for _step in 1..max_tokens {
                let cur_tok  = *all_tokens.last().unwrap();
                let start    = all_tokens.len() - 1; // absolute position of cur_tok
                let logits   = model.decode_step(cur_tok, start, &mut kv_cache);
                let next_tok = sample(&logits);
                all_tokens.push(next_tok);
                let _ = tx.blocking_send(delta_text(
                    &all_tokens[prompt_len..], &mut prev_text_len, &model.tokenizer,
                ));
                if eos_id == Some(next_tok) { break; }
            }
        });

        // Bridge the mpsc receiver into the async stream expected by the gRPC layer.
        let stream = stream! {
            while let Some(item) = rx.recv().await {
                yield item;
            }
        };

        Ok(Box::pin(stream))
    }
}
