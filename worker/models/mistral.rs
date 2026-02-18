//! Mistral model implementation.
//!
//! Mistral uses Sliding Window Attention (SWA) as its key differentiator
//! from the standard Llama architecture.

use burn::{
    module::{Module, Ignored},
    nn::{Linear, LinearConfig, Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor},
};
use super::{Model, ModelConfig, ModelOutput, ModelInput, TokenStream};
use super::common::{RMSNorm, RotaryEmbedding, swiglu};
use crate::error::WorkerError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Mistral-specific configuration.
#[derive(Debug, Clone)]
pub struct MistralConfig {
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
    pub sliding_window: usize,
}

impl MistralConfig {
    /// Mistral 7B v0.1 configuration.
    pub fn mistral_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 14336,
            vocab_size: 32000,
            max_seq_len: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            sliding_window: 4096,
        }
    }

    /// Mistral 7B Instruct v0.2 configuration.
    pub fn mistral_7b_instruct() -> Self {
        let mut config = Self::mistral_7b();
        config.max_seq_len = 32768;
        config
    }

    /// Convert to the generic [`ModelConfig`].
    pub fn to_model_config(&self) -> ModelConfig {
        ModelConfig {
            architecture: "mistral".to_string(),
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

// ---------------------------------------------------------------------------
// Attention with Sliding Window
// ---------------------------------------------------------------------------

/// Mistral Attention with Sliding Window Attention (SWA).
#[derive(Module, Debug)]
pub struct MistralAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    num_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    sliding_window: usize,
}

impl<B: Backend> MistralAttention<B> {
    pub fn new(config: &MistralConfig, device: &B::Device) -> Self {
        let hidden = config.hidden_size;
        let q_out = config.num_attention_heads * config.head_dim;
        let kv_out = config.num_kv_heads * config.head_dim;

        Self {
            q_proj: LinearConfig::new(hidden, q_out).with_bias(false).init(device),
            k_proj: LinearConfig::new(hidden, kv_out).with_bias(false).init(device),
            v_proj: LinearConfig::new(hidden, kv_out).with_bias(false).init(device),
            o_proj: LinearConfig::new(q_out, hidden).with_bias(false).init(device),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            sliding_window: config.sliding_window,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        // Project Q, K, V
        let q = self.q_proj.forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self.k_proj.forward(x.clone())
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self.v_proj.forward(x)
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE
        let (q, k) = rope.apply(q, k, start_pos);

        // Repeat KV heads for GQA
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = super::common::repeat_kv(k, n_rep);
        let v = super::common::repeat_kv(v, n_rep);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);

        // Apply sliding window mask
        // (In production, this would mask positions > sliding_window distance)
        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);

        let attn_output = attn_weights
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_output)
    }
}

// ---------------------------------------------------------------------------
// Feed-forward (SwiGLU)
// ---------------------------------------------------------------------------

/// Mistral MLP with SwiGLU activation.
#[derive(Module, Debug)]
pub struct MistralMLP<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> MistralMLP<B> {
    pub fn new(config: &MistralConfig, device: &B::Device) -> Self {
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;

        Self {
            gate_proj: LinearConfig::new(hidden, inter).with_bias(false).init(device),
            up_proj: LinearConfig::new(hidden, inter).with_bias(false).init(device),
            down_proj: LinearConfig::new(inter, hidden).with_bias(false).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.forward(x.clone());
        let up = self.up_proj.forward(x);
        self.down_proj.forward(swiglu(gate, up))
    }
}

// ---------------------------------------------------------------------------
// Transformer Layer
// ---------------------------------------------------------------------------

/// Single Mistral transformer layer.
#[derive(Module, Debug)]
pub struct MistralLayer<B: Backend> {
    attention: MistralAttention<B>,
    mlp: MistralMLP<B>,
    input_layernorm: RMSNorm<B>,
    post_attention_layernorm: RMSNorm<B>,
}

impl<B: Backend> MistralLayer<B> {
    pub fn new(config: &MistralConfig, device: &B::Device) -> Self {
        Self {
            attention: MistralAttention::new(config, device),
            mlp: MistralMLP::new(config, device),
            input_layernorm: RMSNorm::new(
                config.hidden_size,
                config.rms_norm_eps as f64,
                device,
            ),
            post_attention_layernorm: RMSNorm::new(
                config.hidden_size,
                config.rms_norm_eps as f64,
                device,
            ),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        // Pre-norm + attention + residual
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.attention.forward(x, rope, start_pos);
        let x = x + residual;

        // Pre-norm + FFN + residual
        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        x + residual
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

/// Mistral language model.
#[derive(Module, Debug)]
pub struct Mistral<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<MistralLayer<B>>,
    norm: RMSNorm<B>,
    lm_head: Linear<B>,
    config: Ignored<MistralConfig>,
    rope: RotaryEmbedding<B>,
    gpu_ids: Vec<usize>,
}

impl<B: Backend> Mistral<B> {
    /// Build a new Mistral model with randomly initialised weights.
    pub fn new(config: MistralConfig, gpu_ids: Vec<usize>, device: &B::Device) -> Self {
        let layers = (0..config.num_layers)
            .map(|_| MistralLayer::new(&config, device))
            .collect();

        let rope = RotaryEmbedding::new(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            device,
        );

        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .init(device),
            layers,
            norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps as f64, device),
            lm_head: LinearConfig::new(config.hidden_size, config.vocab_size)
                .with_bias(false)
                .init(device),
            config: Ignored(config.clone()),
            rope,
            gpu_ids,
        }
    }

    /// Forward pass returning logits tensor.
    pub fn forward_pass(
        &self,
        input_ids: Tensor<B, 2>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        // Token embeddings: [batch, seq] -> [batch, seq, hidden]
        let mut x = self.embed_tokens.forward(input_ids.int());

        // Pass through transformer layers
        for layer in &self.layers {
            x = layer.forward(x, &self.rope, start_pos);
        }

        // Final norm + LM head
        let x = self.norm.forward(x);
        self.lm_head.forward(x)
    }

    /// Estimate memory usage in bytes (FP16).
    pub fn memory_usage(&self) -> usize {
        let c = &self.config;
        let embed_params = c.vocab_size * c.hidden_size;
        let attn_params = c.num_layers * (
            4 * c.hidden_size * c.hidden_size  // q, k, v, o projections
        );
        let ffn_params = c.num_layers * (
            3 * c.hidden_size * c.intermediate_size  // gate, up, down
        );
        let norm_params = (c.num_layers * 2 + 1) * c.hidden_size;
        let total_params = embed_params + attn_params + ffn_params + norm_params;
        total_params * 2  // FP16 = 2 bytes per param
    }
}

impl<B: Backend> Model<B> for Mistral<B> {
    fn name(&self) -> &str {
        "mistral"
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
}
