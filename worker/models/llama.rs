//! Llama model implementation
//!
//! Implements Meta's Llama architecture with grouped-query attention,
//! RMS normalization, and SwiGLU activation.

use burn::{
    module::{Module, Ignored},
    nn::{Linear, LinearConfig, Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor},
};
use super::{Model, ModelConfig, ModelOutput, ModelInput, TokenStream};
use super::common::{RMSNorm, RotaryEmbedding, swiglu, repeat_kv};
use crate::error::WorkerError;

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Llama attention mechanism with Grouped-Query Attention (GQA).
#[derive(Module, Debug)]
pub struct LlamaAttention<B: Backend> {
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

    /// Forward pass
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
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

        // Attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
        let attn = burn::tensor::activation::softmax(attn, 3);
        let output = attn.matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(output)
    }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// Llama MLP with SwiGLU activation
#[derive(Module, Debug)]
pub struct LlamaMLP<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
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
    attention: LlamaAttention<B>,
    mlp: LlamaMLP<B>,
    input_layernorm: RMSNorm<B>,
    post_attention_layernorm: RMSNorm<B>,
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
    ) -> Tensor<B, 3> {
        let residual = input.clone();
        let x = self.input_layernorm.forward(input);
        let x = self.attention.forward(x, rope, start_pos);
        let x = x + residual;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        x + residual
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

/// Complete Llama model
#[derive(Module, Debug)]
pub struct Llama<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<LlamaLayer<B>>,
    norm: RMSNorm<B>,
    lm_head: Linear<B>,
    config: Ignored<LlamaConfig>,
    rope: RotaryEmbedding<B>,
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
    pub fn new(config: &LlamaConfig, device: &B::Device) -> Self {
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
        }
    }

    /// Forward pass through the model
    pub fn forward_pass(
        &self,
        input_ids: Tensor<B, 2>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let mut x = self.embed_tokens.forward(input_ids.int());

        for layer in &self.layers {
            x = layer.forward(x, &self.rope, start_pos);
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
        (embed + attn + ffn + norm) * 2
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
}