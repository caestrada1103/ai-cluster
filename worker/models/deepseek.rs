//! DeepSeek model implementation
//!
//! DeepSeek models use Mixture of Experts (MoE) architecture with
//! specialized routing and load balancing mechanisms.

use burn::{
    module::{Module, Ignored},
    nn::{Linear, LinearConfig, Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor},
};
use super::{Model, ModelConfig, ModelOutput, ModelInput, TokenStream};
use super::common::{RMSNorm, RotaryEmbedding, swiglu, repeat_kv};
use crate::error::WorkerError;

// ---------------------------------------------------------------------------
// Expert Activation
// ---------------------------------------------------------------------------

/// Expert activation functions
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum ExpertActivation {
    /// GELU activation
    Gelu,
    /// SiLU activation (SwiGLU)
    Silu,
}

impl Default for ExpertActivation {
    fn default() -> Self {
        Self::Silu
    }
}

// ---------------------------------------------------------------------------
// Individual Expert
// ---------------------------------------------------------------------------

/// Individual expert in MoE layer
#[derive(Module, Debug)]
pub struct Expert<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> Expert<B> {
    /// Create a new expert
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            gate_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(false)
                .init(device),
        }
    }

    /// Forward pass through expert
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.forward(input.clone());
        let up = self.up_proj.forward(input);
        // Default to Silu (SwiGLU) for now
        self.down_proj.forward(swiglu(gate, up))
    }
}

// ---------------------------------------------------------------------------
// MoE Layer
// ---------------------------------------------------------------------------

/// DeepSeek Mixture of Experts layer
#[derive(Module, Debug)]
pub struct DeepSeekMoE<B: Backend> {
    /// Expert networks
    experts: Vec<Expert<B>>,
    /// Gating/routing network
    gate: Linear<B>,
    #[module(skip)]
    num_experts_per_tok: usize,
}

impl<B: Backend> DeepSeekMoE<B> {
    /// Create a new MoE layer
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        device: &B::Device,
    ) -> Self {
        let experts = (0..num_experts)
            .map(|_| Expert::new(hidden_size, intermediate_size, device))
            .collect();

        Self {
            experts,
            gate: LinearConfig::new(hidden_size, num_experts).with_bias(false).init(device),
            num_experts_per_tok,
        }
    }

    /// Forward pass with expert routing
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, hidden] = input.dims();

        // Compute routing probabilities
        let routing_logits = self.gate.forward(input.clone());
        let routing_probs = burn::tensor::activation::softmax(routing_logits, 2);

        // For now, use a simplified routing: weighted sum of all experts
        // (Production would use top-k sparse routing)
        let mut output = Tensor::zeros([batch, seq_len, hidden], &input.device());

        for (i, expert) in self.experts.iter().enumerate() {
            let expert_output = expert.forward(input.clone());
            let weight = routing_probs.clone().slice([0..batch, 0..seq_len, i..i+1]);
            output = output + expert_output * weight;
        }

        output
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// DeepSeek attention layer
#[derive(Module, Debug)]
pub struct DeepSeekAttention<B: Backend> {
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

impl<B: Backend> DeepSeekAttention<B> {
    pub fn new(config: &DeepSeekConfig, device: &B::Device) -> Self {
        let q_out = config.num_attention_heads * config.head_dim;
        let kv_out = config.num_kv_heads * config.head_dim;

        Self {
            q_proj: LinearConfig::new(config.hidden_size, q_out).with_bias(false).init(device),
            k_proj: LinearConfig::new(config.hidden_size, kv_out).with_bias(false).init(device),
            v_proj: LinearConfig::new(config.hidden_size, kv_out).with_bias(false).init(device),
            o_proj: LinearConfig::new(q_out, config.hidden_size).with_bias(false).init(device),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();

        let q = self.q_proj.forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self.k_proj.forward(x.clone())
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self.v_proj.forward(x)
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = rope.apply(q, k, start_pos);

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_rep);
        let v = repeat_kv(v, n_rep);

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
// Transformer Layer
// ---------------------------------------------------------------------------

/// DeepSeek Transformer layer
#[derive(Module, Debug)]
pub struct DeepSeekLayer<B: Backend> {
    attention: DeepSeekAttention<B>,
    moe: DeepSeekMoE<B>,
    input_layernorm: RMSNorm<B>,
    post_attention_layernorm: RMSNorm<B>,
}

impl<B: Backend> DeepSeekLayer<B> {
    pub fn new(config: &DeepSeekConfig, device: &B::Device) -> Self {
        Self {
            attention: DeepSeekAttention::new(config, device),
            moe: DeepSeekMoE::new(
                config.hidden_size,
                config.intermediate_size,
                config.num_experts,
                config.num_experts_per_tok,
                device,
            ),
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps as f64, device),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps as f64, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.attention.forward(x, rope, start_pos);
        let x = x + residual;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.moe.forward(x);
        x + residual
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

/// Complete DeepSeek model
#[derive(Module, Debug)]
pub struct DeepSeek<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<DeepSeekLayer<B>>,
    norm: RMSNorm<B>,
    lm_head: Linear<B>,
    config: Ignored<DeepSeekConfig>,
    context_device: Ignored<B::Device>,
    rope: RotaryEmbedding<B>,
}

/// DeepSeek configuration
#[derive(Debug, Clone)]
pub struct DeepSeekConfig {
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
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
}

impl DeepSeekConfig {
    /// Create configuration for DeepSeek 7B
    pub fn deepseek_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 30,
            num_attention_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            vocab_size: 102400,
            max_seq_len: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            num_experts: 64,
            num_experts_per_tok: 6,
        }
    }

    /// Create configuration for DeepSeek 67B
    pub fn deepseek_67b() -> Self {
        Self {
            hidden_size: 8192,
            num_layers: 95,
            num_attention_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 22016,
            vocab_size: 102400,
            max_seq_len: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            num_experts: 160,
            num_experts_per_tok: 6,
        }
    }

    /// Convert to generic ModelConfig
    pub fn to_model_config(&self) -> ModelConfig {
        ModelConfig {
            architecture: "deepseek".to_string(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
            intermediate_size: self.intermediate_size,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            is_moe: true,
            num_experts: Some(self.num_experts),
            num_experts_per_tok: Some(self.num_experts_per_tok),
        }
    }
}

impl<B: Backend> DeepSeek<B> {
    /// Create a new DeepSeek model
    pub fn new(
        config: DeepSeekConfig,
        device: &B::Device,
    ) -> Result<Self, WorkerError> {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;
        
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, hidden_size)
            .init(device);
            
        let layers = (0..num_layers)
            .map(|_| DeepSeekLayer::new(&config, device))
            .collect();
            
        let norm = RMSNorm::new(hidden_size, config.rms_norm_eps as f64, device);
        let lm_head = LinearConfig::new(hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        let rope = RotaryEmbedding::new(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            device,
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: Ignored(config),
            context_device: Ignored(device.clone()),
            rope,
        })
    }

    /// Forward pass
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
        let expert_ffn = c.num_layers * c.num_experts * 3 * c.hidden_size * c.intermediate_size;
        let norm = (c.num_layers * 2 + 1) * c.hidden_size;
        let routing = c.num_layers * c.hidden_size * c.num_experts;
        (embed + attn + expert_ffn + norm + routing) * 2
    }
}

impl<B: Backend> Model<B> for DeepSeek<B> {
    fn name(&self) -> &str {
        "deepseek"
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