//! Llama model implementation
//!
//! Implements Meta's Llama architecture with grouped-query attention,
//! RMS normalization, and SwiGLU activation.

use std::sync::Arc;

use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig,
        Embedding, EmbeddingConfig,
        RmsNorm, RmsNormConfig,
        Dropout, DropoutConfig,
        cache::Cache,
    },
    tensor::{Tensor, backend::Backend},
    tensor::activation::silu,
    nn::transformer::{RotaryEncoding, RotaryEncodingConfig},
    config::Config,
};
use tracing::{info, debug, instrument};
use async_trait::async_trait;

use super::{
    Model, ModelConfig, ModelOutput, ModelInput, TokenStream,
    common::SwiGLU,
};
use crate::cluster::Quantization;
use crate::error::WorkerError;

/// Llama attention mechanism with Grouped-Query Attention (GQA)
#[derive(Module, Debug)]
pub struct LlamaAttention<B: Backend> {
    /// Query projection
    q_proj: Linear<B>,
    
    /// Key projection
    k_proj: Linear<B>,
    
    /// Value projection
    v_proj: Linear<B>,
    
    /// Output projection
    o_proj: Linear<B>,
    
    /// Number of query heads
    num_heads: usize,
    
    /// Number of key/value heads (for GQA)
    num_kv_heads: usize,
    
    /// Head dimension
    head_dim: usize,
    
    /// Rotary positional embeddings
    rotary: RotaryEncoding<B>,
    
    /// Attention dropout
    dropout: Dropout,
    
    /// Scale factor for attention scores
    scale: f64,
}

impl<B: Backend> LlamaAttention<B> {
    /// Create new Llama attention layer
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f32,
        dropout_prob: f64,
        device: &B::Device,
    ) -> Self {
        assert_eq!(hidden_size, num_heads * head_dim);
        
        Self {
            q_proj: LinearConfig::new(hidden_size, num_heads * head_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(hidden_size, num_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(hidden_size, num_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(num_heads * head_dim, hidden_size)
                .with_bias(false)
                .init(device),
            num_heads,
            num_kv_heads,
            head_dim,
            rotary: RotaryEncodingConfig::new(head_dim)
                .with_max_seq_len(max_seq_len)
                .with_theta(rope_theta)
                .init(device),
            dropout: DropoutConfig::new(dropout_prob).init(),
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }
    
    /// Forward pass with optional cache
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        cache: Option<&mut Cache<B>>,
    ) -> Tensor<B, 3> {
        let (batch_size, seq_len, _) = hidden.dims();
        
        // Project to queries, keys, values
        let q = self.q_proj.forward(hidden.clone()); // [batch, seq, num_heads * head_dim]
        let k = self.k_proj.forward(hidden.clone()); // [batch, seq, num_kv_heads * head_dim]
        let v = self.v_proj.forward(hidden); // [batch, seq, num_kv_heads * head_dim]
        
        // Reshape for multi-head attention
        let q = q.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);
        
        // Apply rotary embeddings
        let q = self.rotary.forward(q);
        let k = self.rotary.forward(k);
        
        // Handle cache for incremental decoding
        let (k, v) = if let Some(cache) = cache {
            cache.update(k, v)
        } else {
            (k, v)
        };
        
        // Repeat k/v heads to match query heads (GQA)
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);
        
        // Transpose for attention computation
        let q = q.transpose(1, 2); // [batch, num_heads, seq, head_dim]
        let k = k.transpose(1, 2); // [batch, num_heads, seq, head_dim]
        let v = v.transpose(1, 2); // [batch, num_heads, seq, head_dim]
        
        // Compute attention scores
        let scores = q.matmul(&k.transpose(2, 3)) * self.scale; // [batch, num_heads, seq, seq]
        
        // Apply causal mask
        let scores = match mask {
            Some(m) => scores.mask_fill(m, f32::NEG_INFINITY),
            None => scores,
        };
        
        // Softmax and dropout
        let probs = scores.softmax(-1);
        let probs = self.dropout.forward(probs);
        
        // Apply attention to values
        let output = probs.matmul(v); // [batch, num_heads, seq, head_dim]
        
        // Transpose and reshape
        let output = output.transpose(1, 2); // [batch, seq, num_heads, head_dim]
        let output = output.reshape([batch_size, seq_len, self.num_heads * self.head_dim]);
        
        // Final projection
        self.o_proj.forward(output)
    }
    
    /// Repeat key/value heads to match number of query heads (GQA)
    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (batch_size, seq_len, num_kv_heads, head_dim) = x.dims();
        
        if self.num_heads == num_kv_heads {
            x
        } else {
            let num_reps = self.num_heads / num_kv_heads;
            
            // Repeat each head num_reps times
            let expanded = x.unsqueeze::<5>(); // [batch, seq, 1, num_kv_heads, head_dim]
            let repeated = expanded.repeat(2, num_reps); // [batch, seq, num_reps, num_kv_heads, head_dim]
            
            // Reshape to combine the repetition dimension with num_kv_heads
            repeated.reshape([batch_size, seq_len, self.num_heads, head_dim])
        }
    }
}

/// Llama MLP with SwiGLU activation
#[derive(Module, Debug)]
pub struct LlamaMLP<B: Backend> {
    /// Gate projection (for SwiGLU)
    gate_proj: Linear<B>,
    
    /// Up projection
    up_proj: Linear<B>,
    
    /// Down projection
    down_proj: Linear<B>,
}

impl<B: Backend> LlamaMLP<B> {
    /// Create new Llama MLP
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
    
    /// Forward pass with SwiGLU
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(input.clone()));
        let up = self.up_proj.forward(input);
        let hidden = gate * up;
        self.down_proj.forward(hidden)
    }
}

/// Llama transformer layer
#[derive(Module, Debug)]
pub struct LlamaLayer<B: Backend> {
    /// Self-attention
    attention: LlamaAttention<B>,
    
    /// MLP
    mlp: LlamaMLP<B>,
    
    /// Pre-attention layer norm
    input_layernorm: RmsNorm<B>,
    
    /// Pre-MLP layer norm
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> LlamaLayer<B> {
    /// Create new Llama layer
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        max_seq_len: usize,
        rope_theta: f32,
        rms_norm_eps: f32,
        device: &B::Device,
    ) -> Self {
        Self {
            attention: LlamaAttention::new(
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seq_len,
                rope_theta,
                0.0, // Llama doesn't use attention dropout
                device,
            ),
            mlp: LlamaMLP::new(hidden_size, intermediate_size, device),
            input_layernorm: RmsNormConfig::new(hidden_size)
                .with_eps(rms_norm_eps)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(hidden_size)
                .with_eps(rms_norm_eps)
                .init(device),
        }
    }
    
    /// Forward pass through layer
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        cache: Option<&mut Cache<B>>,
    ) -> Tensor<B, 3> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(input.clone());
        let attn_out = self.attention.forward(normed, mask, cache);
        
        // First residual
        let after_attn = attn_out + input.clone();
        
        // Pre-norm MLP
        let normed_mlp = self.post_attention_layernorm.forward(after_attn.clone());
        let mlp_out = self.mlp.forward(normed_mlp);
        
        // Second residual
        mlp_out + after_attn
    }
}

/// Complete Llama model
#[derive(Module, Debug)]
pub struct Llama<B: Backend> {
    /// Token embeddings
    embed_tokens: Embedding<B>,
    
    /// Transformer layers
    layers: Vec<LlamaLayer<B>>,
    
    /// Final layer norm
    norm: RmsNorm<B>,
    
    /// Output projection (sometimes tied with embeddings)
    lm_head: Linear<B>,
    
    /// Whether to tie weights
    tie_word_embeddings: bool,
    
    /// Model configuration
    config: Arc<ModelConfig>,
    
    /// Load timestamp
    loaded_at: std::time::SystemTime,
    
    /// Inference count
    inference_count: std::sync::atomic::AtomicU64,
    
    /// GPU IDs this model is loaded on
    gpu_ids: Vec<usize>,
    
    /// Quantization type
    quantization: Quantization,
}

/// Llama configuration
#[derive(Debug, Clone, Config)]
pub struct LlamaConfig {
    /// Hidden size
    pub hidden_size: usize,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Number of attention heads
    pub num_attention_heads: usize,
    
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    
    /// Vocabulary size
    pub vocab_size: usize,
    
    /// Maximum sequence length
    pub max_seq_len: usize,
    
    /// Intermediate size (FFN dimension)
    pub intermediate_size: usize,
    
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    
    /// Rotary embedding theta
    pub rope_theta: f32,
    
    /// Whether to tie word embeddings
    pub tie_word_embeddings: bool,
}

impl LlamaConfig {
    /// Create configuration for Llama 3 8B
    pub fn llama3_8b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,  // GQA with 8 KV heads
            vocab_size: 128256,
            max_seq_len: 8192,
            intermediate_size: 14336,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            tie_word_embeddings: false,
        }
    }
    
    /// Create configuration for Llama 3 70B
    pub fn llama3_70b() -> Self {
        Self {
            hidden_size: 8192,
            num_layers: 80,
            num_attention_heads: 64,
            num_kv_heads: 8,  // GQA with 8 KV heads
            vocab_size: 128256,
            max_seq_len: 8192,
            intermediate_size: 28672,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            tie_word_embeddings: false,
        }
    }
    
    /// Create configuration for Llama 2 7B
    pub fn llama2_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,  // MHA (no GQA)
            vocab_size: 32000,
            max_seq_len: 4096,
            intermediate_size: 11008,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
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
    pub fn new(
        config: &LlamaConfig,
        device: &B::Device,
    ) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(LlamaLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_kv_heads,
                head_dim,
                config.intermediate_size,
                config.max_seq_len,
                config.rope_theta,
                config.rms_norm_eps,
                device,
            ));
        }
        
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .init(device);
        
        let lm_head = if config.tie_word_embeddings {
            // In tied embeddings, we'll share the weight matrix
            LinearConfig::new(config.hidden_size, config.vocab_size)
                .with_bias(false)
                .init(device)
                .with_weights(embed_tokens.weight().clone().transpose())
        } else {
            LinearConfig::new(config.hidden_size, config.vocab_size)
                .with_bias(false)
                .init(device)
        };
        
        Self {
            embed_tokens,
            layers,
            norm: RmsNormConfig::new(config.hidden_size)
                .with_eps(config.rms_norm_eps)
                .init(device),
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
            config: Arc::new(config.to_model_config()),
            loaded_at: std::time::SystemTime::now(),
            inference_count: std::sync::atomic::AtomicU64::new(0),
            gpu_ids: vec![0],
            quantization: Quantization::Fp16,
        }
    }
    
    /// Forward pass through the model
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, i64>,
        mask: Option<Tensor<B, 2>>,
        cache: Option<&mut Cache<B>>,
    ) -> Tensor<B, 3> {
        // Token embeddings
        let mut hidden = self.embed_tokens.forward(input_ids);
        
        // Apply transformer layers
        for layer in &self.layers {
            hidden = layer.forward(hidden, mask.clone(), cache.as_deref_mut());
        }
        
        // Final normalization
        hidden = self.norm.forward(hidden);
        
        // Output projection
        self.lm_head.forward(hidden)
    }
    
    /// Generate logits for the next token (with caching)
    pub fn forward_next(
        &self,
        input_ids: Tensor<B, 2, i64>,
        cache: &mut Cache<B>,
    ) -> Tensor<B, 2> {
        let logits = self.forward(input_ids, None, Some(cache));
        
        // Take last token's logits
        let last_logits = logits.slice([0..1, -1.., 0..self.config.vocab_size]);
        last_logits.squeeze(1)
    }
}

#[async_trait]
impl<B: Backend> Model<B> for Llama<B> {
    fn name(&self) -> &str {
        "llama"
    }
    
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    
    async fn forward(&self, input: ModelInput<B>) -> Result<ModelOutput<B>, WorkerError> {
        let input_ids = input.clone().int();
        let output = self.forward(input_ids, None, None);
        Ok(output)
    }
    
    async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<TokenStream<B>, WorkerError> {
        self.inference_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        Ok(TokenStream::new(
            Arc::new(self.clone()),
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
        ))
    }
    
    fn memory_used(&self) -> usize {
        // Estimate based on config
        let num_params = self.config.vocab_size * self.config.hidden_size
            + self.layers.len() * (
                self.config.hidden_size * self.config.hidden_size * 3  // Q,K,V projections
                + self.config.hidden_size * self.config.hidden_size    // O projection
                + self.config.hidden_size * self.config.intermediate_size * 3  // MLP
            );
        
        // FP16 = 2 bytes per param
        num_params * 2
    }
    
    fn gpu_ids(&self) -> &[usize] {
        &self.gpu_ids
    }
    
    fn quantization(&self) -> Quantization {
        self.quantization
    }
    
    fn parallelism(&self) -> crate::cluster::ParallelismStrategy {
        if self.layers.len() > 1 {
            crate::cluster::ParallelismStrategy::Pipeline
        } else {
            crate::cluster::ParallelismStrategy::SingleDevice
        }
    }
    
    fn loaded_at(&self) -> std::time::SystemTime {
        self.loaded_at
    }
    
    fn inference_count(&self) -> u64 {
        self.inference_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}