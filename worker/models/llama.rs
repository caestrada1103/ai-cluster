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
use tokenizers::Tokenizer;
use async_stream::stream;
use futures::Stream;
use std::pin::Pin;
use std::path::Path;

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

        // Attention with causal mask
        let scale = (self.head_dim as f64).sqrt();
        let attn_scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
        
        // Apply causal mask: positions can only attend to <= their own position
        let attn_scores = if seq_len > 1 {
            // Build lower-triangular mask [seq_len, seq_len]
            // 1.0 for allowed positions, 0.0 for masked (future) positions
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..=i {
                    mask_data[i * seq_len + j] = 1.0;
                }
            }
            let mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &attn_scores.device())
                .reshape([1, 1, seq_len, seq_len]);
            
            // Where mask == 0, set attention score to -inf
            let neg_inf = Tensor::<B, 4>::full([1, 1, seq_len, seq_len], -1e9, &attn_scores.device());
            let ones = Tensor::<B, 4>::full([1, 1, seq_len, seq_len], 1.0, &attn_scores.device());
            // masked_fill: attn = attn * mask + neg_inf * (1 - mask)
            attn_scores * mask.clone() + neg_inf * (ones - mask)
        } else {
            // Single token: no masking needed
            attn_scores
        };
        
        let attn = burn::tensor::activation::softmax(attn_scores, 3);
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
    pub fn new(config: &LlamaConfig, device: &B::Device, tokenizer_path: &Path) -> Self {
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
            .unwrap_or_else(|e| {
                eprintln!("[WARN] Failed to load tokenizer from {:?}: {}. Trying HF pretrained...", tok_file, e);
                // Try from_pretrained with the model name as last resort
                Tokenizer::from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", None)
                    .expect("Failed to load tokenizer")
            });

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
            tokenizer: Ignored(tokenizer),
            device: Ignored(device.clone()),
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
impl<B: Backend> TextGeneration for Llama<B> {
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, WorkerError>> + Send>>, WorkerError> {
        let device = self.device.clone();
        
        // Encode prompt
        // Helper to check for known special tokens and parsed prompts
        // We manually handle these to ensure they are encoded as single tokens (if in vocab)
        // rather than being split into constituent characters.
        let special_tokens = ["</s>", "<s>", "<|user|>", "<|assistant|>"];

        let mut tokens = Vec::new();
        let mut current_pos = 0;

        while current_pos < prompt.len() {
            // Find the nearest special token
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

            // Encode text before the match (or the rest of the string if no match)
            if min_idx > current_pos {
                let text_segment = &prompt[current_pos..min_idx];
                // Only add special tokens (BOS) for the very first segment if it's at the start
                let add_special = current_pos == 0; 
                let encoding = self.tokenizer.encode(text_segment, add_special)
                    .map_err(|e| WorkerError::Internal(format!("Tokenizer error: {}", e)))?;
                tokens.extend_from_slice(encoding.get_ids());
            }

            // Handle the matched special token
            if let Some(st) = best_match {
                if let Some(id) = self.tokenizer.token_to_id(st) {
                    tokens.push(id);
                } else {
                    // Fallback: special token not in vocab, encode as text
                    let encoding = self.tokenizer.encode(*st, false)
                        .map_err(|e| WorkerError::Internal(format!("Tokenizer error (special): {}", e)))?;
                    tokens.extend_from_slice(encoding.get_ids());
                }
                current_pos = min_idx + st.len();
            } else {
                // No more special tokens found, remainder processed above
                break;
            }
        }

        let prompt_len = tokens.len();
        
        // Clone model so stream doesn't borrow &self
        let model = self.clone();
        
        // Stream
        let stream = stream! {
            let mut prev_text_len = 0usize;
            for _ in 0..max_tokens {
                // Create input tensor
                let token_floats: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
                let len = token_floats.len();
                let input = Tensor::<B, 1>::from_floats(token_floats.as_slice(), &device).reshape([1, len]);
                
                // Forward pass
                let output = model.forward_pass(input, 0); 
                
                // Get last token logits
                let [_batch, seq, vocab] = output.dims();
                let last_logits = output.slice([0..1, seq-1..seq, 0..vocab]).reshape([vocab]);
                
                // Sample next token
                let logit_data: Vec<f32> = last_logits.into_data().to_vec().unwrap_or_default();
                
                let token_id = if temperature < 0.01 {
                    // Greedy (argmax)
                    logit_data.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i).unwrap_or(0)
                } else {
                    // Temperature + top-k/top-p sampling
                    super::common::top_k_top_p_sample(&logit_data, temperature, top_p, top_k)
                };
                let token_id_u32 = token_id as u32;
                
                tokens.push(token_id_u32);
                
                // Decode full generated sequence and emit delta (preserves spaces)
                let generated_ids = &tokens[prompt_len..];
                let full_text = model.tokenizer.decode(generated_ids, true)
                    .map_err(|e| WorkerError::Internal(format!("Decode error: {}", e)));
                    
                match full_text {
                    Ok(t) => {
                        let delta = t[prev_text_len..].to_string();
                        prev_text_len = t.len();
                        yield Ok(delta);
                    },
                    Err(e) => { yield Err(e); break; }
                }

                // Stop if EOS using tokenizer vocab
                if let Some(eos_id) = model.tokenizer.get_vocab(true).get("</s>") {
                     if token_id_u32 == *eos_id {
                         break;
                     }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}
