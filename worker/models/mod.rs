//! Model implementations for various architectures
//!
//! This module contains the implementations of different model architectures
//! supported by the AI cluster, including DeepSeek, Llama, and Mistral.

mod deepseek;
mod llama;
mod mistral;
mod common;

pub use deepseek::{DeepSeek, DeepSeekConfig};
pub use llama::{Llama, LlamaConfig};
pub use mistral::{Mistral, MistralConfig};
pub use common::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use burn::tensor::{Tensor, backend::Backend};
use crate::error::WorkerError;

/// Type alias for model output (logits: [batch, seq, vocab])
pub type ModelOutput<B> = Tensor<B, 3>;

/// Type alias for model input (token IDs: [batch, seq])
pub type ModelInput<B> = Tensor<B, 2>;

/// Trait that all models must implement.
///
/// Methods are synchronous because inference is GPU-bound, not I/O-bound.
/// The surrounding runtime handles async scheduling.
pub trait Model<B: Backend>: Send + 'static {
    /// Get the model name
    fn name(&self) -> &str;

    /// Get the model configuration
    fn config(&self) -> ModelConfig;

    /// Forward pass through the model
    fn forward(&self, input: ModelInput<B>) -> Result<ModelOutput<B>, WorkerError>;

    /// Generate text from a prompt  
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<TokenStream, WorkerError>;

    /// Get memory usage in bytes
    fn memory_used(&self) -> usize;
}

/// Configuration common to all models
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture name
    pub architecture: String,

    /// Number of layers
    pub num_layers: usize,

    /// Hidden size
    pub hidden_size: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of KV heads (for GQA/MQA)
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

    /// Whether model uses MoE
    pub is_moe: bool,

    /// Number of experts (for MoE models)
    pub num_experts: Option<usize>,

    /// Number of experts per token (for MoE models)
    pub num_experts_per_tok: Option<usize>,
}

/// A loaded model instance with metadata for lifecycle management.
///
/// This is backend-agnostic — it stores metadata only. The actual model
/// (which is parameterized by `B: Backend`) lives behind an `Arc<dyn Model<B>>`
/// inside worker internals. `ModelInstance` is used for observation and
/// lifecycle tracking without needing to know the concrete backend.
#[derive(Clone)]
pub struct ModelInstance {
    /// Model name
    name: String,

    /// Memory used in bytes
    memory_bytes: usize,

    /// GPU IDs this model is loaded on
    gpu_ids: Vec<u32>,

    /// Quantization type
    quantization: i32,

    /// Parallelism strategy
    parallelism: i32,

    /// Load timestamp
    loaded_at: chrono::DateTime<chrono::Utc>,

    /// Inference count
    inference_count: Arc<AtomicU64>,
}

impl ModelInstance {
    /// Create a new model instance
    pub fn new(
        name: String,
        memory_bytes: usize,
        gpu_ids: Vec<u32>,
        quantization: i32,
        parallelism: i32,
    ) -> Self {
        Self {
            name,
            memory_bytes,
            gpu_ids,
            quantization,
            parallelism,
            loaded_at: chrono::Utc::now(),
            inference_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get memory usage in bytes
    pub fn memory_used(&self) -> usize {
        self.memory_bytes
    }

    /// Get GPU IDs
    pub fn gpu_ids(&self) -> &[u32] {
        &self.gpu_ids
    }

    /// Get quantization type (as protobuf enum i32)
    pub fn quantization(&self) -> i32 {
        self.quantization
    }

    /// Get parallelism strategy (as protobuf enum i32)
    pub fn parallelism(&self) -> i32 {
        self.parallelism
    }

    /// Get load timestamp
    pub fn loaded_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.loaded_at
    }

    /// Get inference count
    pub fn inference_count(&self) -> u64 {
        self.inference_count.load(Ordering::Relaxed)
    }

    /// Increment inference count
    pub fn record_inference(&self) {
        self.inference_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Generate text (placeholder — delegates to underlying model)
    pub async fn generate(
        &self,
        _prompt: &str,
        _max_tokens: usize,
        _temperature: f32,
        _top_p: f32,
        _top_k: usize,
    ) -> Result<TokenStream, WorkerError> {
        // In a real implementation, this would dispatch to the loaded model.
        // For now, return a placeholder stream.
        Ok(TokenStream::new(_max_tokens))
    }
}

/// Backend-agnostic token stream for generation.
///
/// This simplified version doesn't hold model references so it's
/// trivially `Send + Sync + Unpin`.
pub struct TokenStream {
    /// Current position
    position: usize,

    /// Maximum tokens to generate
    max_tokens: usize,
}

impl TokenStream {
    /// Create a new token stream
    pub fn new(max_tokens: usize) -> Self {
        Self {
            position: 0,
            max_tokens,
        }
    }

    /// Get next token in the stream
    pub async fn next(&mut self) -> Option<Result<String, WorkerError>> {
        if self.position >= self.max_tokens {
            return None;
        }

        self.position += 1;
        Some(Ok(" generated".to_string()))
    }
}

impl futures::Stream for TokenStream {
    type Item = Result<String, WorkerError>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // SAFETY: TokenStream is Unpin (no self-referential fields)
        let this = self.get_mut();
        if this.position >= this.max_tokens {
            return std::task::Poll::Ready(None);
        }

        this.position += 1;
        std::task::Poll::Ready(Some(Ok(" generated".to_string())))
    }
}