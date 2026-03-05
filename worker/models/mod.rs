//! Model implementations for various architectures
//!
//! This module contains the implementations of different model architectures
//! supported by the AI cluster, including DeepSeek, Llama, and Mistral.

pub mod deepseek;
pub mod llama;
pub mod mistral;
pub mod common;



use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tracing::debug;
use crate::error::WorkerError;

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
    #[allow(dead_code)]
    pub is_moe: bool,

    /// Number of experts (for MoE models)
    #[allow(dead_code)]
    pub num_experts: Option<usize>,

    /// Number of experts per token (for MoE models)
    #[allow(dead_code)]
    pub num_experts_per_tok: Option<usize>,
}

/// A loaded model instance with metadata for lifecycle management.
///
/// This is backend-agnostic — it stores metadata only. The actual model
/// (which is parameterized by `B: Backend`) lives behind an `Arc<dyn Model<B>>`
/// inside worker internals. `ModelInstance` is used for observation and
/// lifecycle tracking without needing to know the concrete backend.
use futures::Stream;
use std::pin::Pin;

/// Pinned, heap-allocated, `Send`-able stream of generated token chunks.
pub type TextStream = Pin<Box<dyn Stream<Item = Result<String, WorkerError>> + Send>>;

/// Trait for type-erased text generation
pub trait TextGeneration: Send {
    /// Generate text stream
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<TextStream, WorkerError>;
}

/// A loaded model instance with metadata for lifecycle management.
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

    /// The actual model (type-erased, behind Mutex for Sync)
    model: Option<Arc<Mutex<dyn TextGeneration + Send>>>,
}

impl ModelInstance {
    /// Create a new model instance
    pub fn new(
        name: String,
        memory_bytes: usize,
        gpu_ids: Vec<u32>,
        quantization: i32,
        parallelism: i32,
        model: Option<Arc<Mutex<dyn TextGeneration + Send>>>,
    ) -> Self {
        Self {
            name,
            memory_bytes,
            gpu_ids,
            quantization,
            parallelism,
            loaded_at: chrono::Utc::now(),
            inference_count: Arc::new(AtomicU64::new(0)),
            model,
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

    /// Generate text (delegates to underlying model)
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<TextStream, WorkerError> {
        if let Some(model) = &self.model {
            let stream = {
                debug!("ModelInstance::generate starting for {} - waiting for Mutex", self.name);
                let guard = model.lock()
                    .map_err(|e| WorkerError::Internal(format!("Lock error: {}", e)))?;
                debug!("ModelInstance::generate acquired Mutex for {}", self.name);
                let res = guard.generate(prompt, max_tokens, temperature, top_p, top_k);
                debug!("ModelInstance::generate trait call finished for {}", self.name);
                res?
            }; // guard dropped here, stream is 'static
            Ok(stream)
        } else {
             // Placeholder for now (should be error or dummy stream)
             Ok(Box::pin(TokenStream::new(max_tokens)))
        }
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