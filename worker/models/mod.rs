//! Model implementations for various architectures
//!
//! This module contains the implementations of different model architectures
//! supported by the AI cluster, including DeepSeek, Llama, and Mistral.

pub mod common;
pub mod deepseek;
pub mod llama;
pub mod mistral;
pub mod qwen;

/// Re-export shared KV cache types used by llama, qwen, and deepseek.
#[allow(unused_imports)]
pub use llama::{KvCache, KvEntry};

use crate::error::WorkerError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use tracing::debug;

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

    /// Explicit head dimension from config.json (None → hidden_size / num_attention_heads).
    pub head_dim: Option<usize>,

    /// Whether q/k/v projections carry biases (Qwen2/2.5 style).
    pub attention_bias: bool,

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
/// Backend-agnostic — stores metadata only, not the concrete model.
use futures::Stream;
use std::pin::Pin;

/// Pinned, heap-allocated, `Send`-able stream of generated token chunks.
pub type TextStream = Pin<Box<dyn Stream<Item = Result<String, WorkerError>> + Send>>;

/// Trait for type-erased text generation
pub trait TextGeneration: Send {
    /// Generate text stream. `seed` = Some(n) gives deterministic sampling.
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: Option<u64>,
    ) -> Result<TextStream, WorkerError>;

    /// Real tokenized prompt length; None when the engine has no tokenizer.
    fn count_prompt_tokens(&self, _prompt: &str) -> Option<u32> {
        None
    }
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

    /// Real tokenized prompt length; None if no model or the lock is poisoned.
    pub fn count_prompt_tokens(&self, prompt: &str) -> Option<u32> {
        let model = self.model.as_ref()?;
        let guard = model.lock().ok()?;
        guard.count_prompt_tokens(prompt)
    }

    /// Generate text (delegates to underlying model)
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: Option<u64>,
    ) -> Result<TextStream, WorkerError> {
        if let Some(model) = &self.model {
            self.inference_count.fetch_add(1, Ordering::Relaxed);
            let stream = {
                debug!(
                    "ModelInstance::generate starting for {} - waiting for Mutex",
                    self.name
                );
                let guard = model
                    .lock()
                    .map_err(|e| WorkerError::Internal(format!("Lock error: {}", e)))?;
                debug!("ModelInstance::generate acquired Mutex for {}", self.name);
                let res = guard.generate(prompt, max_tokens, temperature, top_p, top_k, seed);
                debug!(
                    "ModelInstance::generate trait call finished for {}",
                    self.name
                );
                res?
            }; // guard dropped here, stream is 'static
            Ok(stream)
        } else {
            Err(WorkerError::Internal(format!(
                "Model instance {} holds no runnable model",
                self.name
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal engine with no tokenizer; only `generate` is implemented.
    struct DummyEngine;

    impl TextGeneration for DummyEngine {
        fn generate(
            &self,
            _prompt: &str,
            _max_tokens: usize,
            _temperature: f32,
            _top_p: f32,
            _top_k: usize,
            _seed: Option<u64>,
        ) -> Result<TextStream, WorkerError> {
            Err(WorkerError::Internal("not called in this test".into()))
        }
    }

    /// An engine that cannot count must report absence, not a guess.
    #[test]
    fn text_generation_default_prompt_token_count_is_none() {
        let engine = DummyEngine;
        assert_eq!(engine.count_prompt_tokens("hello world"), None);
    }
}
