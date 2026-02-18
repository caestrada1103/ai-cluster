//! Model parallelism strategies
//!
//! This module implements various parallelism strategies for running
//! models across multiple GPUs.
//!
//! Note: Current implementation stubs out complex tensor/pipeline parallelism
//! to focus on single-GPU and simple data-parallel correctness first.

use std::sync::Arc;
use tokio::sync::Mutex;
use burn::tensor::backend::Backend;

use crate::error::WorkerError;
use crate::models::{Model, ModelOutput, ModelInput, TokenStream};
use crate::gpu_manager::GPUManager;

/// Parallel execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStrategy {
    /// Single GPU (no parallelism)
    Single,
    /// Data parallelism (replicate model on multiple GPUs)
    DataParallel,
    /// Tensor parallelism (split layers across GPUs)
    TensorParallel,
    /// Pipeline parallelism (split layers across GPUs)
    PipelineParallel,
}

/// A wrapper around a model that handles parallel execution.
pub struct ParallelModel<B: Backend> {
    /// The underlying model (wrapped in Mutex because Burn models are !Sync)
    model: Arc<Mutex<dyn Model<B>>>,
    
    /// The parallelism strategy
    strategy: ParallelStrategy,
    
    /// The GPU manager
    gpu_manager: Arc<GPUManager>,
    
    /// The device IDs this model is running on
    device_ids: Vec<usize>,
}

impl<B: Backend> ParallelModel<B> {
    /// Create a new parallel model wrapper
    pub fn new(
        model: Arc<Mutex<dyn Model<B>>>,
        strategy: ParallelStrategy,
        gpu_manager: Arc<GPUManager>,
        device_ids: Vec<usize>,
    ) -> Self {
        Self {
            model,
            strategy,
            gpu_manager,
            device_ids,
        }
    }

    /// Forward pass with parallelism handling
    pub async fn forward(&self, input: ModelInput<B>) -> Result<ModelOutput<B>, WorkerError> {
        match self.strategy {
            ParallelStrategy::Single => {
                let model = self.model.lock().await;
                model.forward(input)
            }
            ParallelStrategy::DataParallel => self.data_parallel_forward(input).await,
            ParallelStrategy::TensorParallel => self.tensor_parallel_forward(input).await,
            ParallelStrategy::PipelineParallel => self.pipeline_forward(input).await,
        }
    }

    /// Generate text with parallelism handling
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<TokenStream, WorkerError> {
        // For generation, we primarily support single GPU or data parallel (which just picks one)
        // Tensor/Pipeline parallel generation is complex and not yet implemented.
        match self.strategy {
            ParallelStrategy::TensorParallel | ParallelStrategy::PipelineParallel => {
                return Err(WorkerError::Internal("Parallel generation not implemented".into()));
            }
            _ => {
                let model = self.model.lock().await;
                model.generate(prompt, max_tokens, temperature, top_p, top_k)
            }
        }
    }

    /// Data parallel forward pass
    async fn data_parallel_forward(&self, input: ModelInput<B>) -> Result<ModelOutput<B>, WorkerError> {
        // For data parallelism, we ideally split the batch across GPUs.
        // Here we implement a simple version that runs on the primary device.
        let model = self.model.lock().await;
        model.forward(input)
    }

    /// Tensor parallel forward pass
    async fn tensor_parallel_forward(&self, _input: ModelInput<B>) -> Result<ModelOutput<B>, WorkerError> {
         // TODO: Implement tensor parallelism
         // This requires splitting weights and synchronizing activations.
         todo!("Tensor parallelism not implemented")
    }

    /// Pipeline parallel forward pass
    async fn pipeline_forward(&self, _input: ModelInput<B>) -> Result<ModelOutput<B>, WorkerError> {
        // TODO: Implement pipeline parallelism
        // This requires splitting layers and passing activations between GPUs.
        todo!("Pipeline parallelism not implemented")
    }
}

/// Create a parallel model from a loaded model
pub async fn create_parallel_model<B: Backend>(
    model: Arc<Mutex<dyn Model<B>>>,
    strategy: ParallelStrategy,
    gpu_manager: Arc<GPUManager>,
    device_ids: Vec<usize>,
) -> Result<ParallelModel<B>, WorkerError> {
    if device_ids.is_empty() {
        return Err(WorkerError::Config("No devices specified for parallel model".into()));
    }

    Ok(ParallelModel::new(model, strategy, gpu_manager, device_ids))
}