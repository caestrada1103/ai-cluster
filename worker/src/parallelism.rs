//! Model parallelism strategies
//!
//! This module implements various parallelism strategies for running
//! models across multiple GPUs:
//! - Pipeline parallelism: Split layers across devices
//! - Tensor parallelism: Split tensors across devices
//! - Data parallelism: Replicate model, split batch
//! - Expert parallelism: Split experts across devices (MoE models)

use std::collections::HashMap;
use std::sync::Arc;

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
    nn::cache::Cache,
};
use tokio::sync::RwLock;
use tracing::{info, debug, warn};

use crate::cluster::ParallelismStrategy;
use crate::error::WorkerError;
use crate::gpu_manager::GPUManager;
use crate::models::Model;

/// Parallelism configuration
#[derive(Debug, Clone)]
pub struct ParallelismConfig {
    /// Strategy to use
    pub strategy: ParallelismStrategy,
    
    /// Number of pipeline stages
    pub pipeline_num_stages: usize,
    
    /// Number of microbatches for pipeline
    pub pipeline_num_microbatches: usize,
    
    /// Tensor parallel size
    pub tensor_parallel_size: usize,
    
    /// Data parallel replicas
    pub data_parallel_replicas: usize,
    
    /// Number of experts per GPU (for MoE)
    pub num_experts_per_gpu: usize,
    
    /// Whether to enable load balancing
    pub enable_load_balancing: bool,
}

impl Default for ParallelismConfig {
    fn default() -> Self {
        Self {
            strategy: ParallelismStrategy::Auto,
            pipeline_num_stages: 1,
            pipeline_num_microbatches: 4,
            tensor_parallel_size: 1,
            data_parallel_replicas: 1,
            num_experts_per_gpu: 8,
            enable_load_balancing: true,
        }
    }
}

/// Parallel model wrapper
pub struct ParallelModel<B: Backend> {
    /// Underlying model
    model: Arc<dyn Model<B>>,
    
    /// Parallelism configuration
    config: ParallelismConfig,
    
    /// GPU manager
    gpu_manager: Arc<GPUManager>,
    
    /// Device assignments for pipeline stages
    pipeline_devices: Option<Vec<B::Device>>,
    
    /// Tensor parallel groups
    tp_groups: Option<Vec<Vec<usize>>>,
    
    /// Expert to GPU mapping (for MoE)
    expert_map: Option<HashMap<usize, usize>>,
}

impl<B: Backend> ParallelModel<B> {
    /// Create a new parallel model
    pub fn new(
        model: Arc<dyn Model<B>>,
        config: ParallelismConfig,
        gpu_manager: Arc<GPUManager>,
    ) -> Self {
        Self {
            model,
            config,
            gpu_manager,
            pipeline_devices: None,
            tp_groups: None,
            expert_map: None,
        }
    }
    
    /// Initialize parallelism based on strategy
    pub async fn init_parallelism(&mut self) -> Result<(), WorkerError> {
        let num_gpus = self.gpu_manager.device_count();
        
        if num_gpus == 1 {
            self.config.strategy = ParallelismStrategy::SingleDevice;
            return Ok(());
        }
        
        match self.config.strategy {
            ParallelismStrategy::Auto => {
                // Auto-select best strategy based on model and hardware
                self.select_auto_strategy(num_gpus).await?;
            }
            ParallelismStrategy::Pipeline => {
                self.init_pipeline_parallelism(num_gpus).await?;
            }
            ParallelismStrategy::Tensor => {
                self.init_tensor_parallelism(num_gpus).await?;
            }
            ParallelismStrategy::Data => {
                self.init_data_parallelism(num_gpus).await?;
            }
            ParallelismStrategy::Expert => {
                self.init_expert_parallelism(num_gpus).await?;
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Auto-select best parallelism strategy
    async fn select_auto_strategy(&mut self, num_gpus: usize) -> Result<(), WorkerError> {
        let model_config = self.model.config();
        
        // Check if model is MoE
        if model_config.is_moe {
            // MoE models benefit from expert parallelism
            self.config.strategy = ParallelismStrategy::Expert;
            self.init_expert_parallelism(num_gpus).await?;
        } else if model_config.num_layers > num_gpus * 4 {
            // Many layers, use pipeline parallelism
            self.config.strategy = ParallelismStrategy::Pipeline;
            self.init_pipeline_parallelism(num_gpus).await?;
        } else {
            // Default to tensor parallelism
            self.config.strategy = ParallelismStrategy::Tensor;
            self.init_tensor_parallelism(num_gpus).await?;
        }
        
        info!("Auto-selected strategy: {:?}", self.config.strategy);
        
        Ok(())
    }
    
    /// Initialize pipeline parallelism
    async fn init_pipeline_parallelism(&mut self, num_gpus: usize) -> Result<(), WorkerError> {
        let num_layers = self.model.config().num_layers;
        let stages = self.config.pipeline_num_stages.min(num_gpus);
        let layers_per_stage = (num_layers + stages - 1) / stages;
        
        let mut devices = Vec::with_capacity(stages);
        for i in 0..stages {
            let device = self.gpu_manager.get_device(i)
                .ok_or_else(|| WorkerError::Gpu(format!("GPU {} not available", i)))?
                .device.clone();
            devices.push(device);
            
            info!("Pipeline stage {}: layers {}-{} on GPU {}",
                i,
                i * layers_per_stage,
                ((i + 1) * layers_per_stage).min(num_layers) - 1,
                i
            );
        }
        
        self.pipeline_devices = Some(devices);
        
        Ok(())
    }
    
    /// Initialize tensor parallelism
    async fn init_tensor_parallelism(&mut self, num_gpus: usize) -> Result<(), WorkerError> {
        let tp_size = self.config.tensor_parallel_size.min(num_gpus);
        let num_groups = num_gpus / tp_size;
        
        let mut groups = Vec::with_capacity(num_groups);
        for i in 0..num_groups {
            let group: Vec<usize> = (i * tp_size..(i + 1) * tp_size).collect();
            groups.push(group);
            
            // Enable P2P access within group
            for &gpu1 in &group {
                for &gpu2 in &group {
                    if gpu1 != gpu2 {
                        self.gpu_manager.enable_p2p(gpu1, gpu2).await?;
                    }
                }
            }
            
            info!("Tensor parallel group {}: GPUs {:?}", i, group);
        }
        
        self.tp_groups = Some(groups);
        
        Ok(())
    }
    
    /// Initialize data parallelism
    async fn init_data_parallelism(&mut self, num_gpus: usize) -> Result<(), WorkerError> {
        let replicas = self.config.data_parallel_replicas.min(num_gpus);
        
        info!("Data parallelism with {} replicas", replicas);
        
        // Data parallelism doesn't need special setup
        // Each replica gets its own copy of the model
        
        Ok(())
    }
    
    /// Initialize expert parallelism (for MoE models)
    async fn init_expert_parallelism(&mut self, num_gpus: usize) -> Result<(), WorkerError> {
        let model_config = self.model.config();
        
        if !model_config.is_moe {
            warn!("Expert parallelism requested but model is not MoE");
            return self.init_tensor_parallelism(num_gpus).await;
        }
        
        let num_experts = model_config.num_experts
            .ok_or_else(|| WorkerError::Parallelism("MoE model has no experts".to_string()))?;
        
        let experts_per_gpu = self.config.num_experts_per_gpu.min(num_experts);
        let num_gpus_needed = (num_experts + experts_per_gpu - 1) / experts_per_gpu;
        
        if num_gpus_needed > num_gpus {
            return Err(WorkerError::Parallelism(format!(
                "Need {} GPUs for expert parallelism, have {}",
                num_gpus_needed, num_gpus
            )));
        }
        
        let mut expert_map = HashMap::new();
        for expert_id in 0..num_experts {
            let gpu_id = expert_id / experts_per_gpu;
            expert_map.insert(expert_id, gpu_id);
        }
        
        self.expert_map = Some(expert_map);
        
        info!("Expert parallelism: {} experts distributed across {} GPUs ({} per GPU)",
            num_experts, num_gpus_needed, experts_per_gpu);
        
        Ok(())
    }
    
    /// Forward pass with parallelism
    pub async fn forward(
        &self,
        input: Tensor<B, 2>,
    ) -> Result<Tensor<B, 3>, WorkerError> {
        match self.config.strategy {
            ParallelismStrategy::SingleDevice => {
                // Single GPU - direct forward
                self.model.forward(input).await
            }
            ParallelismStrategy::Pipeline => {
                self.pipeline_forward(input).await
            }
            ParallelismStrategy::Tensor => {
                self.tensor_parallel_forward(input).await
            }
            ParallelismStrategy::Data => {
                self.data_parallel_forward(input).await
            }
            ParallelismStrategy::Expert => {
                self.expert_parallel_forward(input).await
            }
            _ => self.model.forward(input).await,
        }
    }
    
    /// Pipeline parallel forward pass
    async fn pipeline_forward(
        &self,
        input: Tensor<B, 2>,
    ) -> Result<Tensor<B, 3>, WorkerError> {
        // Get pipeline devices
        let devices = self.pipeline_devices.as_ref()
            .ok_or_else(|| WorkerError::Parallelism("Pipeline not initialized".to_string()))?;
        
        let num_stages = devices.len();
        let num_microbatches = self.config.pipeline_num_microbatches;
        
        // Split batch into microbatches
        let batch_size = input.dims()[0];
        let micro_batch_size = (batch_size + num_microbatches - 1) / num_microbatches;
        
        let mut microbatches = Vec::with_capacity(num_microbatches);
        for i in 0..num_microbatches {
            let start = i * micro_batch_size;
            let end = (start + micro_batch_size).min(batch_size);
            if start < end {
                let micro = input.clone().slice([start..end]);
                microbatches.push(micro);
            }
        }
        
        // Execute pipeline
        let mut outputs = Vec::with_capacity(microbatches.len());
        
        for micro in microbatches {
            // Move to first stage
            let mut hidden = micro.to_device(&devices[0]);
            
            // Process through pipeline stages
            for stage_device in devices {
                // This would actually run the layer(s) on that device
                // For now, just simulate
                hidden = hidden.to_device(stage_device);
                // hidden = self.run_pipeline_stage(hidden, stage_idx).await?;
            }
            
            outputs.push(hidden);
        }
        
        // Concatenate outputs
        let output = Tensor::cat(outputs, 0);
        
        Ok(output.unsqueeze())
    }
    
    /// Tensor parallel forward pass
    async fn tensor_parallel_forward(
        &self,
        input: Tensor<B, 2>,
    ) -> Result<Tensor<B, 3>, WorkerError> {
        // Get tensor parallel groups
        let groups = self.tp_groups.as_ref()
            .ok_or_else(|| WorkerError::Parallelism("Tensor parallel not initialized".to_string()))?;
        
        // For now, just use first group
        let group = &groups[0];
        
        // Split input across group
        let hidden_size = input.dims()[1];
        let split_size = hidden_size / group.len();
        
        let mut splits = Vec::with_capacity(group.len());
        for (i, &gpu_id) in group.iter().enumerate() {
            let device = self.gpu_manager.get_device(gpu_id)
                .ok_or_else(|| WorkerError::Gpu(format!("GPU {} not found", gpu_id)))?
                .device.clone();
            
            let start = i * split_size;
            let end = if i == group.len() - 1 { hidden_size } else { start + split_size };
            
            let split = input.clone().slice([0..input.dims()[0], start..end]);
            splits.push(split.to_device(&device));
        }
        
        // Process each split in parallel
        // This would run the model on each device with its split
        
        // For now, just concatenate back
        let mut outputs = Vec::new();
        for (i, split) in splits.iter().enumerate() {
            let device = self.gpu_manager.get_device(group[i])
                .unwrap()
                .device.clone();
            outputs.push(split.to_device(&device));
        }
        
        let output = Tensor::cat(outputs, 2);
        
        Ok(output)
    }
    
    /// Data parallel forward pass
    async fn data_parallel_forward(
        &self,
        input: Tensor<B, 2>,
    ) -> Result<Tensor<B, 3>, WorkerError> {
        // Split batch across replicas
        let batch_size = input.dims()[0];
        let replicas = self.config.data_parallel_replicas;
        let per_replica = (batch_size + replicas - 1) / replicas;
        
        let mut handles = Vec::with_capacity(replicas);
        
        for replica in 0..replicas {
            let start = replica * per_replica;
            let end = (start + per_replica).min(batch_size);
            
            if start < end {
                let replica_input = input.clone().slice([start..end]);
                
                // Spawn task for each replica
                let model = self.model.clone();
                let handle = tokio::spawn(async move {
                    model.forward(replica_input).await
                });
                
                handles.push(handle);
            }
        }
        
        // Collect results
        let mut outputs = Vec::new();
        for handle in handles {
            if let Ok(Ok(output)) = handle.await {
                outputs.push(output);
            }
        }
        
        // Combine outputs
        let output = Tensor::cat(outputs, 0);
        
        Ok(output)
    }
    
    /// Expert parallel forward pass (for MoE models)
    async fn expert_parallel_forward(
        &self,
        input: Tensor<B, 2>,
    ) -> Result<Tensor<B, 3>, WorkerError> {
        let expert_map = self.expert_map.as_ref()
            .ok_or_else(|| WorkerError::Parallelism("Expert parallel not initialized".to_string()))?;
        
        // Route tokens to appropriate GPUs based on expert routing
        // This is complex and model-specific
        
        // For now, just do normal forward
        self.model.forward(input).await
    }
    
    /// Generate text with parallelism
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<crate::models::TokenStream<B>, WorkerError> {
        // For generation, we typically use a single device
        // Parallelism is more important for training/batch inference
        self.model.generate(prompt, max_tokens, temperature, top_p, top_k).await
    }
}

/// Create a parallel model based on strategy
pub async fn create_parallel_model<B: Backend>(
    model: Arc<dyn Model<B>>,
    strategy: ParallelismStrategy,
    gpu_manager: Arc<GPUManager>,
    config: Option<ParallelismConfig>,
) -> Result<ParallelModel<B>, WorkerError> {
    let mut config = config.unwrap_or_default();
    config.strategy = strategy;
    
    let mut parallel_model = ParallelModel::new(model, config, gpu_manager);
    parallel_model.init_parallelism().await?;
    
    Ok(parallel_model)
}