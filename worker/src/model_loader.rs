//! Model loading and management
//!
//! This module handles loading models from various formats,
//! managing model instances, and coordinating with the GPU manager.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::backend::Backend,
};
use dashmap::DashMap;
use safetensors::SafeTensors;
use tokio::sync::{RwLock, Semaphore};
use tracing::{info, warn, error, debug};
use hf_hub::{api::sync::Api, Repo, RepoType};

use crate::cluster::{Quantization, ParallelismStrategy};
use crate::error::WorkerError;
use crate::gpu_manager::GPUManager;
use crate::models::{
    Model, ModelConfig, DeepSeek, DeepSeekConfig,
    Llama, LlamaConfig, Mistral, MistralConfig,
};

/// Model loader configuration
#[derive(Debug, Clone)]
pub struct ModelLoaderConfig {
    /// Directory to cache models
    pub cache_dir: PathBuf,
    
    /// Directory to download models
    pub download_dir: PathBuf,
    
    /// Maximum number of concurrent model loads
    pub max_concurrent_loads: usize,
    
    /// Load timeout in seconds
    pub load_timeout_secs: u64,
    
    /// Whether to verify checksums
    pub verify_checksums: bool,
    
    /// Whether to use memory mapping
    pub enable_mmap: bool,
    
    /// Whether to pin memory
    pub pin_memory: bool,
    
    /// Prefetch size in GB
    pub prefetch_size_gb: f32,
}

impl Default for ModelLoaderConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from("/data/models"),
            download_dir: PathBuf::from("/data/downloads"),
            max_concurrent_loads: 2,
            load_timeout_secs: 300,
            verify_checksums: true,
            enable_mmap: true,
            pin_memory: true,
            prefetch_size_gb: 2.0,
        }
    }
}

/// Model instance with metadata
pub struct ModelInstance {
    /// The actual model
    model: Arc<dyn Model<ActiveBackend>>,
    
    /// Model name
    name: String,
    
    /// Model configuration
    config: ModelConfig,
    
    /// GPUs this model is loaded on
    gpu_ids: Vec<usize>,
    
    /// Quantization type
    quantization: Quantization,
    
    /// Parallelism strategy
    parallelism: ParallelismStrategy,
    
    /// Load timestamp
    loaded_at: SystemTime,
    
    /// Memory used in bytes
    memory_used: usize,
    
    /// Inference count
    inference_count: Arc<std::sync::atomic::AtomicU64>,
}

impl ModelInstance {
    /// Create a new model instance
    pub fn new(
        model: Arc<dyn Model<ActiveBackend>>,
        name: String,
        config: ModelConfig,
        gpu_ids: Vec<usize>,
        quantization: Quantization,
        parallelism: ParallelismStrategy,
        memory_used: usize,
    ) -> Self {
        Self {
            model,
            name,
            config,
            gpu_ids,
            quantization,
            parallelism,
            loaded_at: SystemTime::now(),
            memory_used,
            inference_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
    
    /// Get the model
    pub fn model(&self) -> Arc<dyn Model<ActiveBackend>> {
        self.model.clone()
    }
    
    /// Get model name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get model config
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    
    /// Get GPU IDs
    pub fn gpu_ids(&self) -> &[usize] {
        &self.gpu_ids
    }
    
    /// Get quantization type
    pub fn quantization(&self) -> Quantization {
        self.quantization
    }
    
    /// Get parallelism strategy
    pub fn parallelism(&self) -> ParallelismStrategy {
        self.parallelism
    }
    
    /// Get load timestamp
    pub fn loaded_at(&self) -> SystemTime {
        self.loaded_at
    }
    
    /// Get memory used in bytes
    pub fn memory_used(&self) -> usize {
        self.memory_used
    }
    
    /// Get inference count
    pub fn inference_count(&self) -> u64 {
        self.inference_count.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// Increment inference count
    pub fn increment_inference_count(&self) {
        self.inference_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Model loader
pub struct ModelLoader {
    /// Configuration
    config: ModelLoaderConfig,
    
    /// GPU manager
    gpu_manager: Arc<GPUManager>,
    
    /// Loaded models (model_name -> ModelInstance)
    loaded_models: Arc<DashMap<String, ModelInstance>>,
    
    /// Load semaphore for limiting concurrent loads
    load_semaphore: Arc<Semaphore>,
    
    /// HuggingFace API client
    hf_api: Option<Api>,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(
        config: ModelLoaderConfig,
        gpu_manager: Arc<GPUManager>,
    ) -> Result<Self, WorkerError> {
        // Initialize HuggingFace API
        let hf_api = Api::new().ok();
        
        // Create directories if they don't exist
        std::fs::create_dir_all(&config.cache_dir)?;
        std::fs::create_dir_all(&config.download_dir)?;
        
        Ok(Self {
            config,
            gpu_manager,
            loaded_models: Arc::new(DashMap::new()),
            load_semaphore: Arc::new(Semaphore::new(config.max_concurrent_loads)),
            hf_api,
        })
    }
    
    /// Load a model
    pub async fn load_model(
        &self,
        model_name: &str,
        model_config: Option<&crate::cluster::ModelConfig>,
        gpu_ids: &[usize],
        quantization: Quantization,
        parallelism: ParallelismStrategy,
    ) -> Result<ModelInstance, WorkerError> {
        // Acquire load permit
        let _permit = self.load_semaphore.acquire().await.map_err(|e| {
            WorkerError::Resource(format!("Failed to acquire load permit: {}", e))
        })?;
        
        info!("Loading model {} with quantization {:?}", model_name, quantization);
        
        // Check if already loaded
        if let Some(instance) = self.loaded_models.get(model_name) {
            info!("Model {} already loaded", model_name);
            return Ok(instance.clone());
        }
        
        // Determine model path
        let model_path = self.get_model_path(model_name).await?;
        
        // Load model configuration
        let config = self.load_model_config(model_name, model_config).await?;
        
        // Allocate GPU memory
        self.allocate_model_memory(&config, gpu_ids).await?;
        
        // Load the actual model
        let model = self.load_model_weights(
            model_name,
            &config,
            gpu_ids,
            quantization,
            parallelism,
            &model_path,
        ).await?;
        
        // Calculate memory usage
        let memory_used = self.calculate_memory_usage(&config, quantization);
        
        // Create model instance
        let instance = ModelInstance::new(
            model,
            model_name.to_string(),
            config,
            gpu_ids.to_vec(),
            quantization,
            parallelism,
            memory_used,
        );
        
        // Store in cache
        self.loaded_models.insert(model_name.to_string(), instance.clone());
        
        info!("Model {} loaded successfully", model_name);
        
        Ok(instance)
    }
    
    /// Get model path (download if necessary)
    async fn get_model_path(&self, model_name: &str) -> Result<PathBuf, WorkerError> {
        // Check if model exists in cache
        let cache_path = self.config.cache_dir.join(model_name);
        if cache_path.exists() {
            return Ok(cache_path);
        }
        
        // Download from HuggingFace
        if let Some(api) = &self.hf_api {
            info!("Downloading model {} from HuggingFace", model_name);
            
            let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));
            
            // Download config
            let config_path = repo.get("config.json").map_err(|e| {
                WorkerError::ModelLoad(format!("Failed to download config: {}", e))
            })?;
            
            // Download model weights (safetensors)
            let mut model_paths = Vec::new();
            
            // Try to find safetensors index
            if let Ok(index_path) = repo.get("model.safetensors.index.json") {
                // Sharded model
                let index: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(index_path)?
                )?;
                
                if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                    for (_, filename) in weight_map {
                        let path = repo.get(filename.as_str().unwrap()).map_err(|e| {
                            WorkerError::ModelLoad(format!("Failed to download shard: {}", e))
                        })?;
                        model_paths.push(path);
                    }
                }
            } else {
                // Single file model
                if let Ok(path) = repo.get("model.safetensors") {
                    model_paths.push(path);
                } else if let Ok(path) = repo.get("pytorch_model.bin") {
                    model_paths.push(path);
                }
            }
            
            if model_paths.is_empty() {
                return Err(WorkerError::ModelLoad("No model weights found".to_string()));
            }
            
            // Copy to cache
            let target_dir = self.config.cache_dir.join(model_name);
            std::fs::create_dir_all(&target_dir)?;
            
            for path in model_paths {
                let filename = path.file_name().unwrap();
                std::fs::copy(&path, target_dir.join(filename))?;
            }
            
            // Copy config
            if let Ok(config_path) = repo.get("config.json") {
                std::fs::copy(config_path, target_dir.join("config.json"))?;
            }
            
            // Copy tokenizer
            for tokenizer_file in ["tokenizer.json", "tokenizer_config.json", "vocab.json"] {
                if let Ok(path) = repo.get(tokenizer_file) {
                    let _ = std::fs::copy(path, target_dir.join(tokenizer_file));
                }
            }
            
            return Ok(target_dir);
        }
        
        Err(WorkerError::ModelLoad(format!(
            "Model {} not found in cache and HuggingFace download not available",
            model_name
        )))
    }
    
    /// Load model configuration
    async fn load_model_config(
        &self,
        model_name: &str,
        provided_config: Option<&crate::cluster::ModelConfig>,
    ) -> Result<ModelConfig, WorkerError> {
        if let Some(config) = provided_config {
            // Use provided configuration
            return Ok(ModelConfig {
                architecture: config.architecture.clone(),
                num_layers: config.num_layers as usize,
                hidden_size: config.hidden_size as usize,
                num_attention_heads: config.num_attention_heads as usize,
                num_kv_heads: config.num_kv_heads as usize,
                vocab_size: config.vocab_size as usize,
                max_seq_len: config.max_position_embeddings as usize,
                intermediate_size: config.intermediate_size as usize,
                rms_norm_eps: config.rms_norm_eps,
                rope_theta: config.rope_theta,
                is_moe: false, // Would need to detect from config
                num_experts: None,
                num_experts_per_tok: None,
            });
        }
        
        // Load from config file
        let config_path = self.config.cache_dir
            .join(model_name)
            .join("config.json");
        
        if !config_path.exists() {
            return Err(WorkerError::ModelLoad(format!(
                "Config file not found at {:?}",
                config_path
            )));
        }
        
        let config_str = std::fs::read_to_string(config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        
        // Extract common fields
        let architecture = config_json
            .get("architectures")
            .and_then(|a| a.get(0))
            .and_then(|a| a.as_str())
            .unwrap_or("unknown")
            .to_string();
        
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        
        let num_layers = config_json
            .get("num_hidden_layers")
            .or_else(|| config_json.get("num_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        
        let hidden_size = config_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        
        let num_attention_heads = config_json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        
        let num_kv_heads = config_json
            .get("num_key_value_heads")
            .or_else(|| config_json.get("num_attention_heads"))
            .and_then(|v| v.as_u64())
            .unwrap_or(num_attention_heads as u64) as usize;
        
        let vocab_size = config_json
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        
        let max_seq_len = config_json
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;
        
        let intermediate_size = config_json
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        
        let rms_norm_eps = config_json
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;
        
        let rope_theta = config_json
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;
        
        // Detect MoE
        let is_moe = config_json.get("num_local_experts").is_some();
        let num_experts = config_json
            .get("num_local_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let num_experts_per_tok = config_json
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        
        Ok(ModelConfig {
            architecture,
            num_layers,
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            vocab_size,
            max_seq_len,
            intermediate_size,
            rms_norm_eps,
            rope_theta,
            is_moe,
            num_experts,
            num_experts_per_tok,
        })
    }
    
    /// Allocate memory for model
    async fn allocate_model_memory(
        &self,
        config: &ModelConfig,
        gpu_ids: &[usize],
    ) -> Result<(), WorkerError> {
        // Calculate memory requirements per GPU
        let memory_per_gpu = self.calculate_memory_usage(config, Quantization::Fp16) / gpu_ids.len();
        
        for &gpu_id in gpu_ids {
            self.gpu_manager.allocate_memory(
                gpu_id,
                memory_per_gpu as u64,
                format!("model:{}", config.architecture),
            ).await?;
        }
        
        Ok(())
    }
    
    /// Calculate memory usage for a model
    fn calculate_memory_usage(&self, config: &ModelConfig, quantization: Quantization) -> usize {
        // Rough estimate based on parameters
        // Each parameter: hidden_size * intermediate_size for each layer
        let num_params = config.vocab_size * config.hidden_size
            + config.num_layers * (
                config.hidden_size * config.hidden_size * 4 +  // Attention
                config.hidden_size * config.intermediate_size * 3  // MLP
            );
        
        // Add MoE parameters if applicable
        let num_params = if config.is_moe {
            if let (Some(num_experts), Some(per_tok)) = (config.num_experts, config.num_experts_per_tok) {
                num_params + config.num_layers * num_experts * config.intermediate_size * config.hidden_size * 2
            } else {
                num_params
            }
        } else {
            num_params
        };
        
        // Bytes per parameter based on quantization
        let bytes_per_param = match quantization {
            Quantization::Fp32 => 4,
            Quantization::Fp16 => 2,
            Quantization::Bf16 => 2,
            Quantization::Int8 => 1,
            Quantization::Int4 => 0, // Special handling
            Quantization::Fp8 => 1,
        };
        
        if bytes_per_param == 0 {
            // Int4: approximately 0.5 bytes per param + overhead for scales
            (num_params / 2) + (num_params / 64) * 4
        } else {
            num_params * bytes_per_param
        }
    }
    
    /// Load model weights
    async fn load_model_weights(
        &self,
        model_name: &str,
        config: &ModelConfig,
        gpu_ids: &[usize],
        quantization: Quantization,
        parallelism: ParallelismStrategy,
        model_path: &Path,
    ) -> Result<Arc<dyn Model<ActiveBackend>>, WorkerError> {
        // Determine model type from architecture
        let model: Arc<dyn Model<ActiveBackend>> = match config.architecture.to_lowercase() {
            arch if arch.contains("deepseek") => {
                let deepseek_config = DeepSeekConfig::deepseek_7b(); // Would load actual config
                let device = self.gpu_manager.get_device(gpu_ids[0])
                    .ok_or_else(|| WorkerError::Gpu("Invalid GPU ID".to_string()))?
                    .device.clone();
                Arc::new(DeepSeek::new(&deepseek_config, &device))
            }
            arch if arch.contains("llama") => {
                let llama_config = LlamaConfig::llama3_8b(); // Would load actual config
                let device = self.gpu_manager.get_device(gpu_ids[0])
                    .ok_or_else(|| WorkerError::Gpu("Invalid GPU ID".to_string()))?
                    .device.clone();
                Arc::new(Llama::new(&llama_config, &device))
            }
            arch if arch.contains("mistral") => {
                let mistral_config = MistralConfig::mistral_7b(); // Would load actual config
                let device = self.gpu_manager.get_device(gpu_ids[0])
                    .ok_or_else(|| WorkerError::Gpu("Invalid GPU ID".to_string()))?
                    .device.clone();
                Arc::new(Mistral::new(&mistral_config, &device))
            }
            _ => {
                return Err(WorkerError::ModelLoad(format!(
                    "Unsupported architecture: {}",
                    config.architecture
                )));
            }
        };
        
        // Load weights
        // This would use Burn's record system to load from file
        
        Ok(model)
    }
    
    /// Unload a model
    pub async fn unload_model(&self, model_name: &str) -> Result<(), WorkerError> {
        if let Some((_, instance)) = self.loaded_models.remove(model_name) {
            // Free GPU memory
            for &gpu_id in instance.gpu_ids() {
                self.gpu_manager.free_memory(&format!("model:{}", model_name)).await;
            }
            
            info!("Model {} unloaded", model_name);
        }
        
        Ok(())
    }
    
    /// Get loaded model
    pub fn get_model(&self, model_name: &str) -> Option<ModelInstance> {
        self.loaded_models.get(model_name).map(|entry| entry.clone())
    }
    
    /// List loaded models
    pub fn list_models(&self) -> Vec<String> {
        self.loaded_models.iter().map(|entry| entry.key().clone()).collect()
    }
    
    /// Get model info
    pub fn get_model_info(&self, model_name: &str) -> Option<serde_json::Value> {
        self.loaded_models.get(model_name).map(|entry| {
            let instance = entry.value();
            serde_json::json!({
                "name": instance.name(),
                "architecture": instance.config().architecture,
                "gpu_ids": instance.gpu_ids(),
                "quantization": format!("{:?}", instance.quantization()),
                "parallelism": format!("{:?}", instance.parallelism()),
                "memory_used_mb": instance.memory_used() / 1024 / 1024,
                "loaded_at": instance.loaded_at()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                "inference_count": instance.inference_count(),
            })
        })
    }
}

use crate::gpu_manager::ActiveBackend;