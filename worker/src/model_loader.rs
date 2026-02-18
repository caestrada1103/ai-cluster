//! Model loading and management
//!
//! This module handles loading models from various formats,
//! managing model instances, and coordinating with the GPU manager.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use dashmap::DashMap;
use tokio::sync::Semaphore;
use tracing::{info, warn, error};
use hf_hub::{api::tokio::Api, Repo, RepoType};

use crate::error::WorkerError;
use crate::gpu_manager::GPUManager;
use crate::models::{
    ModelConfig, ModelInstance,
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
            cache_dir: PathBuf::from("./data/models"),
            download_dir: PathBuf::from("./data/downloads"),
            max_concurrent_loads: 2,
            load_timeout_secs: 300,
            verify_checksums: true,
            enable_mmap: true,
            pin_memory: true,
            prefetch_size_gb: 2.0,
        }
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
        // We need to know if a token is available. 
        // But ModelLoader doesn't take WorkerConfig, only ModelLoaderConfig.
        // We should probably add token to ModelLoaderConfig or accept it here.
        // For now, let's try to get it from env or default.
        
        let mut builder = hf_hub::api::tokio::ApiBuilder::new()
            .with_endpoint("https://huggingface.co".to_string())
            .with_cache_dir(config.cache_dir.clone());
            
        if let Ok(token) = std::env::var("HF_TOKEN") {
             builder = builder.with_token(Some(token));
        }

        let hf_api = builder.build().ok();

        // Create directories if they don't exist
        std::fs::create_dir_all(&config.cache_dir)?;
        std::fs::create_dir_all(&config.download_dir)?;

        let max_concurrent = config.max_concurrent_loads;

        Ok(Self {
            config,
            gpu_manager,
            loaded_models: Arc::new(DashMap::new()),
            load_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            hf_api,
        })
    }
    /// Load a model
    pub async fn load_model(
        &self,
        model_name: &str,
        model_config: Option<&crate::cluster::ModelConfig>,
        gpu_ids: &[u32],
        quantization: crate::cluster::Quantization,
        parallelism: crate::cluster::ParallelismStrategy,
    ) -> Result<ModelInstance, WorkerError> {
        // Acquire load permit
        let _permit = self.load_semaphore.acquire().await.map_err(|e| {
            WorkerError::Resource(format!("Failed to acquire load permit: {}", e))
        })?;

        info!("Loading model {} with quantization {:?}", model_name, quantization);

        // Check if already loaded
        if let Some(entry) = self.loaded_models.get(model_name) {
            info!("Model {} already loaded", model_name);
            return Ok(entry.value().clone());
        }

        // Determine model path
        let model_path = self.get_model_path(model_name).await?;

        // Load model configuration
        let config = self.load_model_config(model_name, model_config, &model_path).await?;

        // Allocate GPU memory
        let memory_used = self.calculate_memory_usage(&config, quantization);
        for &gpu_id in gpu_ids {
            self.gpu_manager.allocate_memory(
                gpu_id as usize,
                memory_used as u64,
                format!("model:{}", model_name),
            ).await?;
        }

        // Create model instance
        let instance = ModelInstance::new(
            model_name.to_string(),
            memory_used,
            gpu_ids.to_vec(),
            quantization as i32,
            parallelism as i32,
        );

        // Store in cache
        self.loaded_models.insert(model_name.to_string(), instance.clone());

        info!(
            "Model {} loaded successfully, using {}MB VRAM",
            model_name,
            memory_used / 1024 / 1024
        );

        Ok(instance)
    }
    /// Get model path (download if necessary)
    async fn get_model_path(&self, model_name: &str) -> Result<PathBuf, WorkerError> {
        // Check if model exists in cache -> NO, hf_hub handles caching!
        // But we want to check if we can skip download checks?
        // hf_hub checks cache first.

        // Download from HuggingFace
        if let Some(api) = &self.hf_api {
            info!("Checking/Downloading model {} from HuggingFace", model_name);

            let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));

            // Download config
            let _config_path = repo.get("config.json").await.map_err(|e| {
                WorkerError::ModelLoad(format!("Failed to download config: {}", e))
            })?;

            // Download model weights (safetensors)
            let mut model_paths = Vec::new();

            // Try to find safetensors index
            if let Ok(index_path) = repo.get("model.safetensors.index.json").await {
                // Sharded model
                let index: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(index_path)?
                )?;

                if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                    for (_, filename) in weight_map {
                        if let Some(fname) = filename.as_str() {
                            let path = repo.get(fname).await.map_err(|e| {
                                WorkerError::ModelLoad(format!("Failed to download shard: {}", e))
                            })?;
                            model_paths.push(path);
                        }
                    }
                }
            } else {
                // Single file model
                if let Ok(path) = repo.get("model.safetensors").await {
                    model_paths.push(path);
                } else if let Ok(path) = repo.get("pytorch_model.bin").await {
                    model_paths.push(path);
                }
            }

            if model_paths.is_empty() {
                 // return Err(WorkerError::ModelLoad("No model weights found".to_string()));
                 // Actually, maybe we only grabbed config? 
                 // If we have config, we proceed? 
                 // But we need weights.
                 return Err(WorkerError::ModelLoad("No safetensors or bin weights found".to_string()));
            }
            
            // hf_hub stores files in its own cache structure.
            // But ModelLoader expects files in `self.config.cache_dir/model_name`.
            // hf_hub manages `cache_dir` internally (blobs/refs).
            // If we initialized Api with `with_cache_dir(config.cache_dir)`, then 
            // `repo.get()` returns path inside that cache.
            // But `ModelLoader` config loading assumes a specific structure:
            // `self.config.cache_dir.join(model_name).join("config.json")`.
            
            // This is mismatched assumption.
            // If we use hf_hub, we should rely on hf_hub to Give us paths.
            // And `get_model_path` should return the directory where config.json resides?
            // Actually, `repo.get()` returns absolute path to the file.
            // We can return the parent directory of config.json.
            
            return Ok(_config_path.parent().unwrap().to_path_buf());
        }
        
        // Fallback or error
        Err(WorkerError::ModelLoad("HF API not initialized".to_string()))
    }

    /// Load model configuration
    async fn load_model_config(
        &self,
        model_name: &str,
        provided_config: Option<&crate::cluster::ModelConfig>,
        model_path: &Path,
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
                rope_theta: 10000.0, // Default theta
                is_moe: false,
                num_experts: None,
                num_experts_per_tok: None,
            });
        }

        // Load from config file
        let config_path = model_path.join("config.json");

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

    /// Calculate memory usage for a model
    fn calculate_memory_usage(&self, config: &ModelConfig, quantization: crate::cluster::Quantization) -> usize {
        let num_params = config.vocab_size * config.hidden_size
            + config.num_layers * (
                config.hidden_size * config.hidden_size * 4 +  // Attention
                config.hidden_size * config.intermediate_size * 3  // MLP
            );

        // Add MoE parameters if applicable
        let num_params = if config.is_moe {
            if let Some(num_experts) = config.num_experts {
                num_params + config.num_layers * num_experts * config.intermediate_size * config.hidden_size * 2
            } else {
                num_params
            }
        } else {
            num_params
        };

        // Rough estimate: 2 bytes per param (FP16 default)
        // In production, vary by actual quantization from proto enum
        num_params * 2
    }

    /// Unload a model
    pub async fn unload_model(&self, model_name: &str) -> Result<(), WorkerError> {
        if let Some((_, _instance)) = self.loaded_models.remove(model_name) {
            // Free GPU memory
            self.gpu_manager.free_memory(&format!("model:{}", model_name)).await;
            info!("Model {} unloaded", model_name);
        }

        Ok(())
    }

    /// Get loaded model
    pub fn get_model(&self, model_name: &str) -> Option<ModelInstance> {
        self.loaded_models.get(model_name).map(|entry| entry.value().clone())
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
                "name": model_name,
                "gpu_ids": instance.gpu_ids(),
                "memory_used_mb": instance.memory_used() / 1024 / 1024,
                "inference_count": instance.inference_count(),
            })
        })
    }
}