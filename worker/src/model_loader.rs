//! Model loading and management
//!
//! This module handles loading models from various formats,
//! managing model instances, and coordinating with the GPU manager.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::collections::HashMap;

use dashmap::DashMap;
use tokio::sync::Semaphore;
use tracing::info;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use safetensors::SafeTensors;
use burn::{
    tensor::Tensor,
    module::{Module, Param, ConstantRecord, ParamId},
    nn::{LinearRecord, EmbeddingRecord},
};
use half::f16;
use crate::error::WorkerError;
use crate::gpu_manager::GPUManager;
use crate::models::{
    ModelConfig, ModelInstance, TextGeneration,
    llama::{Llama, LlamaConfig, LlamaRecord, LlamaLayerRecord, LlamaAttentionRecord, LlamaMLPRecord},
    common::{RMSNormRecord, RotaryEmbeddingRecord, RotaryEmbedding},
};
use crate::backend::WorkerBackend;
use burn::backend::wgpu::WgpuDevice;

/// Model loader configuration
#[derive(Debug, Clone)]
pub struct ModelLoaderConfig {
    pub cache_dir: PathBuf,
    pub download_dir: PathBuf,
    pub max_concurrent_loads: usize,
    pub load_timeout_secs: u64,
    pub verify_checksums: bool,
    pub enable_mmap: bool,
    pub pin_memory: bool,
    pub prefetch_size_gb: f32,
}

impl Default for ModelLoaderConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from("./models"),
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
    config: ModelLoaderConfig,
    gpu_manager: Arc<GPUManager>,
    loaded_models: Arc<DashMap<String, ModelInstance>>,
    load_semaphore: Arc<Semaphore>,
    hf_api: Option<Api>,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(
        config: ModelLoaderConfig,
        gpu_manager: Arc<GPUManager>,
    ) -> Result<Self, WorkerError> {
        let mut builder = hf_hub::api::tokio::ApiBuilder::new()
            .with_endpoint("https://huggingface.co".to_string())
            .with_cache_dir(config.cache_dir.clone());
            
        if let Ok(token) = std::env::var("HF_TOKEN") {
             builder = builder.with_token(Some(token));
        }

        let hf_api = builder.build().ok();

        std::fs::create_dir_all(&config.cache_dir)?;
        std::fs::create_dir_all(&config.download_dir)?;

        Ok(Self {
            config: config.clone(),
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
        gpu_ids: &[u32],
        quantization: crate::cluster::Quantization,
        parallelism: crate::cluster::ParallelismStrategy,
    ) -> Result<ModelInstance, WorkerError> {
        if let Some(entry) = self.loaded_models.get(model_name) {
            info!("Model {} already loaded", model_name);
            return Ok(entry.value().clone());
        }

        let _permit = self.load_semaphore.acquire().await.map_err(|e| {
            WorkerError::Resource(format!("Failed to acquire load permit: {}", e))
        })?;

        info!("Loading model {}...", model_name);
        let model_path = self.get_model_path(model_name).await?;
        let config = self.load_model_config(model_name, model_config, &model_path).await?;
        
        let memory_used = self.calculate_memory_usage(&config, quantization);
        for &gpu_id in gpu_ids {
            self.gpu_manager.allocate_memory(
                gpu_id as usize,
                memory_used as u64,
                format!("model:{}", model_name),
            ).await?;
        }

        let device = WgpuDevice::BestAvailable;

        // Load weights
        let mut weights = self.load_safetensors(&model_name, &device).await?;

        // Get model directory (tokenizer.json lives here)
        let model_path = self.get_model_path(&model_name).await?;
        // Ensure tokenizer.json is downloaded
        if let Some(api) = &self.hf_api {
            let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));
            let _ = repo.get("tokenizer.json").await
                .map_err(|e| WorkerError::ModelLoad(format!("Failed to download tokenizer.json: {}", e)))?;
            info!("Tokenizer downloaded to: {:?}", model_path);
        }

        // Instantiate model
        let model: Arc<Mutex<dyn TextGeneration + Send>> = match config.architecture.as_str() {
            "llama" => {
                let llama_config = LlamaConfig {
                    hidden_size: config.hidden_size,
                    num_layers: config.num_layers,
                    num_attention_heads: config.num_attention_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.hidden_size / config.num_attention_heads,
                    intermediate_size: config.intermediate_size,
                    vocab_size: config.vocab_size,
                    max_seq_len: config.max_seq_len,
                    rms_norm_eps: config.rms_norm_eps,
                    rope_theta: config.rope_theta,
                };
                
                info!("Mapping weights to LlamaRecord...");
                let record = create_llama_record(&mut weights, &llama_config, &device)?;
                info!("Record created. Initializing Llama...");
                
                let model = Llama::new(&llama_config, &device, &model_path).load_record(record);
                Arc::new(Mutex::new(model))
            }
            _ => return Err(WorkerError::ModelLoad(format!("Unsupported architecture: {}", config.architecture))),
        };

        let instance = ModelInstance::new(
            model_name.to_string(),
            memory_used,
            gpu_ids.to_vec(),
            quantization as i32,
            parallelism as i32,
            Some(model),
        );

        self.loaded_models.insert(model_name.to_string(), instance.clone());
        info!("Model {} loaded successfully", model_name);

        Ok(instance)
    }

    async fn load_safetensors(&self, model_name: &str, device: &WgpuDevice) -> Result<HashMap<String, Tensor<WorkerBackend, 1>>, WorkerError> {
        let api = self.hf_api.as_ref().ok_or(WorkerError::ModelLoad("HF API not initialized".to_string()))?;
        let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));
        
        let mut weights = HashMap::new();
        let mut files = Vec::new();

        // Check for index (sharded) or single file
        if let Ok(index_path) = repo.get("model.safetensors.index.json").await {
            let index_content = std::fs::read_to_string(index_path)?;
            let json: serde_json::Value = serde_json::from_str(&index_content)
                .map_err(|e| WorkerError::ModelLoad(format!("Json error: {}", e)))?;
            
            if let Some(map) = json["weight_map"].as_object() {
                let mut filenames: Vec<String> = map.values().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                filenames.sort();
                filenames.dedup();
                for fname in filenames {
                    files.push(repo.get(&fname).await.map_err(|e| WorkerError::ModelLoad(e.to_string()))?);
                }
            }
        } else if let Ok(path) = repo.get("model.safetensors").await {
            files.push(path);
        } else {
             return Err(WorkerError::ModelLoad("No safetensors found".to_string()));
        }

        info!("Loading {} safetensors files...", files.len());

        for file in files {
            let data = std::fs::read(file).map_err(|e| WorkerError::ModelLoad(e.to_string()))?;
            let safetensors = SafeTensors::deserialize(&data).map_err(|e| WorkerError::ModelLoad(e.to_string()))?;
            
            for (name, view) in safetensors.tensors() {
                // Convert to f32
                let floats: Vec<f32> = match view.dtype() {
                    safetensors::Dtype::F16 => {
                        view.data().chunks(2)
                            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                            .collect()
                    },
                    safetensors::Dtype::BF16 => {
                        view.data().chunks(2)
                            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                            .collect()
                    },
                     safetensors::Dtype::F32 => {
                        view.data().chunks(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect()
                    },
                    _ => continue, // Skip unused types
                };
                
                let tensor = Tensor::<WorkerBackend, 1>::from_floats(
                    floats.as_slice(),
                    device,
                );
                weights.insert(name.to_string(), tensor);
            }
        }
        
        Ok(weights)
    }

    async fn get_model_path(&self, model_name: &str) -> Result<PathBuf, WorkerError> {
        if let Some(api) = &self.hf_api {
             let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));
             let config = repo.get("config.json").await.map_err(|e| WorkerError::ModelLoad(e.to_string()))?;
             return Ok(config.parent().unwrap().to_path_buf());
        }
        Err(WorkerError::ModelLoad("No HF API".to_string()))
    }

    async fn load_model_config(&self, _name: &str, _provided: Option<&crate::cluster::ModelConfig>, path: &Path) -> Result<ModelConfig, WorkerError> {
        let config_path = path.join("config.json");
        let s = std::fs::read_to_string(config_path)?;
        let json: serde_json::Value = serde_json::from_str(&s)?;
        let arch = json["architectures"][0].as_str().unwrap_or("llama").to_lowercase();
        Ok(ModelConfig {
            architecture: if arch.contains("llama") { "llama".to_string() } else { arch },
            num_layers: json["num_hidden_layers"].as_u64().unwrap_or(0) as usize,
            hidden_size: json["hidden_size"].as_u64().unwrap_or(0) as usize,
            num_attention_heads: json["num_attention_heads"].as_u64().unwrap_or(0) as usize,
            num_kv_heads: json["num_key_value_heads"].as_u64().unwrap_or(0) as usize,
            vocab_size: json["vocab_size"].as_u64().unwrap_or(32000) as usize,
            max_seq_len: json["max_position_embeddings"].as_u64().unwrap_or(2048) as usize,
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(0) as usize,
            rms_norm_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta: json["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            is_moe: false, num_experts: None, num_experts_per_tok: None,
        })
    }

    fn calculate_memory_usage(&self, config: &ModelConfig, _q: crate::cluster::Quantization) -> usize {
        let embed = config.vocab_size * config.hidden_size;
        let attn = config.num_layers * 4 * config.hidden_size * config.hidden_size;
        let ffn = config.num_layers * 3 * config.hidden_size * config.intermediate_size;
        let norm = (config.num_layers * 2 + 1) * config.hidden_size;
        
        // Return estimation in bytes (FP16 = 2 bytes per param)
        (embed + attn + ffn + norm) * 2
    }
    
    pub async fn unload_model(&self, model_name: &str) -> Result<(), WorkerError> {
        self.loaded_models.remove(model_name);
        Ok(())
    }
    
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
    
    pub fn list_models(&self) -> Vec<String> {
        self.loaded_models.iter().map(|entry| entry.key().clone()).collect()
    }
    pub fn get_model(&self, model_name: &str) -> Option<ModelInstance> {
        self.loaded_models.get(model_name).map(|entry| entry.value().clone())
    }
}

/// Helper to transpose Linear weights (HF [out, in] -> Burn [in, out])
fn load_linear(
    weights: &mut HashMap<String, Tensor<WorkerBackend, 1>>,
    name: &str,
    in_features: usize,
    out_features: usize,
    bias: bool
) -> Result<LinearRecord<WorkerBackend>, WorkerError> {
    let w_name = format!("{}.weight", name);
    let w_flat = weights.remove(&w_name).ok_or(WorkerError::ModelLoad(format!("Missing {}", w_name)))?;
    
    // HF: [out, in]
    // Burn expects: [in, out]
    // So we reshape to HF shape, transpose, then into Burn shape.
    // Actually, simply reshaping to [out, in] and transposing gives [in, out].
    let w = w_flat.reshape([out_features, in_features]).transpose();
    
    let b = if bias {
        let b_name = format!("{}.bias", name);
        if let Some(b_flat) = weights.remove(&b_name) {
            Some(b_flat) // Bias is [out], Burn expects [out]
        } else {
            None
        }
    } else {
        None
    };
    
    Ok(LinearRecord { weight: Param::initialized(ParamId::new(), w), bias: b.map(|t| Param::initialized(ParamId::new(), t)) })
}

fn load_embedding(
    weights: &mut HashMap<String, Tensor<WorkerBackend, 1>>,
    name: &str,
    num_embeddings: usize,
    embedding_dim: usize,
) -> Result<EmbeddingRecord<WorkerBackend>, WorkerError> {
    let w_name = format!("{}.weight", name);
    let w_flat = weights.remove(&w_name).ok_or(WorkerError::ModelLoad(format!("Missing {}", w_name)))?;
    let w = w_flat.reshape([num_embeddings, embedding_dim]);
    Ok(EmbeddingRecord { weight: Param::initialized(ParamId::new(), w) })
}

fn load_norm(
    weights: &mut HashMap<String, Tensor<WorkerBackend, 1>>,
    name: &str,
    _dim: usize,
) -> Result<RMSNormRecord<WorkerBackend>, WorkerError> {
    let w_name = format!("{}.weight", name);
    let w_flat = weights.remove(&w_name).ok_or(WorkerError::ModelLoad(format!("Missing {}", w_name)))?;
    let w = w_flat; // 1D
    Ok(RMSNormRecord { weight: Param::initialized(ParamId::new(), w), eps: ConstantRecord })
    // Check common.rs: RMSNorm has `eps: f64`. `#[module(skip)]`.
    // So record does NOT have eps.
    // Wait, check `RMSNorm` struct again.
    // Step 1439: `pub eps: f64`. `#[module(skip)]`.
    // So `RMSNormRecord` only has `weight`.
}

fn create_llama_record(
    weights: &mut HashMap<String, Tensor<WorkerBackend, 1>>, 
    config: &LlamaConfig, 
    device: &WgpuDevice
) -> Result<LlamaRecord<WorkerBackend>, WorkerError> {
    let embed = load_embedding(weights, "model.embed_tokens", config.vocab_size, config.hidden_size)?;
    let norm = load_norm(weights, "model.norm", config.hidden_size)?;
    let lm_head = load_linear(weights, "lm_head", config.hidden_size, config.vocab_size, false)?;
    
    let mut layers = Vec::new();
    for i in 0..config.num_layers {
        let prefix = format!("model.layers.{}", i);
        
        let q = load_linear(weights, &format!("{}.self_attn.q_proj", prefix), config.hidden_size, config.num_attention_heads * config.head_dim, false)?;
        let k = load_linear(weights, &format!("{}.self_attn.k_proj", prefix), config.hidden_size, config.num_kv_heads * config.head_dim, false)?;
        let v = load_linear(weights, &format!("{}.self_attn.v_proj", prefix), config.hidden_size, config.num_kv_heads * config.head_dim, false)?;
        let o = load_linear(weights, &format!("{}.self_attn.o_proj", prefix), config.num_attention_heads * config.head_dim, config.hidden_size, false)?;
        
        // LlamaAttentionRecord
        // Attention has q,k,v,o
        let attention = LlamaAttentionRecord {
            q_proj: q, k_proj: k, v_proj: v, o_proj: o,
            num_heads: ConstantRecord, num_kv_heads: ConstantRecord, head_dim: ConstantRecord,
        };
        
        let gate = load_linear(weights, &format!("{}.mlp.gate_proj", prefix), config.hidden_size, config.intermediate_size, false)?;
        let up = load_linear(weights, &format!("{}.mlp.up_proj", prefix), config.hidden_size, config.intermediate_size, false)?;
        let down = load_linear(weights, &format!("{}.mlp.down_proj", prefix), config.intermediate_size, config.hidden_size, false)?;
        
        let mlp = LlamaMLPRecord {
            gate_proj: gate, up_proj: up, down_proj: down,
        };
        
        let in_norm = load_norm(weights, &format!("{}.input_layernorm", prefix), config.hidden_size)?;
        let post_norm = load_norm(weights, &format!("{}.post_attention_layernorm", prefix), config.hidden_size)?;
        
        layers.push(LlamaLayerRecord {
            attention, mlp, input_layernorm: in_norm, post_attention_layernorm: post_norm,
        });
    }

    // Generate RoPE
    let _rope_mod: RotaryEmbedding<WorkerBackend> = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_theta, device);
    let rope = RotaryEmbeddingRecord {
        cos: ConstantRecord, sin: ConstantRecord,
    };

    Ok(LlamaRecord {
        embed_tokens: embed,
        layers,
        norm,
        lm_head,
        config: ConstantRecord,
        rope,
        tokenizer: ConstantRecord,
        device: ConstantRecord,
    })
}
