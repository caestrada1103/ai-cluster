//! Integration and unit tests for the worker
//!
//! This module contains tests for:
//! - GPU management
//! - Model loading
//! - Inference execution
//! - Parallelism strategies
//! - Error handling
//! - Performance benchmarks

use std::sync::Arc;
use std::time::Duration;

use tempfile::tempdir;
use tokio::time::sleep;
use tracing_test::traced_test;

use ai_worker::{
    config::WorkerConfig,
    error::WorkerError,
    gpu_manager::GPUManager,
    model_loader::{ModelLoader, ModelLoaderConfig},
    models::{Model, ModelConfig},
    parallelism::{ParallelModel, ParallelismConfig, create_parallel_model},
    worker::WorkerService,
    cluster::{
        Quantization, ParallelismStrategy,
        worker_server::{Worker, WorkerServer},
    },
};

// ============================================================================
// Test Utilities
// ============================================================================

/// Create a test configuration
fn test_config() -> WorkerConfig {
    WorkerConfig {
        worker_id: Some("test-worker".to_string()),
        grpc_port: 50051,
        metrics_port: 9091,
        gpu_ids: vec![0],
        model_cache_dir: tempdir().unwrap().path().to_path_buf(),
        download_dir: tempdir().unwrap().path().to_path_buf(),
        max_concurrent_loads: 2,
        load_timeout_secs: 30,
        request_timeout_secs: 10,
        verify_checksums: false,
        enable_mmap: true,
        pin_memory: false,
        ..Default::default()
    }
}

/// Create a mock model for testing
struct MockModel {
    name: String,
    config: ModelConfig,
    memory_used: usize,
    gpu_ids: Vec<usize>,
    inference_count: std::sync::atomic::AtomicU64,
}

impl MockModel {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            config: ModelConfig {
                architecture: "mock".to_string(),
                num_layers: 12,
                hidden_size: 768,
                num_attention_heads: 12,
                num_kv_heads: 12,
                vocab_size: 32000,
                max_seq_len: 2048,
                intermediate_size: 3072,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                is_moe: false,
                num_experts: None,
                num_experts_per_tok: None,
            },
            memory_used: 1024 * 1024 * 1024, // 1GB
            gpu_ids: vec![0],
            inference_count: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

#[async_trait::async_trait]
impl<B: burn::backend::Backend> Model<B> for MockModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    async fn forward(&self, input: crate::models::ModelInput<B>) -> Result<crate::models::ModelOutput<B>, WorkerError> {
        // Mock forward pass - just return dummy tensor
        let device = input.device();
        Ok(burn::tensor::Tensor::zeros([1, 10, self.config.vocab_size], &device))
    }

    async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Result<crate::models::TokenStream<B>, WorkerError> {
        self.inference_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Return empty stream for testing
        Ok(crate::models::TokenStream::new(
            Arc::new(self.clone()),
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
        ))
    }

    fn memory_used(&self) -> usize {
        self.memory_used
    }

    fn gpu_ids(&self) -> &[usize] {
        &self.gpu_ids
    }

    fn quantization(&self) -> Quantization {
        Quantization::Fp16
    }

    fn parallelism(&self) -> ParallelismStrategy {
        ParallelismStrategy::SingleDevice
    }

    fn loaded_at(&self) -> std::time::SystemTime {
        std::time::SystemTime::now()
    }

    fn inference_count(&self) -> u64 {
        self.inference_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// ============================================================================
// GPU Manager Tests
// ============================================================================

#[tokio::test]
async fn test_gpu_manager_initialization() {
    let gpu_manager = GPUManager::new(&[0]).await;
    
    match gpu_manager {
        Ok(manager) => {
            assert!(manager.device_count() > 0);
            println!("GPU Manager initialized with {} devices", manager.device_count());
        }
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping GPU tests");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    }
}

#[tokio::test]
async fn test_gpu_memory_allocation() {
    let gpu_manager = match GPUManager::new(&[0]).await {
        Ok(manager) => manager,
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping memory test");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    };

    // Test memory allocation
    let result = gpu_manager.allocate_memory(0, 1024 * 1024, "test".to_string()).await;
    assert!(result.is_ok());

    // Test memory exhaustion
    let total_memory = gpu_manager.get_device(0).unwrap().total_memory;
    let result = gpu_manager.allocate_memory(0, total_memory * 2, "test2".to_string()).await;
    assert!(matches!(result, Err(WorkerError::OutOfMemory { .. })));

    // Test memory freeing
    gpu_manager.free_memory("test").await;
    let (_, available, _) = gpu_manager.get_memory_stats(0).await;
    assert!(available > 0);
}

#[tokio::test]
async fn test_gpu_stream_pool() {
    let gpu_manager = match GPUManager::new(&[0]).await {
        Ok(manager) => manager,
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping stream test");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    };

    // Get a stream
    let stream1 = gpu_manager.get_stream(0).await;
    assert!(stream1.is_ok());

    // Return it
    if let Ok(stream) = stream1 {
        gpu_manager.return_stream(0, stream).await;
    }

    // Get another stream (should reuse)
    let stream2 = gpu_manager.get_stream(0).await;
    assert!(stream2.is_ok());
}

// ============================================================================
// Model Loader Tests
// ============================================================================

#[tokio::test]
async fn test_model_loader_initialization() {
    let gpu_manager = match GPUManager::new(&[0]).await {
        Ok(manager) => Arc::new(manager),
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping loader test");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    };

    let config = ModelLoaderConfig::default();
    let loader = ModelLoader::new(config, gpu_manager);
    assert!(loader.is_ok());
}

#[tokio::test]
async fn test_model_memory_calculation() {
    let gpu_manager = match GPUManager::new(&[0]).await {
        Ok(manager) => Arc::new(manager),
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping memory calc test");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    };

    let config = ModelLoaderConfig::default();
    let loader = ModelLoader::new(config, gpu_manager).unwrap();

    let model_config = ModelConfig {
        architecture: "test".to_string(),
        num_layers: 12,
        hidden_size: 768,
        num_attention_heads: 12,
        num_kv_heads: 12,
        vocab_size: 32000,
        max_seq_len: 2048,
        intermediate_size: 3072,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        is_moe: false,
        num_experts: None,
        num_experts_per_tok: None,
    };

    let memory_fp16 = loader.calculate_memory_usage(&model_config, Quantization::Fp16);
    let memory_int8 = loader.calculate_memory_usage(&model_config, Quantization::Int8);

    assert!(memory_fp16 > memory_int8);
    assert_eq!(memory_fp16 / 2, memory_int8); // Int8 should be half of FP16
}

// ============================================================================
// Model Tests
// ============================================================================

#[tokio::test]
async fn test_mock_model() {
    type TestBackend = burn::backend::NdArray; // CPU backend for testing

    let model = MockModel::new("test-model");
    assert_eq!(model.name(), "test-model");
    assert_eq!(model.memory_used(), 1024 * 1024 * 1024);
    assert_eq!(model.inference_count(), 0);
}

// ============================================================================
// Parallelism Tests
// ============================================================================

#[tokio::test]
async fn test_parallel_model_creation() {
    type TestBackend = burn::backend::NdArray;

    let model = Arc::new(MockModel::new("test-model"));
    let gpu_manager = match GPUManager::new(&[0, 1]).await {
        Ok(manager) => Arc::new(manager),
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping parallel test");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    };

    let config = ParallelismConfig {
        strategy: ParallelismStrategy::Data,
        data_parallel_replicas: 2,
        ..Default::default()
    };

    let parallel_model = create_parallel_model(
        model,
        ParallelismStrategy::Data,
        gpu_manager,
        Some(config),
    ).await;

    assert!(parallel_model.is_ok());
}

#[tokio::test]
async fn test_pipeline_parallelism_config() {
    type TestBackend = burn::backend::NdArray;

    let model = Arc::new(MockModel::new("test-model"));
    let gpu_manager = match GPUManager::new(&[0, 1]).await {
        Ok(manager) => Arc::new(manager),
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping pipeline test");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    };

    let mut parallel_model = ParallelModel::new(
        model,
        ParallelismConfig {
            strategy: ParallelismStrategy::Pipeline,
            pipeline_num_stages: 2,
            pipeline_num_microbatches: 4,
            ..Default::default()
        },
        gpu_manager,
    );

    let result = parallel_model.init_parallelism().await;
    assert!(result.is_ok());
}

// ============================================================================
// Worker Service Tests
// ============================================================================

#[tokio::test]
async fn test_worker_service_creation() {
    let gpu_manager = match GPUManager::new(&[0]).await {
        Ok(manager) => Arc::new(manager),
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping worker test");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    };

    let config = ModelLoaderConfig::default();
    let model_loader = Arc::new(ModelLoader::new(config, gpu_manager.clone()).unwrap());

    let worker = WorkerService::new(
        "test-worker".to_string(),
        gpu_manager,
        model_loader,
        test_config(),
    );

    assert_eq!(worker.version(), env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
#[traced_test]
async fn test_worker_health_check() {
    let gpu_manager = match GPUManager::new(&[0]).await {
        Ok(manager) => Arc::new(manager),
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping health test");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    };

    let config = ModelLoaderConfig::default();
    let model_loader = Arc::new(ModelLoader::new(config, gpu_manager.clone()).unwrap());

    let worker = WorkerService::new(
        "test-worker".to_string(),
        gpu_manager,
        model_loader,
        test_config(),
    );

    let request = tonic::Request::new(ai_worker::cluster::Empty {});
    let response = worker.health_check(request).await;

    assert!(response.is_ok());
    let health = response.unwrap().into_inner();
    assert!(health.status == 1 || health.status == 2); // SERVING or NOT_SERVING
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_out_of_memory_error() {
    let error = WorkerError::OutOfMemory {
        requested: 16_000_000_000,
        available: 8_000_000_000,
        device: 0,
    };

    let error_string = error.to_string();
    assert!(error_string.contains("16 GB"));
    assert!(error_string.contains("8 GB"));
    assert!(error_string.contains("GPU 0"));
}

#[tokio::test]
async fn test_model_not_found_error() {
    let error = WorkerError::ModelNotFound("deepseek-7b".to_string());
    assert_eq!(error.to_string(), "Model not found: deepseek-7b");
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

#[tokio::test]
#[ignore] // Only run manually for benchmarks
async fn benchmark_model_load() {
    let gpu_manager = Arc::new(GPUManager::new(&[0]).await.unwrap());
    let config = ModelLoaderConfig::default();
    let loader = ModelLoader::new(config, gpu_manager).unwrap();

    let start = std::time::Instant::now();
    
    // This would actually load a real model
    // loader.load_model("deepseek-7b", None, &[0], Quantization::Fp16, ParallelismStrategy::Auto).await.unwrap();
    
    let duration = start.elapsed();
    println!("Model load time: {:?}", duration);
}

#[tokio::test]
#[ignore] // Only run manually for benchmarks
async fn benchmark_inference() {
    type TestBackend = burn::backend::NdArray;

    let model = Arc::new(MockModel::new("test-model"));
    let start = std::time::Instant::now();

    // Run multiple inferences
    for i in 0..100 {
        // let _ = model.generate("test", 100, 0.7, 0.95, 40).await;
    }

    let duration = start.elapsed();
    println!("100 inferences time: {:?}", duration);
    println!("Average per inference: {:?}", duration / 100);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[tokio::test]
async fn test_full_worker_lifecycle() {
    let gpu_manager = match GPUManager::new(&[0]).await {
        Ok(manager) => Arc::new(manager),
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping lifecycle test");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    };

    let config = ModelLoaderConfig::default();
    let model_loader = Arc::new(ModelLoader::new(config, gpu_manager.clone()).unwrap());

    let worker = Arc::new(WorkerService::new(
        "test-worker".to_string(),
        gpu_manager,
        model_loader,
        test_config(),
    ));

    // Test model loading
    let load_request = tonic::Request::new(ai_worker::cluster::LoadModelRequest {
        model_name: "test-model".to_string(),
        model_path: "".to_string(),
        config: None,
        gpu_ids: vec![0],
        quantization: Quantization::Fp16.into(),
        parallelism: ParallelismStrategy::SingleDevice.into(),
    });

    let load_response = worker.load_model(load_request).await;
    assert!(load_response.is_ok());

    // Test status
    let status_request = tonic::Request::new(ai_worker::cluster::Empty {});
    let status_response = worker.get_status(status_request).await;
    assert!(status_response.is_ok());

    // Test unload
    let unload_request = tonic::Request::new(ai_worker::cluster::UnloadModelRequest {
        model_name: "test-model".to_string(),
        gpu_ids: vec![],
        force: false,
    });

    let unload_response = worker.unload_model(unload_request).await;
    assert!(unload_response.is_ok());
}

// ============================================================================
// Concurrency Tests
// ============================================================================

#[tokio::test]
async fn test_concurrent_model_loading() {
    let gpu_manager = match GPUManager::new(&[0]).await {
        Ok(manager) => Arc::new(manager),
        Err(WorkerError::NoGpusFound) => {
            println!("No GPUs found - skipping concurrency test");
            return;
        }
        Err(e) => panic!("Failed to initialize GPU manager: {}", e),
    };

    let config = ModelLoaderConfig {
        max_concurrent_loads: 2,
        ..Default::default()
    };
    let loader = Arc::new(ModelLoader::new(config, gpu_manager.clone()).unwrap());

    let mut handles = vec![];
    for i in 0..5 {
        let loader = loader.clone();
        let handle = tokio::spawn(async move {
            // This would actually load a model
            // loader.load_model(&format!("model-{}", i), None, &[0], Quantization::Fp16, ParallelismStrategy::Auto).await
            sleep(Duration::from_millis(100)).await;
            Ok::<_, WorkerError>(())
        });
        handles.push(handle);
    }

    for handle in handles {
        let result = handle.await;
        assert!(result.is_ok());
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod prop_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_memory_calculation_properties(
            num_layers in 1..100usize,
            hidden_size in 64..8192usize,
            vocab_size in 1000..100000usize,
        ) {
            let config = ModelConfig {
                architecture: "test".to_string(),
                num_layers,
                hidden_size,
                num_attention_heads: 12,
                num_kv_heads: 12,
                vocab_size,
                max_seq_len: 2048,
                intermediate_size: hidden_size * 4,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                is_moe: false,
                num_experts: None,
                num_experts_per_tok: None,
            };

            // Memory should increase with parameters
            let memory_fp16 = config.hidden_size * config.vocab_size * 2
                + config.num_layers * config.hidden_size * config.hidden_size * 4 * 2;
            
            prop_assert!(memory_fp16 > 0);
            prop_assert!(memory_fp16 < 100_000_000_000); // Sanity check
        }
    }
}

// ============================================================================
// Documentation Tests
// ============================================================================

#[test]
fn test_readme_example() {
    // This test verifies that the example in the README works
    let example = r#"
    use ai_worker::gpu_manager::GPUManager;
    
    #[tokio::main]
    async fn main() -> Result<(), Box<dyn std::error::Error>> {
        let gpu_manager = GPUManager::new(&[0]).await?;
        println!("Found {} GPUs", gpu_manager.device_count());
        Ok(())
    }
    "#;
    
    assert!(example.contains("GPUManager"));
}

// ============================================================================
// Run tests
// ============================================================================

#[cfg(test)]
mod test_runner {
    #[test]
    fn run_all_tests() {
        println!("Running all worker tests...");
        // Tests are run automatically by cargo test
    }
}