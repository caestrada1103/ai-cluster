# 🚀 Complete Project Generation Guide

## Build Your Own Local AI Cluster with Python + Rust

Use this markdown as prompts for an AI assistant (like Cursor, GitHub Copilot, or ChatGPT) to generate your entire project files.

---

# 📁 Project Structure to Generate

```
ai-cluster/
├── Cargo.toml                 # Rust workspace config
├── pyproject.toml             # Python project config
├── docker-compose.yml         # Multi-container setup
├── README.md                  # Project documentation
│
├── coordinator/               # Python orchestration layer
│   ├── __init__.py
│   ├── main.py                # FastAPI server
│   ├── coordinator.py         # Core coordinator logic
│   ├── models.py              # Model registry
│   ├── discovery.py           # Worker discovery
│   ├── router.py              # Request routing
│   └── requirements.txt       # Python dependencies
│
├── worker/                    # Rust worker implementation
│   ├── Cargo.toml             # Worker dependencies
│   ├── build.rs                # Protobuf compilation
│   ├── src/
│   │   ├── main.rs             # Entry point
│   │   ├── worker.rs            # gRPC service
│   │   ├── model_loader.rs      # Load models
│   │   ├── models/
│   │   │   ├── mod.rs
│   │   │   ├── deepseek.rs      # DeepSeek architecture
│   │   │   └── llama.rs         # Llama architecture
│   │   ├── gpu_manager.rs       # Multi-GPU support
│   │   ├── parallelism.rs       # Model parallelism
│   │   └── metrics.rs           # Prometheus metrics
│   └── examples/
│       └── simple_inference.rs  # Test example
│
├── proto/                      # gRPC definitions
│   └── cluster.proto
│
├── scripts/                     # Utility scripts
│   ├── convert_model.py         # HF to Burn converter
│   ├── benchmark.py             # Load testing
│   └── setup_rocm.sh            # ROCm setup helper
│
├── config/                      # Configuration
│   ├── coordinator.yaml
│   ├── worker.toml
│   └── models.toml
│
├── tests/                       # Integration tests
│   ├── test_coordinator.py
│   └── test_worker.rs
│
└── docs/                        # Documentation
    ├── architecture.md
    └── api_reference.md
```

---

# 📝 Prompt 1: Project Initialization

Copy and paste this prompt to your AI assistant:

```
Generate the complete project initialization files for an AI cluster system using Python (coordinator) and Rust (workers) with Burn framework. Include:

1. Top-level README.md explaining the project
2. Cargo.toml for Rust workspace
3. pyproject.toml for Python project
4. docker-compose.yml for containerization
5. Basic .gitignore

The system should support:
- Multi-GPU inference on AMD (ROCm) and NVIDIA (CUDA)
- Model parallelism (pipeline, tensor, data)
- Dynamic model loading
- gRPC communication
- Prometheus metrics

Make the files production-ready with proper error handling and documentation.
```

---

# 📝 Prompt 2: Protocol Buffer Definition

```
Generate the protobuf definition file at proto/cluster.proto for communication between Python coordinator and Rust workers.

Include:
- Worker service with LoadModel, Infer, GetStatus, UnloadModel
- Messages for model config, inference requests, GPU info
- Health check and streaming support
- Error codes and status reporting

Make it efficient for AI inference workloads.
```

---

# 📝 Prompt 3: Python Coordinator Core

```
Generate the complete Python coordinator implementation in coordinator/:

Files to create:
1. coordinator/__init__.py
2. coordinator/main.py - FastAPI server with endpoints:
   - POST /v1/completions
   - GET /v1/models
   - POST /v1/models/load
   - GET /v1/health

3. coordinator/coordinator.py - Core coordinator class with:
   - Worker discovery and registration
   - Health monitoring
   - Request routing
   - Load balancing
   - Fault tolerance (circuit breakers)

4. coordinator/models.py - Model registry with configs for:
   - DeepSeek (7B, 67B)
   - Llama 3 (8B, 70B)
   - Mistral (7B)
   Include memory requirements, quantization options, GPU counts

5. coordinator/discovery.py - Worker discovery via:
   - mDNS/Bonjour
   - Static config file
   - Broadcast discovery

6. coordinator/router.py - Request routing logic with:
   - Model-aware routing
   - Memory-aware scheduling
   - Multi-GPU task splitting

7. coordinator/requirements.txt - All Python dependencies

Use async/await, proper error handling, logging, and type hints.
```

---

# 📝 Prompt 4: Rust Worker Core

```
Generate the Rust worker implementation in worker/ using Burn framework:

Files to create:
1. worker/Cargo.toml - Dependencies including:
   - burn with "hip", "cuda", "train" features
   - tokio, tonic for gRPC
   - prometheus for metrics
   - tracing for logging

2. worker/build.rs - Protobuf compilation

3. worker/src/main.rs - Entry point with:
   - CLI argument parsing (--gpu-ids, --port, --config)
   - GPU device detection
   - Metrics server startup
   - gRPC server initialization

4. worker/src/worker.rs - gRPC service implementation:
   - LoadModel - load model onto specific GPUs
   - Infer - run inference with batching
   - GetStatus - report GPU memory, utilization
   - UnloadModel - free resources

5. worker/src/gpu_manager.rs - GPU management:
   - Device enumeration
   - Memory tracking
   - Stream management
   - Peer-to-peer access detection

6. worker/src/metrics.rs - Prometheus metrics:
   - Request counters, durations
   - GPU memory/usage
   - Batch sizes, token throughput

Use proper error handling (anyhow/thiserror), async patterns, and comprehensive logging.
```

---

# 📝 Prompt 5: Model Architecture Implementations

```
Generate the model implementations in worker/src/models/ using Burn:

Files:
1. worker/src/models/mod.rs - Export models

2. worker/src/models/deepseek.rs - DeepSeek architecture:
   - Mixture of Experts layers
   - Multi-head attention
   - Load balancing implementation
   - Forward pass with MoE routing

3. worker/src/models/llama.rs - Llama architecture:
   - RMSNorm
   - SwiGLU activation
   - Rotary embeddings
   - KV cache support

4. worker/src/model_loader.rs - Model loading:
   - HuggingFace weight conversion
   - Safetensors support
   - Burn record loading
   - Dynamic model type dispatch
   - Quantization (FP16, INT8, INT4)

5. worker/src/parallelism.rs - Parallel strategies:
   - Pipeline parallelism (layer sharding)
   - Tensor parallelism (row/column)
   - Data parallelism with all-reduce
   - Expert parallelism for MoE models
   - Automatic partitioning based on memory

Make each implementation production-ready with benchmarks and tests.
```

---

# 📝 Prompt 6: Performance Optimization

```
Generate the performance optimization modules in worker/src/:

Files:
1. worker/src/batching.rs - Continuous batching:
   - Dynamic batch formation
   - Timeout-based collection
   - Padding and attention masks
   - KV cache management

2. worker/src/kv_cache.rs - Paged attention cache:
   - Block-based memory allocation
   - Request-specific block tables
   - Cache eviction policies
   - Shared prefix caching

3. worker/src/kernels/ - CubeCL kernels:
   - flash_attention.rs - Flash attention implementation
   - moe_kernels.rs - Expert routing
   - rms_norm.rs - Layer normalization

Include benchmarks comparing to baseline implementations.
```

---

# 📝 Prompt 7: Configuration Files

```
Generate the configuration files for the cluster:

1. config/coordinator.yaml:
   - Server host/port
   - Discovery method (mdns/static)
   - Health check intervals
   - Circuit breaker settings
   - Model auto-loading rules

2. config/worker.toml:
   - GPU IDs to use
   - GRPC port
   - Metrics port
   - Batch size limits
   - Cache sizes
   - Logging levels

3. config/models.toml:
   - Model paths
   - Default quantization
   - GPU assignment
   - Warm-up settings
   - Version pinning

Make them well-documented with comments and examples.
```

---

# 📝 Prompt 8: Utility Scripts

```
Generate utility scripts in scripts/:

1. scripts/convert_model.py - Convert HF models:
   - Download from HuggingFace
   - Convert weights to Burn format
   - Save as safetensors/MPK
   - Generate model config
   - Command-line interface

2. scripts/benchmark.py - Load testing:
   - Concurrent request generation
   - Latency percentiles (P50, P95, P99)
   - Throughput measurement
   - GPU utilization tracking
   - Report generation

3. scripts/setup_rocm.sh - AMD GPU setup:
   - ROCm installation
   - Driver verification
   - Permission setup
   - Environment variables
   - Test script

Make them cross-platform where possible.
```

---

# 📝 Prompt 9: Docker Configuration

```
Generate Docker configuration for containerized deployment:

1. docker/Dockerfile.coordinator:
   - Python 3.10 slim base
   - Install dependencies
   - Copy coordinator code
   - Expose API port
   - Health check

2. docker/Dockerfile.worker:
   - Multi-stage build with Rust
   - ROCm base image for AMD
   - CUDA base image for NVIDIA (optional)
   - Copy compiled binary
   - GPU device passthrough

3. docker-compose.yml:
   - Coordinator service
   - Multiple worker services (AMD/NVIDIA)
   - Network configuration
   - Volume mounts for models
   - Environment variables

4. .dockerignore - Exclude unnecessary files
```

---

# 📝 Prompt 10: Tests and Documentation

```
Generate tests and documentation:

1. tests/test_coordinator.py:
   - Unit tests for routing
   - Mock worker tests
   - API endpoint tests
   - Failure recovery tests

2. tests/test_worker.rs:
   - Model loading tests
   - Inference correctness
   - Multi-GPU tests
   - Performance benchmarks

3. docs/architecture.md:
   - System overview diagram
   - Component interactions
   - Data flow
   - Deployment options

4. docs/api_reference.md:
   - All endpoints with examples
   - Request/response formats
   - Error codes
   - Rate limiting

5. examples/ in Rust worker:
   - simple_inference.rs - Basic usage
   - multi_gpu.rs - Using multiple GPUs
   - custom_model.rs - Adding new models
```

---

# 📝 Prompt 11: Integration and Examples

```
Generate integration examples and quick-start guide:

1. examples/chat_demo.py - Web chat interface:
   - Streamlit or Gradio UI
   - Model selection dropdown
   - Multi-GPU status display
   - Generation parameters

2. examples/batch_processor.py - Batch inference:
   - CSV/JSON input
   - Parallel processing
   - Result aggregation
   - Progress tracking

3. QUICKSTART.md:
   - Prerequisites
   - Installation steps
   - Configuration
   - Running first inference
   - Adding more GPUs
   - Troubleshooting

4. Makefile:
   - build-coordinator
   - build-worker
   - run-tests
   - benchmark
   - clean
```

---

# 📝 Prompt 12: Complete Integration Test

```
Generate an end-to-end test that validates the entire system:

Create tests/e2e_test.py that:
1. Starts a mock worker or real worker process
2. Starts the coordinator
3. Loads a small test model (e.g., TinyLlama)
4. Runs inference requests
5. Tests multi-GPU if available
6. Tests failure recovery
7. Measures performance
8. Cleans up processes

Include both success cases and error cases.
```

---

# 🚀 How to Use These Prompts

## Method 1: With an AI Code Assistant (Recommended)

1. **Start a new conversation** with your AI assistant
2. **Copy each prompt** one by one in order
3. **Create the files** as the AI generates them
4. **Review and test** each component
5. **Ask for refinements** if needed

## Method 2: Batch Generation

If your AI supports long contexts, you can combine prompts:

```
Generate a complete AI cluster project with:
[Paste prompts 1-3 together]
Then ask for the next set...
```

## Method 3: Iterative Development

For each component, you can ask:

```
Generate [specific file] for my AI cluster project. It should:
- Handle [specific requirements]
- Include error handling for [edge cases]
- Be optimized for [AMD GPUs]
- Follow best practices for [Rust/Python]
```

---

# 💡 Tips for Best Results

1. **Be specific** - If a generated file doesn't meet your needs, ask: "Can you modify the worker to also support X?"

2. **Ask for explanations** - "Explain how the tensor parallelism in the generated code works"

3. **Request alternatives** - "Show me a simpler version of the batching module first, then we can optimize"

4. **Validate assumptions** - "Will this work with my AMD Radeon 9060 XT?"

5. **Iterate** - Start with minimal versions, then add features

---

# 🎯 Your First Session Prompt

To get started immediately, copy this:

```
I want to build a local AI cluster using Python + Rust that can run DeepSeek models across multiple AMD GPUs (I have a Radeon 9060 XT). Please generate:

1. The project structure and initialization files
2. A basic Python coordinator that can discover workers
3. A minimal Rust worker that can load a model on my AMD GPU
4. Instructions for testing it on my system

Start with the minimal working version - we'll add features later.
```