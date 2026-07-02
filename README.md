# AI Cluster - Distributed Multi-GPU Inference System

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![ROCm](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://rocm.docs.amd.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/architecture.md)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Supported Models](#supported-models)
- [Performance](#performance)
- [Use Cases](#use-cases)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**AI Cluster** is a production-ready distributed system for running large language models (LLMs) across multiple GPUs and machines. It provides a unified API for model inference while automatically handling the complexities of distribution, parallelism, and resource management.

[**Why use AI Cluster? Read our Non-Technical Introduction.**](docs/problem_solution.md)

Whether you have a single workstation with multiple GPUs or a rack of servers, AI Cluster scales to meet your needs while maintaining low latency and high throughput.

### Why AI Cluster?

- **🚀 High Performance**: Optimized for both AMD and NVIDIA GPUs with custom kernels
- **🔧 Hardware Agnostic**: Same code runs on AMD, NVIDIA, or CPU
- **📈 Elastic Scaling**: Add or remove workers without downtime
- **🛡️ Production Ready**: Built-in monitoring, fault tolerance, and security
- **🎯 Easy to Use**: Simple REST API compatible with OpenAI's interface
- **💪 Powerful Parallelism**: Pipeline, tensor, data, and expert parallelism

---

## Features

### Core Features

| Feature | Description |
|---------|-------------|
| **Multi-GPU Support** | Run models across multiple AMD (ROCm) or NVIDIA (CUDA) GPUs via multiple workers |
| **Dynamic Model Loading** | Load/unload models at runtime without restart |
| **REST API** | OpenAI-compatible API for easy integration |
| **Streaming** | Stream tokens as they're generated |
| **Quantization** | Planned — FP16/INT8/INT4/FP8 (weights load as FP32 today; non-NONE requests are rejected) |

### Advanced Features

| Feature | Description |
|---------|-------------|
| **Circuit Breakers** | Prevents cascading failures |
| **Request Queuing** | Priority-based request handling |
| **Affinity Routing** | Session persistence for chatbots |
| **Prometheus Metrics** | Comprehensive monitoring |
| **Grafana Dashboards** | Pre-built visualizations |
| **Kubernetes Support** | Planned — no manifests ship yet |

### Roadmap & Planned Optimizations

**Status legend:**

| Badge | Meaning |
|-------|---------|
| ✅ Done (Completed) | Fully implemented and wired end-to-end |
| 🔶 Done (Partially) | Core logic implemented; integration or secondary features missing |
| 🔄 In-Progress | Actively being developed on the current branch |
| 🔲 To-Do | Not yet started |

| Feature | Status | Notes |
|---------|--------|-------|
| **KV Cache** | ✅ Done (Completed) | Per-layer `KvEntry<B>` / `KvCache<B>`; prefill + autoregressive decode fully wired in Llama; TP-sharded `TpKvCache<B>` also implemented |
| **Tensor Parallelism** | 🔶 Done (Partially) | `tensor_parallel_llama_prefill` and `tensor_parallel_llama_decode_step` implemented with Megatron-LM style column/row split and `AllReduce` abstraction; not yet wired to the gRPC service layer |
| **Pipeline Parallelism** | 🔶 Done (Partially) | `pipeline_parallel_llama_forward` implements layer-chunk partitioning; not yet wired to the gRPC service layer |
| **Expert Parallelism (MoE)** | 🔄 In-Progress | `DeepSeekMoE` sparse top-k routing implemented in `worker/models/deepseek.rs`; distributed expert routing across GPUs is a stub returning an error |
| **Data Parallelism** | ✅ Done (Completed) | Fully operational — deploy multiple independent workers (one per GPU) behind the coordinator's load balancer |
| **Continuous Batching** | 🔲 To-Do | High-throughput inference with dynamic batching |
| **Paged KV Cache** | 🔲 To-Do | vLLM-style memory-efficient paged attention; basic KV cache is done, the paged variant is not |
| **Speculative Decoding** | 🔲 To-Do | 2-3x generation speedup with draft model |
| **Flash Attention** | 🔲 To-Do | Fast and memory-efficient exact attention |
| **Distributed Tracing** | 🔲 To-Do | Jaeger integration for end-to-end request tracing |
| **Extended Infrastructure** | 🔲 To-Do | Redis (caching), MinIO (model storage), Vault (secrets), Elastic (logs) |

---

---

## Tech Stack

- **Coordinator**: Python (FastAPI, gRPC)
- **Worker**: Rust (Burn Framework, Tokio, Tonic)
- **Communication**: gRPC (Inter-service), REST API (Client-facing)
- **GPU Acceleration**:
    - **AMD**: ROCm (HIP)
    - **NVIDIA**: CUDA
- **Infrastructure**: Docker, Prometheus, Grafana
- **Planned Infrastructure**: Jaeger, Redis, MinIO, Vault, Elastic

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client Applications                       │
│                    (REST API, Web UI, CLI, SDKs)                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Coordinator (single instance)                    │
│        FastAPI REST · worker discovery · routing · registry         │
│     (HA replicas + leader election are on the roadmap, not built)   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
┌───────────────────────────────────┐  ┌───────────────────────────────────┐
│        Worker Pool (AMD)          │  │      Worker Pool (NVIDIA)         │
│  ┌─────────────┐ ┌─────────────┐  │  │  ┌─────────────┐ ┌─────────────┐  │
│  │ Worker AMD  │ │ Worker AMD  │  │  │  │Worker NVIDIA│ │Worker NVIDIA│  │
│  │   GPU 0-3   │ │   GPU 4-7   │  │  │  │   GPU 0-3   │ │   GPU 4-7   │  │
│  └─────────────┘ └─────────────┘  │  │  └─────────────┘ └─────────────┘  │
│                                   │  │                                   │
│  ┌─────────────┐ ┌─────────────┐  │  │  ┌─────────────┐ ┌─────────────┐  │
│  │ Worker AMD  │ │ Worker AMD  │  │  │  │Worker NVIDIA│ │Worker NVIDIA│  │
│  │   CPU Only  │ │  Mixture    │  │  │  │   CPU Only  │ │  Mixture    │  │
│  └─────────────┘ └─────────────┘  │  │  └─────────────┘ └─────────────┘  │
└───────────────────────────────────┘  └───────────────────────────────────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer (shipped today)                 │
│        ┌─────────────┐              ┌─────────────┐                  │
│        │  Prometheus │              │   Grafana   │                  │
│        └─────────────┘              └─────────────┘                  │
│  Planned: Redis · MinIO · Jaeger · Consul · Vault · Elastic          │
└─────────────────────────────────────────────────────────────────────┘
```

For detailed architecture information, see the [Architecture Guide](docs/architecture.md).

---

## Quick Start

### Method 1: Docker Compose

The easiest way to get started is using Docker Compose, which handles all dependencies automatically.

```bash
# 1. Clone the repository
git clone https://github.com/caestrada1103/ai-cluster.git
cd ai-cluster

# 2. Configure Environment (REQUIRED: compose refuses to start Grafana without
#    GRAFANA_ADMIN_PASSWORD; HF_TOKEN is needed for gated models like Llama 3)
cp .env.example .env
# Edit .env: set GRAFANA_ADMIN_PASSWORD=<something> and HF_TOKEN=hf_...

# 3. Build and start with Docker Compose
docker compose up -d --build

# 4. Check that everything is running
curl http://localhost:8000/health

# 5. Run your first inference (Model will auto-download and load)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-7b",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### Method 2: Local Execution (Native GPU)

If you prefer to run the cluster natively on your host machine to manually manage the GPU environment:

```bash
# 1. Clone the repository
git clone https://github.com/caestrada1103/ai-cluster.git
cd ai-cluster

# 2. Run the setup script for your GPU architecture
# For AMD GPUs
./scripts/setup_rocm.sh
# Or for NVIDIA GPUs
./scripts/setup_cuda.sh

# 3. Set up Python environment for Coordinator
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r coordinator/requirements.txt

# 4. Build and run the Rust Worker locally
# In a new terminal:
cd worker
# For AMD: cargo run --release --features rocm
# For NVIDIA: cargo run --release --features cuda
cargo run --release --features cuda

# 5. Start the Coordinator locally
# In the original python terminal:
# From the repo root (module path matters — cd coordinator breaks imports):
uvicorn coordinator.main:app --host 0.0.0.0 --port 8000

# 6. Run your first inference (Model will auto-download and load)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-7b",
    "prompt": "Explain quantum computing in simple terms",
    "max_tokens": 100
  }'
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Guide](docs/architecture.md) | System design, components, data flow |
| [API Reference](docs/api_reference.md) | Complete API documentation with examples |
| [Configuration Guide](docs/configuration.md) | All configuration options explained |
| [Deployment Guide](docs/deployment.md) | Single machine, cluster, Kubernetes, cloud |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |

### Quick Links

- [Installation Guide](#installation)
- [Configuration Examples](#configuration)
- [API Examples](#api-reference)
- [Performance Tuning](docs/configuration.md#performance-tuning)
- [Security Hardening](docs/deployment.md#security-hardening)

---

## Supported Models

AI Cluster supports a wide range of popular models:

| Model Family | Sizes | Status | Notes |
|--------------|-------|--------|-------|
| **Llama 3** | 8B, 70B | ✅ Implemented | reference architecture |
| **Qwen 2.5 Coder** | 32B | ✅ Implemented | GQA + RoPE + SwiGLU (Qwen3 not yet supported) |
| **DeepSeek (dense)** | 7B, 67B | ✅ Implemented | loads via the Llama-layout path |
| **DeepSeek (MoE/V3)** | 671B | 🔶 Model code present | V3 routing (sigmoid/MLA) not implemented |
| **Mistral** | 7B | 🔶 Model code present, not wired to loader | no TextGeneration/tokenizer/KV cache yet |
| **Mixtral** | 8x7B | 🔲 Planned | |
| **Gemma** | 2B, 7B | 🔲 Planned | |
| **Phi** | 2, 3-mini | 🔲 Planned | |

### Adding Custom Models

```python
# 1. Convert your model
python scripts/convert_model.py your-username/your-model --output ./models/

# 2. Add to model registry (config/models.toml)
[models."your-model"]
family = "custom"
parameters = "7B"
min_memory_gb = 16

[models."your-model".architecture]
num_layers = 32
hidden_size = 4096
# ... model architecture

# 3. Load and use
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model_name": "your-model"}'
```

---

## Performance

> **Disclaimer**: The benchmarks and metrics below represent **projected goals** leveraging our planned future optimizations (such as Continuous Batching, Paged KV Cache, and advanced Tensor Parallelism). Current single-worker baseline performance will vary depending on your hardware environment.

### Projected Benchmarks

| Model | GPUs | Batch Size | Projected Tokens/sec | Projected Latency (P95) |
|-------|------|------------|----------------------|-------------------------|
| DeepSeek-7B | 1x AMD 9060 XT | 1 | 45 | 120ms |
| DeepSeek-7B | 1x AMD 9060 XT | 8 | 210 | 380ms |
| DeepSeek-7B | 2x AMD 9060 XT | 16 | 410 | 420ms |
| DeepSeek-67B | 4x AMD 9060 XT | 1 | 12 | 450ms |
| Llama-3-8B | 1x NVIDIA T4 | 1 | 52 | 105ms |
| Llama-3-8B | 1x NVIDIA T4 | 8 | 245 | 350ms |
| Llama-3-70B | 4x NVIDIA A100 | 1 | 28 | 210ms |
| Mixtral-8x7B | 2x NVIDIA A100 | 1 | 18 | 320ms |

### Scaling Efficiency (Data Parallel Multi-Worker)

```
Throughput vs. Number of GPUs (DeepSeek-7B)
─────────────────────────────────────────────
4 GPUs ──────────────────▒ 410 tok/s (91%)
3 GPUs ────────────────▒ 320 tok/s (94%)
2 GPUs ─────────────▒ 210 tok/s (97%)
1 GPU ───────▒ 100 tok/s (100%)
    0    100   200   300   400   500
          Tokens per second
```

### Roadmap Optimizations Impact

Once fully implemented, these optimizations are projected to provide the following benefits for constrained hardware settings:
- **Continuous Batching**: 2-3x throughput improvement
- **Paged KV Cache**: 50-70% memory reduction
- **Flash Attention**: 2-4x faster attention
- **Speculative Decoding**: 2-3x faster generation
- **Quantization**: 75% memory reduction with INT8 (Currently available)

---

## Use Cases

### 1. **Chat Applications**

```python
import requests

messages = []
while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})
    r = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "messages": messages},
        timeout=300,
    )
    content = r.json()["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": content})
    print(f"Assistant: {content}")
```

### 2. **Scripted Batch Processing**

```python
import requests

prompts = ["Summarize: ...", "Translate: ..."]
results = [
    requests.post(
        "http://localhost:8000/v1/completions",
        json={"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": p, "max_tokens": 64},
        timeout=300,
    ).json()["text"]
    for p in prompts
]
```

> A dedicated `ai_cluster` client SDK (Python/JS) is planned — today the API is plain HTTP.

---

## Advanced Use Cases

### 1. **Multi-Agent Systems (Agent per GPU)**

You can assign specific models to specific workers to create specialized agents.
- **Worker 1 (AMD)**: Loads `llama3-8b` for Coding tasks.
- **Worker 2 (NVIDIA)**: Loads `qwen3-coder-32b` for Reasoning/Review tasks.

**Configuration**:
No special config needed. Just route your load requests:
```bash
# Load Coding Agent on AMD Worker
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3-8b", "worker_id": "<id from GET /v1/workers>"}'

# Load Reasoning Agent on NVIDIA Worker
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "qwen3-coder-32b", "worker_id": "<id from GET /v1/workers>"}'
```

### 2. **Running Large Models (Future Optimization)**

> **Note**: Tensor and Pipeline parallelism core algorithms are implemented in `worker/src/parallelism.rs` (functions `tensor_parallel_llama_prefill`, `tensor_parallel_llama_decode_step`, `pipeline_parallel_llama_forward`), but are not yet wired into the gRPC inference service. Currently, models must fit within the memory of a single GPU, or you can leverage Data Parallelism by deploying multiple identical model workers on the same machine.

When fully implemented, you will be able to configure model splitting like this:
1.  **Configure**: Ensure `config/models.toml` allows sufficient GPUs (Planned feature).
    ```toml
    [models.llama3-70b]
    max_gpus = 8
    parallelism.default = "tensor"
    ```
2.  **Load**:
    ```bash
    curl -X POST http://localhost:8000/v1/models/load \
      -H "Content-Type: application/json" \
      -d '{"model_name": "llama3-70b", "worker_id": "nvidia-gpu-0"}'
    ```
The system will automatically allocate 4 GPUs and use **Tensor Parallelism** (faster) or **Pipeline Parallelism** (inter-node) based on topology.

---

## Installation

### Prerequisites

- **OS**: Ubuntu 22.04+ (recommended), RHEL 9, Rocky Linux 9
- **Python**: 3.10+
- **Rust**: 1.70+
- **Docker**: 20.10+ (optional)
- **GPU Drivers**: ROCm 6.0+ (AMD) or CUDA 12.1+ (NVIDIA)

### Method 1: From Source

```bash
# Clone repository
git clone https://github.com/caestrada1103/ai-cluster.git
cd ai-cluster

# Set up Python environment
python -m venv venv
source venv/bin/activate
pip install -r coordinator/requirements.txt

# Build Rust worker
cd worker
cargo build --release --features=rocm  # For AMD
# or
cargo build --release --features=cuda  # For NVIDIA

# Return to root
cd ..

# Create model directory
mkdir -p models

# Start coordinator
# From the repo root (module path matters — cd coordinator breaks imports):
uvicorn coordinator.main:app --host 0.0.0.0 --port 8000

# In another terminal, start worker
cd worker
./target/release/ai-worker --port 50051 --gpu-ids 0
```

### Method 2: Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Method 3: Kubernetes (planned)

Kubernetes manifests are not shipped yet. Deploy with Docker Compose (Method 2)
or run the services natively. K8s manifests/Helm are tracked on the roadmap.

For detailed installation instructions, see the [Deployment Guide](docs/deployment.md).

---

## Configuration

### Coordinator Configuration (environment variables)

The coordinator is configured exclusively via `COORDINATOR_*` environment
variables (or `.env`). Key settings:

```bash
COORDINATOR_HOST=0.0.0.0
COORDINATOR_PORT=8000
COORDINATOR_DISCOVERY_METHOD=static          # static only today (mdns/consul planned)
COORDINATOR_STATIC_WORKERS=localhost:50051   # comma-separated host:port list
COORDINATOR_ROUTING_STRATEGY=least_load      # least_load|round_robin|random|affinity|power_of_two
```

See the [Configuration Guide](docs/configuration.md) for the full table.

### Minimal Worker Configuration

```toml
# config/worker.toml — FLAT schema (unknown keys are rejected)
grpc_port = 50051
metrics_port = 9091
gpu_ids = [0]
model_cache_dir = "./models"
```

For complete configuration options, see the [Configuration Guide](docs/configuration.md).

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/workers` | List workers |
| POST | `/v1/workers/manual` | Manually register worker addresses |
| GET | `/v1/models` | List models (OpenAI-style list) |
| POST | `/v1/models/load` | Load a model |
| DELETE | `/v1/models/{name}` | Unload a model |
| POST | `/v1/completions` | Generate text (buffered) |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (supports SSE streaming) |
| GET | `/metrics` | Prometheus metrics |

### Example: Text Completion

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-7b",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

Response:
```json
{
  "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "text": " in a faraway land, there lived a brave knight...",
  "tokens_generated": 50,
  "processing_time_ms": 2140.5,
  "worker_id": "worker-0"
}
```

For complete API documentation, see the [API Reference](docs/api_reference.md).

---

## Monitoring

### Prometheus Metrics

```bash
# Scrape metrics (Prometheus is on 9099, Coordinator on 8000)
curl http://localhost:8000/metrics

# Example metrics
# HELP coordinator_requests_total Total requests processed
# TYPE coordinator_requests_total counter
coordinator_requests_total{model="deepseek-7b"} 1250

# HELP worker_gpu_utilization_percent GPU utilization
# TYPE worker_gpu_utilization_percent gauge
worker_gpu_utilization_percent{worker="worker-gpu-0",gpu="0"} 75.2
```

### Grafana Dashboards

Provisioning ships in `monitoring/`: the Prometheus datasource plus an
**AI Cluster Overview** dashboard (request rate, P95 latency, GPU
utilization/temperature, coordinator backlog). Grafana: http://localhost:3000
(admin / $GRAFANA_ADMIN_PASSWORD).

### Distributed Tracing (planned)

Jaeger integration is on the roadmap — no tracing ships today.

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Clone your fork
git clone https://github.com/caestrada1103/ai-cluster.git
cd ai-cluster

# Set up pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests (unit suite lives under coordinator/)
pytest coordinator/
cd worker && cargo test --features wgpu
```

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- **Python**: Black with line length 100
- **Rust**: rustfmt with default settings
- **Documentation**: Markdown with linter

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2026 AI Cluster Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Acknowledgments

### Built With

- [Burn](https://burn.dev/) - Rust deep learning framework
- [ROCm](https://rocm.docs.amd.com) - AMD GPU computing platform
- [CUDA](https://developer.nvidia.com/cuda-toolkit) - NVIDIA GPU computing platform
- [FastAPI](https://fastapi.tiangolo.com/) - Python web framework
- [PyTorch](https://pytorch.org/) - For model conversion
- [HuggingFace](https://huggingface.co) - Model hub and tokenizers

### Inspired By

- [vLLM](https://vllm.readthedocs.io/) - PagedAttention and continuous batching
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - Tensor parallelism
- [DeepSpeed](https://www.deepspeed.ai/) - Pipeline parallelism
- [NVIDIA Dynamo](https://github.com/NVIDIA/dynamo) - Distributed inference

### Contributors

<a href="https://github.com/caestrada1103/ai-cluster/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=caestrada1103/ai-cluster" />
</a>

### Support

- 📚 [Documentation](docs/)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=caestrada1103/ai-cluster&type=Date)](https://star-history.com/#caestrada1103/ai-cluster&Date)

---

<div align="center">
  <sub>Built with ❤️ by the AI Cluster Team</sub>
  <br>
  <sub>© 2026 AI Cluster. All rights reserved.</sub>
</div>
