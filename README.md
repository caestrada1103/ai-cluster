# AI Cluster - Distributed Multi-GPU Inference System

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![ROCm](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://rocm.docs.amd.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/architecture.md)
[![Discord](https://img.shields.io/badge/chat-discord-7289da.svg)](https://discord.gg/ai-cluster)

<div align="center">
  <img src="docs/images/ai-cluster-logo.png" alt="AI Cluster Logo" width="200"/>
  <p><strong>Run large language models across multiple GPUs with ease</strong></p>
</div>

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
| **Quantization** | FP16, INT8, INT4, FP8 support |

### Advanced Features

| Feature | Description |
|---------|-------------|
| **Circuit Breakers** | Prevents cascading failures |
| **Request Queuing** | Priority-based request handling |
| **Affinity Routing** | Session persistence for chatbots |
| **Prometheus Metrics** | Comprehensive monitoring |
| **Grafana Dashboards** | Pre-built visualizations |
| **Kubernetes Support** | Helm charts and operators |

### Roadmap & Planned Optimizations

The following features are under active development to improve performance, particularly for limited hardware environments or constrained networks:

| Feature | Description |
|---------|-------------|
| **Advanced Parallelism** | Tensor and Pipeline parallelism within a single worker (currently supports Data Parallelism via multi-worker scaling) |
| **Continuous Batching** | High-throughput inference with dynamic batching |
| **Paged KV Cache** | Memory-efficient attention cache for long contexts |
| **Speculative Decoding** | 2-3x speedup for generation |
| **Flash Attention** | Fast and memory-efficient exact attention |
| **Distributed Tracing** | Jaeger integration for debugging and tracing requests |
| **Extended Infrastructure** | Redis (caching), MinIO (model storage), Vault (secrets), Elastic (logs) |

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
│                        Coordinator Cluster                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │   Coordinator   │  │   Coordinator   │  │   Coordinator   │      │
│  │     Primary     │──│    Replica 1    │──│    Replica 2    │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
│                         (Leader Election)                           │
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
│                       Infrastructure Layer                          │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│   │   Prometheus│ │    Grafana  │ │    Redis    │ │    MinIO    │   │
│   │   Metrics   │ │  Dashboards │ │    Cache    │ │Model Storage│   │
│   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│                                                                     │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│   │    Jaeger   │ │    Consul   │ │    Vault    │ │   Elastic   │   │
│   │   Tracing   │ │   Discovery │ │   Secrets   │ │    Logs     │   │
│   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
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

# 2. Configure Environment (Optional but Recommended)
# Create a .env file to store secrets like your Hugging Face Token (required for Llama 3)
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...

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
# For AMD: cargo run --release --features hip
# For NVIDIA: cargo run --release --features cuda
cargo run --release --features cuda

# 5. Start the Coordinator locally
# In the original python terminal:
cd coordinator
uvicorn main:app --host 0.0.0.0 --port 8000

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

| Model Family | Sizes | Parallelism Support | Quantization |
|--------------|-------|---------------------|--------------|
| **DeepSeek** | 7B, 67B | Single Worker / Data Parallel | FP16, INT8, INT4 |
| **Llama 3** | 8B, 70B | Single Worker / Data Parallel | FP16, INT8, INT4 |
| **Mistral** | 7B | Single Worker / Data Parallel | FP16, INT8 |
| **Mixtral** | 8x7B | Single Worker / Data Parallel | INT8, INT4 |
| **Gemma** | 2B, 7B | Single Worker / Data Parallel | FP16, INT8 |
| **Phi** | 2, 3-mini | Single Worker / Data Parallel | FP16 |
| **Qwen** | 7B, 14B | Single Worker / Data Parallel | FP16, INT8 |

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
from ai_cluster import Client

client = Client("http://localhost:8000", api_key="sk-...")

# Streaming chat
messages = []
while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat_completion(
        model="llama3-8b",
        messages=messages,
        stream=True
    )
    
    for chunk in response:
        print(chunk.content, end="", flush=True)
    print()
```

### 2. **Batch Processing**

```python
import pandas as pd
from ai_cluster import Client

client = Client("http://localhost:8000")

# Load dataset
df = pd.read_csv("reviews.csv")

# Process in batch
results = client.batch_complete(
    model="deepseek-7b",
    prompts=df["text"].tolist(),
    max_tokens=50
)

df["summary"] = [r.text for r in results]
df.to_csv("reviews_processed.csv")
```

### 3. **RAG Pipeline**

```python
from ai_cluster import Client
from sentence_transformers import SentenceTransformer

client = Client("http://localhost:8000")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Retrieve relevant documents
query = "What is machine learning?"
query_embedding = encoder.encode(query)
docs = vector_db.search(query_embedding, k=5)

# Generate answer with context
response = client.complete(
    model="llama3-8b",
    prompt=f"Context: {docs}\n\nQuestion: {query}\n\nAnswer:",
    max_tokens=200
)
```

---

## Advanced Use Cases

### 1. **Multi-Agent Systems (Agent per GPU)**

You can assign specific models to specific workers to create specialized agents.
- **Worker 1 (AMD)**: Loads `llama-3-8b` for Coding tasks.
- **Worker 2 (NVIDIA)**: Loads `mistral-7b` for Reasoning/Review tasks.

**Configuration**:
No special config needed. Just route your load requests:
```bash
# Load Coding Agent on AMD Worker
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3-8b", "worker_id": "amd-gpu-0"}'

# Load Reasoning Agent on NVIDIA Worker
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "mistral-7b", "worker_id": "nvidia-gpu-0"}'
```

### 2. **Running Large Models (Future Optimization)**

> **Note**: Tensor and Pipeline parallelism within a single worker are active roadmap items utilizing the `burn` framework's multi-device capabilities. Currently, models must fit within the memory of a single GPU, or you can leverage Data Parallelism by deploying multiple identical model workers on the same machine.

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
cargo build --release --features=hip  # For AMD
# or
cargo build --release --features=cuda  # For NVIDIA

# Return to root
cd ..

# Create model directory
mkdir -p models

# Start coordinator
cd coordinator
uvicorn main:app --host 0.0.0.0 --port 8000

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

### Method 3: Kubernetes

```bash
# Create namespace
kubectl create namespace ai-cluster

# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/coordinator.yaml
kubectl apply -f k8s/worker.yaml

# Check status
kubectl get pods -n ai-cluster -w
```

For detailed installation instructions, see the [Deployment Guide](docs/deployment.md).

---

## Configuration

### Minimal Coordinator Configuration

```yaml
# config/coordinator.yaml
server:
  host: "0.0.0.0"
  port: 8000

discovery:
  method: "static"
  static_workers:
    - "localhost:50051"

models:
  cache_dir: "/data/models"
```

### Minimal Worker Configuration

```toml
# config/worker.toml
[worker]
id = "worker-1"

[grpc]
port = 50051

[gpu]
device_ids = [0]

[model_loader]
cache_dir = "/data/models"
```

For complete configuration options, see the [Configuration Guide](docs/configuration.md).

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/workers` | List workers |
| GET | `/v1/models` | List models |
| POST | `/v1/models/load` | Load a model |
| DELETE | `/v1/models/{name}` | Unload a model |
| POST | `/v1/completions` | Generate text |
| POST | `/v1/completions/batch` | Batch inference |
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
  "id": "cmpl-123456",
  "choices": [{
    "text": " in a faraway land, there lived a brave knight...",
    "tokens_generated": 50,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 50,
    "total_tokens": 54
  }
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

Pre-built dashboards are available in `monitoring/grafana-dashboards/`:

- **Cluster Overview**: Worker health, request rates, latency
- **GPU Details**: Utilization, memory, temperature, power
- **Model Performance**: Load times, cache hit rates, token throughput
- **Resource Usage**: CPU, memory, disk, network

### Distributed Tracing with Jaeger

```bash
# Access Jaeger UI
open http://localhost:16686

# View traces for specific operations
# - model_load
# - inference
# - worker_communication
```

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

# Run tests
pytest tests/
cd worker && cargo test

# Build documentation
cd docs && mkdocs build
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
- 💬 [Discord](https://discord.gg/ai-cluster)
- 🐦 [Twitter](https://twitter.com/ai_cluster)
- 📧 [Email](mailto:support@ai-cluster.com)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=caestrada1103/ai-cluster&type=Date)](https://star-history.com/#caestrada1103/ai-cluster&Date)

---

<div align="center">
  <sub>Built with ❤️ by the AI Cluster Team</sub>
  <br>
  <sub>© 2026 AI Cluster. All rights reserved.</sub>
</div>
