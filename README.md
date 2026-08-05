# AI Cluster - Distributed Multi-GPU Inference System

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![ROCm](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://rocm.docs.amd.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/architecture.md)

## Run it

One command, then open a browser:

```bash
./scripts/run-local.sh --ui
```

Chat UI at **http://localhost:8080**, API at **http://localhost:8000**. Ctrl-C
stops everything. Pick a model in the UI's model dropdown; the coordinator
loads it on first use.

Without `--ui` it starts the API only. `--build` rebuilds the worker first,
`--model NAME` preloads a model, `--help` lists everything. First-time setup
is in [Installation](#installation); `--ui` also needs Docker.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Security: Secure by Default](#security-secure-by-default)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Supported Models](#supported-models)
- [Performance](#performance)
- [Use Cases](#use-cases)
- [Use on Real Scenarios](#use-on-real-scenarios)
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
| **Distributed GGUF split (cross-node ggml-RPC)** | 🔶 Done (Partially) | Coordinator side fully implemented: resolves the `[models.X.distributed]` lead/peers, loads peers before the lead, derives or applies the tensor split, unloads lead-first, and rolls back a failed load (`coordinator/coordinator.py`). Worker side is still a stub — no real `ggml-rpc-server`/lead process yet, so this can't run end-to-end. See `docs/configuration.md`. |
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
- **Worker**: Rust (Burn Framework + llama.cpp, Tokio, Tonic)
- **Inference Engines**: **llama.cpp/GGUF** (`llama-cpp-2 =0.1.150`) — **recommended engine** for quantized inference on consumer NVIDIA/AMD GPUs (native Q4_K_M/Q5_K_M/Q8_0 quantization; opt-in via the `llamacpp` cargo feature) + **Burn 0.19** (safetensors, FP32, the **default** cargo build feature) — experimental/reference engine, no quantization yet
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

## Security: Secure by Default

Everything binds to loopback until you explicitly open it up:

- Worker gRPC and spawned `llama-server` children bind `127.0.0.1` by
  default, on bare metal and in containers alike.
- `WORKER_GRPC_AUTH_TOKEN` is a shared secret required on every gRPC call
  once set; the worker logs a warning if it ends up bound to a non-loopback
  address with no token configured.
- The coordinator refuses to start bound to a non-loopback address unless
  `COORDINATOR_API_KEYS` is set.
- Individual keys can carry a role (`admin`/`user`) and a model scope via
  `COORDINATOR_API_KEY_FILE`, layered on top of `COORDINATOR_API_KEYS`. See
  [Configuration Guide](docs/configuration.md) for the TOML shape and the
  compatibility guarantee for deployments that don't set it.
- `docker compose up` publishes only the coordinator (8000), Grafana (3000),
  and Prometheus (9099) — worker gRPC/metrics, `llama-server`, and
  Open-WebUI ports stay off the host. Opt in with
  `docker compose -f docker-compose.yml -f docker-compose.expose-ports.yml up`.

Containers open the worker binds up via `WORKER_GRPC_BIND_HOST` /
`LLAMASERVER_BIND_HOST` (both default to `0.0.0.0` in `docker-compose.yml`);
`config/worker.toml` itself never widens a bind. See
[Deployment Guide](docs/deployment.md) and
[Configuration Guide](docs/configuration.md) for the full picture.

---

## Quick Start

> **Recommended for consumer GPUs**: the steps below bring up the full stack
> using the default Burn/wgpu build (FP32, single GPU per worker) so you can
> get everything running quickly. For **quantized** models on consumer
> NVIDIA/AMD GPUs (~8-16 GB VRAM) — the project's primary goal — build the
> worker with the **llama.cpp/GGUF engine** instead (`--features llamacpp`
> [+ `llamacpp-vulkan`/`llamacpp-cuda`]) and load a `[models.X.gguf]` entry;
> see [Adding a GGUF model](#adding-a-gguf-model-llamacpp-engine-recommended)
> below for the recommended end-to-end example.

### Method 1: Docker Compose

The easiest way to get started is using Docker Compose, which handles all dependencies automatically.

```bash
# 1. Clone the repository
git clone https://github.com/caestrada1103/ai-cluster.git
cd ai-cluster

# 2. Configure Environment. REQUIRED:
#    - GRAFANA_ADMIN_PASSWORD (compose refuses to start Grafana without it)
#    - COORDINATOR_API_KEYS (the coordinator's port is published to the host,
#      a non-loopback bind, so it refuses to start with no keys set — secure
#      by default; generate one with `openssl rand -hex 32`)
#    - COMPOSE_PROFILES matching your GPU vendor (no profile = no worker starts)
#    HF_TOKEN is needed for gated models like Llama 3.
cp .env.example .env
# Edit .env: set GRAFANA_ADMIN_PASSWORD, COORDINATOR_API_KEYS, COMPOSE_PROFILES, HF_TOKEN

# 3. Build and start with Docker Compose
docker compose up -d --build

# 4. Check that everything is running (/health needs no key)
curl http://localhost:8000/health

# 5. Run your first inference (Model will auto-download and load)
H='Authorization: Bearer <your COORDINATOR_API_KEYS value>'
curl -H "$H" -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-7b",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

See [GPU variants](docs/deployment.md#gpu-variants) for the exact
`COMPOSE_PROFILES` value per vendor.

### Method 2: Local Execution (Native GPU) — verified

**Shortcut:** once the one-time setup below is done, `./scripts/run-local.sh`
starts the worker and coordinator together, waits for both to be ready, prints
working `curl` examples, and stops both on Ctrl-C. `--build` rebuilds the
worker first; `--model NAME` loads a model once healthy. The manual steps
follow for when you want the two processes in separate terminals.

Runs entirely on loopback (secure by default) — no ports leave the machine.
This example uses `qwen3.6-35b-a3b-gguf`, a `llamaserver`-engine model, so it
also needs a `llama-server` binary on `PATH` (build steps in
[Deployment → llama-server for agentic serving](docs/deployment.md); Node
20+ is required for that build only, not for anything below).

```bash
# one-time
sudo apt-get install -y protobuf-compiler libclang-dev libssl-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
python3 -m venv .venv && . .venv/bin/activate && pip install -r coordinator/requirements-dev.txt
cd worker && cargo build --release --features wgpu,llamacpp,llamacpp-cuda && cd ..

# terminal 1
export PATH="$HOME/.local/bin:$PATH"          # so the worker finds llama-server
RUST_LOG=info ./target/release/ai-worker --port 50051

# terminal 2
. .venv/bin/activate
export COORDINATOR_STATIC_WORKERS='["localhost:50051"]'
export COORDINATOR_API_KEYS='sk-local-dev'
uvicorn coordinator.main:app --host 127.0.0.1 --port 8000

# use it
H='Authorization: Bearer sk-local-dev'
curl -H "$H" -X POST localhost:8000/v1/models/load -H 'Content-Type: application/json' \
  -d '{"model_name":"qwen3.6-35b-a3b-gguf"}'
curl -H "$H" -X POST localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b-gguf","messages":[{"role":"user","content":"hi"}],"max_tokens":200}'
```

The worker binary lands at the **workspace root** `target/release/ai-worker`
— not `worker/target/` — because `worker` is a Cargo workspace member. Swap
`llamacpp-cuda` for `llamacpp-vulkan` on AMD/Intel, or drop it for CPU-only
GGUF inference; add `cuda`/`rocm` in place of `wgpu` for a native (non-Vulkan)
Burn backend. See [Adding a GGUF model](#adding-a-gguf-model-llamacpp-engine-recommended)
for a lighter `llamacpp`-engine example instead of `llamaserver`.

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
- [Sizing Examples (measured on real hardware)](docs/configuration.md#sizing-examples-measured-on-real-hardware)
- [Security: Secure by Default](#security-secure-by-default)

---

## Supported Models

AI Cluster runs models through two inference engines: **llama.cpp/GGUF**
(recommended — quantized, runs on consumer NVIDIA + AMD GPUs) and
**Burn/safetensors** (experimental FP32 reference engine, single GPU per
worker).

### GGUF models via llama.cpp (recommended)

This is the recommended way to run models on consumer-grade GPUs (~8-16 GB
VRAM): weights are loaded **quantized** (Q4_K_M, Q5_K_M, Q8_0, …) so larger
models fit in limited VRAM, and inference runs on both NVIDIA (CUDA/Vulkan)
and AMD (ROCm/Vulkan). It's opt-in — build the worker with `--features
llamacpp` (see [Adding a GGUF model](#adding-a-gguf-model-llamacpp-engine-recommended)
below). Splitting one GGUF model across several consumer GPUs is on the
worker roadmap; today one GGUF model loads per worker process.

| Model | Status | Notes |
|-------|--------|-------|
| **Any GGUF checkpoint** (e.g. Qwen2.5, Llama 3.1, Qwen2.5-Coder, …) | ✅ Implemented (opt-in: build worker with `--features llamacpp` [+ `llamacpp-vulkan`/`llamacpp-cuda`]) | Native quantization: Q4_K_M, Q5_K_M, Q8_0, …; NVIDIA + AMD; multi-GPU split upcoming |

### Agentic serving via llama-server (tool calling)

A third serving mode, `engine = "llamaserver"`, runs OpenAI/Anthropic **tool
calling** for coding agents (Claude Code, Cline, aider, …): the worker supervises
a `llama-server` process per model and the coordinator proxies agentic HTTP
requests (including `/v1/messages`) straight to it. The default Docker worker
image bundles the `llama-server` binary; bare-metal setup, the pinned llama.cpp
version, and the coordinator↔worker port requirements are covered in the
[Deployment Guide](docs/deployment.md). The number of concurrent
conversation slots is `[models.X.llamaserver] instances` in
`config/models.toml` (default 1), also overridable per request with
`POST /v1/models/load {"instances": N}` — see
[Configuration Guide](docs/configuration.md). See
[Use on Real Scenarios](#use-on-real-scenarios) for the end-to-end Claude Code
walkthrough.

### Burn / safetensors models (experimental FP32 reference)

The Burn engine is the **default** cargo build feature (`--features
wgpu|cuda|rocm`) and serves as the experimental / reference path: it loads
full-precision **FP32** safetensors weights — it does not quantize (non-`none`
quantization requests are rejected) — and runs a model on a **single GPU** per
worker. Tensor/pipeline parallelism exist in `worker/src/parallelism.rs` but
are not yet wired to inference.

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

### Adding a GGUF model (llama.cpp engine, recommended)

```toml
# config/models.toml — no architecture block needed (read from GGUF metadata)
[models."qwen2.5-0.5b-gguf"]
family = "qwen"
parameters = "0.5B"
min_memory_gb = 1
engine = "llamacpp"

[models."qwen2.5-0.5b-gguf".gguf]
repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
file = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
n_gpu_layers = -1  # -1 = all layers on GPU
n_ctx = 4096
```

The worker must be built with the engine enabled: `cargo build --release --features wgpu,llamacpp` (add `llamacpp-vulkan` or `llamacpp-cuda` for GPU offload), or `docker build -f docker/Dockerfile.worker --build-arg WORKER_FEATURES="llamacpp,llamacpp-vulkan" .`. Then load and run it like any other model:

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model_name": "qwen2.5-0.5b-gguf"}'

curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-gguf",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### Adding a Burn/safetensors model (reference engine)

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

**Quantization** isn't on this roadmap list — it's available **today**, but only via the **llama.cpp/GGUF engine** (Q4_K_M, Q5_K_M, Q8_0, …; opt-in `--features llamacpp`), which typically cuts memory ~50-75% vs. full precision depending on the quant level. The default Burn/safetensors path still loads weights as FP32 and rejects non-`none` quantization requests (see [Features](#features)).

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

## Use on Real Scenarios

AICluster turns your own PCs into a private AI coding assistant. One PC (the
**coordinator**) receives requests; the PCs with graphics cards (the **workers**)
run the AI models. Once it's running, coding tools like
[Claude Code](https://docs.claude.com/en/docs/claude-code) on your laptop talk to
it exactly like they'd talk to a cloud AI — but every request, and your code,
stay on your own network.

Here's the whole thing, end to end:

**1. On a PC with a graphics card, start a worker.** It needs a small helper
program (`llama-server`) that actually runs the models; the Docker image already
includes it, so `docker compose up -d --build` is all you need. (On a bare-metal
install you build it once — see
[Deployment → llama-server for agentic serving](docs/deployment.md).)

**2. On the PC that will receive requests, start the coordinator.** With Docker
Compose it comes up next to the worker. To require a password on every request,
set one variable before starting (skip it to run with no password):

```bash
export COORDINATOR_API_KEYS=your-secret-key
```

**3. From your laptop, check it works** — replace `<coordinator-host>` with the
coordinator PC's name or IP address:

```bash
python scripts/validate_agentic.py \
  --base-url http://<coordinator-host>:8000 \
  --api-key your-secret-key
```

**4. Point Claude Code at it.** On each laptop, copy these four lines into the
terminal — changing `<coordinator-host>` to the coordinator PC's address — then
run `claude`:

```bash
export ANTHROPIC_BASE_URL=http://<coordinator-host>:8000
export ANTHROPIC_AUTH_TOKEN=your-secret-key                          # any text if you skipped the password
export ANTHROPIC_MODEL=qwen3-coder-30b-a3b-instruct-gguf             # the "big" model
export ANTHROPIC_SMALL_FAST_MODEL=devstral-small-2-24b-instruct-gguf # the "quick" model
claude
```

That's it — Claude Code now uses your cluster. It can read files, run commands,
and edit code (tool use) just as it does against the cloud, because the models
above are trained for exactly that.

**Good to know:**

- **The first question is slow, then it's fast.** The first time you use a model,
  its file (~15 GB) downloads and starts up automatically; after that it stays
  loaded and answers quickly. Running the check in step 3 first gets this out of
  the way.
- **Your chat history stays on your laptop.** Claude Code keeps each conversation
  locally; the cluster just answers one request at a time and remembers nothing
  between them.
- **The model name must match one from `config/models.toml`** (like the two
  above). Claude Code's built-in menu only lists cloud models, so pick yours with
  the `ANTHROPIC_MODEL` variables, not the in-app menu.
- **Seeing `400` errors?** Add `export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`
  and restart `claude`.
- **Using a different tool?** Setups for Cline, aider, Continue, and Open WebUI
  live in [docs/clients.md](docs/clients.md).

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
uvicorn coordinator.main:app --host 127.0.0.1 --port 8000

# In another terminal (from the repo root — the binary is at the workspace
# root target/, not worker/target/, since `worker` is a Cargo workspace member):
./target/release/ai-worker --port 50051 --gpu-ids 0
```

### Method 2: Docker Compose

See [Quick Start → Method 1](#method-1-docker-compose) above for the full,
current command sequence (env vars required, GPU profile selection).

```bash
docker compose up -d --build
docker compose logs -f
docker compose down
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
# 0.0.0.0 requires COORDINATOR_API_KEYS to be set too — the coordinator refuses
# to bind non-loopback with no auth configured. Use 127.0.0.1 for local-only.
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
# grpc_bind_host / llamaserver_bind_host both default to 127.0.0.1 — see
# Security: Secure by Default above for the opt-in env vars.
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
