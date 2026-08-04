# AI Cluster Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Communication Flow](#communication-flow)
4. [Data Models](#data-models)
5. [Parallelism Strategies](#parallelism-strategies)
6. [Deployment Architecture](#deployment-architecture)
7. [Performance Considerations](#performance-considerations)
8. [Security Architecture](#security-architecture)
9. [Monitoring & Observability](#monitoring--observability)
10. [Fault Tolerance](#fault-tolerance)

---

## System Overview

The AI Cluster lets you run LLM inference on the **consumer GPUs you already
own** — gaming-PC cards in the 8–16 GB VRAM range, including cards sitting
idle in a second machine — across **both NVIDIA and AMD**. The core idea is
a quantized GGUF model that fits in one card's VRAM, or one model split
across several consumer cards when it doesn't. It provides a unified
interface for model inference while handling the complexities of
distribution, parallelism, and resource management. Datacenter-class
hardware, 70B-class models, and multi-machine/InfiniBand setups are not the
target — the architecture *also scales* in that direction, but it is not
what the system is designed around.

### High-Level Architecture

The diagram below shows the general, fully-scaled-out topology (multiple
coordinator replicas, AMD and NVIDIA worker pools, many GPUs per worker) so
every component has a place to live. The everyday deployment is a small
slice of this: one coordinator, one or two workers, one to a few consumer
GPUs — see [Deployment Architecture](#deployment-architecture) for that
picture.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Applications                        │
│                    (REST API, Web UI, CLI, SDKs)                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Load Balancer (Optional)                       │
│                       (HAProxy, Nginx, etc.)                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Coordinator Cluster                           │
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
│                 Implementation Layer (Current)                      │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │
│   │   Prometheus│ │    Grafana  │ │    Consul   │                   │
│   │    (Docker) │ │   (Docker)  │ │   Discovery │                   │
│   └─────────────┘ └─────────────┘ └─────────────┘                   │
│                                                                     │
│                 Infrastructure Layer (Roadmap)                      │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│   │    Redis    │ │    MinIO    │ │    Jaeger   │ │    Vault    │   │
│   │    Cache    │ │Model Storage│ │   Tracing   │ │   Secrets   │   │
│   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│                                                                     │
│   ┌─────────────┐                                                   │
│   │   Elastic   │                                                   │
│   │    Logs     │                                                   │
│   └─────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Consumer Hardware First**: Built around gaming-PC GPUs (~8–16 GB VRAM),
   including idle cards, not datacenter accelerators
2. **Cross-Vendor**: Supports AMD and NVIDIA (plus CPU fallback) with a
   unified interface — mix vendors in the same cluster
3. **Fit or Split**: Run a quantized model that fits in one card's VRAM, or
   split one model across several consumer cards when it doesn't
4. **Decoupling**: Separates control plane (coordinator) from data plane (workers)
5. **Elastic Scaling**: Workers can join/leave dynamically
6. **Fault Tolerance**: Automatic recovery from failures
7. **Performance First**: Optimized for low latency and high throughput
   within consumer VRAM/bandwidth limits

The same architecture also scales to multi-machine, higher-VRAM setups (see
[Deployment Architecture](#deployment-architecture)), but that is not the
headline use case.

---

## Core Components

### 1. Coordinator

The coordinator is the brain of the cluster, written in Python using FastAPI.

#### Responsibilities:
- **Service Discovery**: Find and register workers
- **Request Routing**: Direct inference requests to appropriate workers
- **Load Balancing**: Distribute load across workers
- **Health Monitoring**: Track worker health and availability
- **Model Registry**: Manage available models and their locations
- **API request handling** (API-key auth is implemented, opt-in; rate limiting is still planned)
- **Metrics Collection**: Expose Prometheus metrics

#### Internal Architecture:

```
┌─────────────────────────────────────────────────────────┐
│                     Coordinator                         │
├─────────────────────────────────────────────────────────┤
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│    │    FastAPI  │  │   gRPC      │  │  Prometheus │    │
│    │   Endpoints │  │   Client    │  │   Metrics   │    │
│    └─────────────┘  └─────────────┘  └─────────────┘    │
│           │               │               │             │
│     ┌─────▼───────────────▼───────────────▼──────┐      │
│     │           ClusterCoordinator Core          │      │
│     │  • Worker Management    • Request Routing  │      │
│     │  • Health Checks        • Circuit Breakers │      │
│     │  • Model Loading        • Queue Management │      │
│     └────────────────────────────────────────────┘      │
│          │                │                │            │
│   ┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐    │
│   │   Discovery  │ │    Router    │ │    Models    │    │
│   │   • Static   │ │  • Strategies│ │  • Registry  │    │
│   │   • mDNS     │ │  • Affinity  │ │  • Configs   │    │
│   │   • Consul   │ │  • Priorities│ │  • Validation│    │
│   └──────────────┘ └──────────────┘ └──────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 2. Worker

Workers perform the actual inference, written in Rust. Each worker runs
**one of two inference engines** against its GPU(s) — see
[Model Layer](#3-model-layer) below for how they compare.

#### Responsibilities:
- **GPU Management**: Detect and manage GPU resources. Supports mixed-GPU environments (e.g., mixing NVIDIA and AMD cards).
- **Model Loading**: Load models into GPU memory
- **Inference Execution**: Run forward passes and generate text
- **Parallelism**: Implement various parallelism strategies (current wiring
  status in [Parallelism Strategies](#parallelism-strategies))
- **Metrics**: Expose performance and resource metrics
- **Health Checks**: Report status to coordinator

#### Internal Architecture:

```
┌──────────────────────────────────────────────────────────┐
│                        Worker                            │
├──────────────────────────────────────────────────────────┤
│   ┌─────────────┐  ┌─────────────┐  ┌──────────────┐     │
│   │   gRPC      │  │   Metrics   │  │   Health     │     │
│   │   Server    │  │   Server    │  │   Checks     │     │
│   └─────────────┘  └─────────────┘  └──────────────┘     │
│           │               │               │              │
│    ┌──────▼───────────────▼───────────────▼──────┐       │
│    │              WorkerService Core             │       │
│    │  • LoadModel     • Infer     • GetStatus    │       │
│    │  • UnloadModel   • HealthCheck              │       │
│    └─────────────────────────────────────────────┘       │
│          │                │                │             │
│   ┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼────────┐    │
│   │ GPU Manager  │ │ Model Loader │ │ Parallelism   │    │
│   │ • Detection  │ │ • Download   │ │ (Burn only —  │    │
│   │ • Memory     │ │ • Convert    │ │  not wired to │    │
│   │ • wgpu/CUDA  │ │ • Cache      │ │  gRPC yet)    │    │
│   │ • ROCm/Vulkan│ │              │ │ • Pipeline    │    │
│   │              │ │              │ │ • Tensor      │    │
│   │              │ │              │ │ • Data        │    │
│   │              │ │              │ │ • Expert(stub)│    │
│   └──────────────┘ └──────────────┘ └───────────────┘    │
│           │               │                              │
│    ┌──────▼───────────────▼──────────────────────┐       │
│    │        Inference Engines (two, per model)    │       │
│    │  ┌────────────────────┐ ┌──────────────────┐ │       │
│    │  │ llama.cpp (GGUF)   │ │ Burn (safetensors)│ │       │
│    │  │  PRIMARY            │ │  FP32 reference   │ │       │
│    │  │ • Quantized weights │ │ • No quantization │ │       │
│    │  │   (Q4_K_M/Q5_K_M/   │ │   (FP32 only)     │ │       │
│    │  │   Q8_0, from disk)  │ │ • Llama/Qwen2.5/  │ │       │
│    │  │ • NVIDIA + AMD      │ │   DeepSeek(dense) │ │       │
│    │  │   (CUDA/ROCm/Vulkan)│ │ • Single GPU only │ │       │
│    │  │ • Single-GPU offload│ │                    │ │       │
│    │  │   today; upstream   │ │                    │ │       │
│    │  │   multi-GPU split   │ │                    │ │       │
│    │  │   not yet exposed   │ │                    │ │       │
│    │  │ • opt-in build      │ │                    │ │       │
│    │  │   (--features       │ │                    │ │       │
│    │  │   llamacpp)         │ │                    │ │       │
│    │  └────────────────────┘ └──────────────────┘ │       │
│    └───────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────┘
```

The `Parallelism` module (TP/PP/DP/Expert) only applies to the Burn engine
path today. See [Parallelism Strategies](#parallelism-strategies).

### 3. Model Layer

#### Supported Models — GGUF / llama.cpp engine (recommended):

This is the recommended way to run models on consumer GPUs: pick a
pre-quantized GGUF checkpoint (Q4_K_M/Q5_K_M/Q8_0/…) from Hugging Face and
point a registry entry at it with `engine = "llamacpp"` (see
[configuration.md](configuration.md)). Architecture support comes from
upstream llama.cpp itself — not a per-family reimplementation — so it covers
Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, Mixtral, and most other GGUF
exports, at whatever quantization the file ships in. Requires the worker to
be built with `--features llamacpp` (opt-in; see
[deployment.md](deployment.md)).

#### Supported Models — Burn engine (FP32 reference):

The Burn engine loads full-precision safetensors checkpoints and needs a
per-architecture implementation in `worker/models/`. Loadable today (worker
`model_loader.rs`): **Llama** (reference), **Qwen 2.5** (GQA + biases; Qwen3
rejected until q/k-norm lands), **DeepSeek dense** (Llama layout). DeepSeek
MoE model code exists but V3-style routing/MLA is not implemented. Mistral
model code exists but is not wired to the loader. Phi/Gemma/Mixtral: planned.
This is the default cargo build target (`--features wgpu`), but treat it as
the experimental/reference path — it always loads FP32, and runs on a single
GPU per worker.

---

## Communication Flow

### 1. Worker Discovery

```
┌───────────┐                  ┌──────────┐                   ┌──────────┐
│Coordinator│                  │Discovery │                   │  Worker  │
│           │                  │ Provider │                   │          │
└─────┬─────┘                  └─────┬────┘                   └─────┬────┘
      │                              │                              │
      │ Start                        │                              │
      │─────────────────────────────>│                              │
      │                              │                              │
      │                              │ Broadcast/Query              │
      │                              │─────────────────────────────>│
      │                              │                              │
      │                              │        Announce/Response     │
      │                              │<─────────────────────────────│
      │                              │                              │
      │        Worker Found          │                              │
      │<──────────────────────────────                              │
      │                              │                              │
      │ Connect and GetStatus        │                              │
      │────────────────────────────────────────────────────────────>│
      │                              │                              │
      │           Worker Status      │                              │
      │<────────────────────────────────────────────────────────────│
      │                              │                              │
      │ Add to active workers        │                              │
      │─────────────────────────────>│                              │
      │                              │                              │
```

### 2. Model Loading

```
┌───────────┐                  ┌──────────┐                   ┌──────────┐
│Coordinator│                  │  Worker  │                   │   GPU    │
│           │                  │          │                   │  Memory  │
└─────┬─────┘                  └─────┬────┘                   └─────┬────┘
      │                              │                              │
      │ LoadModel Request            │                              │
      │─────────────────────────────>│                              │
      │                              │                              │
      │                              │ Check available memory       │
      │                              │─────────────────────────────>│
      │                              │                              │
      │                              │ Memory OK                    │
      │                              │<─────────────────────────────│
      │                              │                              │
      │                              │ Load model weights           │
      │                              │──┐                           │
      │                              │  │ Download/Read             │
      │                              │<─┘                           │
      │                              │                              │
      │                              │ Allocate memory              │
      │                              │─────────────────────────────>│
      │                              │                              │
      │                              │ Transfer weights             │
      │                              │─────────────────────────────>│
      │                              │                              │
      │                              │ Initialize model             │
      │                              │──┐                           │
      │                              │  │ Create compute graph      │
      │                              │<─┘                           │
      │                              │                              │
      │ LoadModel Response           │                              │
      │<─────────────────────────────│                              │
      │                              │                              │
```

### 3. Inference Request

```
┌──────────┐                  ┌───────────┐                  ┌──────────┐
│  Client  │                  │Coordinator│                  │  Worker  │
└────┬─────┘                  └─────┬─────┘                  └─────┬────┘
     │                              │                              │
     │ POST /v1/completions         │                              │
     │─────────────────────────────>│                              │
     │                              │                              │
     │                              │ Select worker                │
     │                              │──┐                           │
     │                              │  │ Load balancing            │
     │                              │<─┘                           │
     │                              │                              │
     │                              │ gRPC Infer Request           │
     │                              │─────────────────────────────>│
     │                              │                              │
     │                              │                              │ Run inference
     │                              │                              │──┐
     │                              │                              │  │ Forward pass
     │                              │                              │  │ Generate tokens
     │                              │                              │<─┘
     │                              │                              │
     │                              │ Stream Response (tokens)     │
     │                              │<─────────────────────────────│
     │                              │                              │
     │ HTTP Stream Response         │                              │
     │<─────────────────────────────│                              │
     │                              │                              │
     │                              │ Complete                     │
     │                              │<─────────────────────────────│
     │                              │                              │
     │ Complete Response            │                              │
     │<─────────────────────────────│                              │
     │                              │                              │
```

---

## Data Models

### 1. WorkerInfo (Coordinator)

```python
@dataclass
class WorkerInfo:
    id: str                          # Unique worker identifier
    address: str                      # host:port for gRPC
    state: WorkerState                 # CONNECTING, HEALTHY, UNHEALTHY, OFFLINE
    gpus: List[GPUInfo]                # List of GPU devices
    loaded_models: Dict[str, LoadedModelInfo]  # Models loaded on this worker
    active_requests: int                # Currently processing requests
    total_requests: int                 # Lifetime request count
    avg_latency_ms: float               # Average inference latency
    last_health_check: float            # Timestamp of last check
    consecutive_failures: int           # Health check failures
```

### 2. GPUInfo (Worker)

```protobuf
message GPUInfo {
    int32 id = 1;                       // GPU index
    string name = 2;                     // GPU model name
    uint64 total_memory = 3;              // Total VRAM in bytes
    uint64 available_memory = 4;          // Free VRAM in bytes
    float utilization = 5;                 // GPU utilization (0-100)
    float temperature = 6;                 // Temperature in Celsius
    uint32 power_usage = 7;                // Power usage in watts
    repeated string capabilities = 8;       // e.g., ["fp16", "tensorcore"]
}
```

### 3. ModelConfig

```rust
pub struct ModelConfig {
    pub architecture: String,           // Model architecture name
    pub num_layers: usize,              // Number of transformer layers
    pub hidden_size: usize,              // Hidden dimension size
    pub num_attention_heads: usize,      // Number of attention heads
    pub num_kv_heads: usize,              // Number of KV heads (GQA)
    pub vocab_size: usize,                // Vocabulary size
    pub max_seq_len: usize,               // Maximum sequence length
    pub intermediate_size: usize,         // FFN intermediate size
    pub rms_norm_eps: f32,                // RMS norm epsilon
    pub rope_theta: f32,                   // Rotary embedding theta
    pub is_moe: bool,                      // Whether model uses MoE
    pub num_experts: Option<usize>,        // Number of experts (MoE)
    pub num_experts_per_tok: Option<usize>, // Experts per token (MoE)
}
```

---

## Parallelism Strategies

> **Note on Implementation Status**: The AI Cluster natively supports **Data Parallelism** (running independent models on multiple workers, either on the same machine or across the network). **Tensor and Pipeline Parallelism** core algorithms are implemented in `worker/src/parallelism.rs` (Burn engine) but are not yet wired to the gRPC inference service — models currently run on a single GPU per worker. **Expert Parallelism** is a stub (returns an error). Wiring TP/PP into the service layer is the next development step.
>
> **Practical path to "split one model across consumer GPUs" today**:
> upstream llama.cpp natively supports splitting a single GGUF model across
> multiple GPUs (layer-split or row-split) — this is the realistic way to
> run a model too big for one consumer card. Our llama.cpp engine wrapper
> currently only exposes single-device offload via `n_gpu_layers`; exposing
> multi-GPU split through the worker is the next llama.cpp-engine feature,
> not yet available. Until then, the model-fits-on-one-card case (a
> quantized GGUF sized to a single ~8–16 GB card) is the well-supported
> path.

### 1. Pipeline Parallelism (Core Implemented — Service Wiring Pending)

Splits model layers across multiple GPUs.

```
                  Pipeline Parallelism
┌────────────────────────────────────────────────────────┐
│                     Input Batch                        │
│                         │                              │
│                         ▼                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   GPU 0      │  │   GPU 1      │  │   GPU 2      │  │
│  │  Layers 0-10 │─>│ Layers 11-20 │─>│ Layers 21-30 │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                          │                             │
│                          ▼                             │
│                     Output Batch                       │
└────────────────────────────────────────────────────────┘

Micro-batching for efficiency:
┌───────────────────────────────────────────────────────┐
│ Batch: [MB0][MB1][MB2][MB3]                           │
│                                                       │
│ GPU0: MB0 FWD → GPU1: MB0 FWD → GPU2: MB0 FWD         │
│       MB1 FWD → GPU1: MB1 FWD → GPU2: MB1 FWD         │
│       MB2 FWD → GPU1: MB2 FWD → GPU2: MB2 FWD         │
│       MB3 FWD → GPU1: MB3 FWD → GPU2: MB3 FWD         │
└───────────────────────────────────────────────────────┘
```

### 2. Tensor Parallelism (Core Implemented — Service Wiring Pending)

Splits individual tensors across multiple GPUs. `num_shards == 1` is
mathematically identical to the non-parallel forward pass. GQA requires
`num_shards` to evenly divide `num_kv_heads`; a non-divisor request is
silently clamped down to the nearest valid value (`clamp_shards`, which
logs a warning on clamp).

```
                    Tensor Parallelism
┌────────────────────────────────────────────────────────┐
│                      Input X [N, D]                    │
│                          │                             │
│        ┌─────────────────┼─────────────────┐           │
│        ▼                 ▼                 ▼           │
│  ┌────────────┐    ┌──────────────┐   ┌─────────────┐  │
│  │   GPU 0    │    │   GPU 1      │   │   GPU 2     │  │
│  │ X[:,0:D/3] │    │ X[:,D/3:2D/3]│   │ X[:,2D/3:D] │  │
│  │ W0 [D/3,4D]│    │ W1 [D/3,4D]  │   │ W2 [D/3,4D] │  │
│  │ Y0 = X0·W0 │    │ Y1 = X1·W1   │   │ Y2 = X2·W2  │  │
│  └───────┬────┘    └────────┬─────┘   └───────┬─────┘  │
│          └──────────────────┼─────────────────┘        │
│                             ▼                          │
│                    All-Reduce Y = Y0+Y1+Y2             │
│                             │                          │
│                             ▼                          │
│                      Output Y [N,4D]                   │
└────────────────────────────────────────────────────────┘
```

### 3. Data Parallelism

Replicates model across GPUs, splits batch.

```
                    Data Parallelism
┌──────────────────────────────────────────────────────┐
│                      Input Batch                     │
│                          │                           │
│        ┌─────────────────┼─────────────────┐         │
│        ▼                 ▼                 ▼         │
│  ┌───────────┐     ┌────────────┐    ┌────────────┐  │
│  │   GPU 0   │     │   GPU 1    │    │   GPU 2    │  │
│  │  Model    │     │  Model     │    │  Model     │  │
│  │  Copy     │     │  Copy      │    │  Copy      │  │
│  │           │     │            │    │            │  │
│  │ Batch 0-33│     │ Batch 34-66│    │ Batch 67-99│  │
│  │ Inference │     │ Inference  │    │ Inference  │  │
│  └───────┬───┘     └──────┬─────┘    └───────┬────┘  │
│          └────────────────┼──────────────────┘       │
│                           ▼                          │
│                  Concatenate Results                 │
│                           │                          │
│                           ▼                          │
│                    Output Batch                      │
└──────────────────────────────────────────────────────┘
```

### 4. Expert Parallelism (MoE) (Stub — Returns Error)

Distributes experts across GPUs for Mixture of Experts models.

```
                    Expert Parallelism
┌─────────────────────────────────────────────────────────┐
│                      Input Tokens                       │
│                            │                            │
│                      Router/Gate                        │
│          ┌─────────────────┼─────────────────┐          │
│          ▼                 ▼                 ▼          │
│    ┌───────────┐     ┌───────────┐     ┌───────────┐    │
│    │   GPU 0   │     │   GPU 1   │     │   GPU 2   │    │
│    │ Experts   │     │ Experts   │     │ Experts   │    │
│    │ 0-3       │     │ 4-7       │     │ 8-11      │    │
│    │           │     │           │     │           │    │
│    │ Tokens →  │     │ Tokens →  │     │ Tokens →  │    │
│    │ Expert 2  │     │ Expert 5  │     │ Expert 9  │    │
│    └──────┬────┘     └──────┬────┘     └──────┬────┘    │
│           └─────────────────┼─────────────────┘         │
│                             ▼                           │
│                    Combine Expert Outputs               │
│                             │                           │
│                             ▼                           │
│                       Output Tokens                     │
└─────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

The common case this project targets: one machine (often a gaming PC) with
one or more consumer GPUs — including a card that would otherwise sit idle —
running a quantized GGUF model via the llama.cpp engine. The same worker/
coordinator design also scales out to multiple machines and larger GPU
counts (below), but multi-machine/InfiniBand setups are "also scales to,"
not the common deployment.

### 1. Single Machine, Multiple GPUs

```
┌────────────────────────────────────────────────────────────┐
│                      Single Server                         │
│    ┌─────────────────────────────────────────────────┐     │
│    │              Coordinator Container              │     │
│    │              Port: 8000 (API + /metrics)         │     │
│    └──────────────────────┬──────────────────────────┘     │
│                           │                                │
│  ┌─────────────┬──────────┴──┬─────────────┬─────────────┐ │
│  │    GPU 0    │    GPU 1    │    GPU 2    │    GPU 3    │ │
│  │ Worker 0    │ Worker 1    │ Worker 2    │ Worker 3    │ │
│  │ Port: 50051 │ Port: 50052 │ Port: 50053 │ Port: 50054 │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
│                            │                               │
│    ┌───────────────────────┴─────────────────────────┐     │
│    │         Prometheus │ Grafana │ Redis            │     │
│    │         Ports: 9090, 3000, 6379                 │     │
│    └─────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────┘
```

### 2. Multi-Machine Cluster (also scales to)

> The design also scales to multiple machines/10GbE/InfiniBand, but that is
> a datacenter-style deployment, not the project's headline consumer-GPU
> scenario above.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Machine 1     │     │   Machine 2     │     │   Machine 3     │
│ (Coordinator)   │     │  (Worker Pool)  │     │  (Worker Pool)  │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ Coordinator     │     │ GPU 0 │ GPU 1   │     │ GPU 0 │ GPU 1   │
│ Load Balancer   │     │ Worker│ Worker  │     │ Worker│ Worker  │
│ API Gateway     │     │       │         │     │       │         │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ Prometheus      │     │ GPU 2 │ GPU 3   │     │ GPU 2 │ GPU 3   │
│ Grafana         │     │ Worker│ Worker  │     │ Worker│ Worker  │
│ Redis           │     │       │         │     │       │         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌───────▼─────────┐
                         │   Network       │
                         │ 10GbE/InfiniBand│
                         └─────────────────┘
```

### 3. Kubernetes Deployment

> **Planned** — no manifests ship in this repo yet; the sketch below is a design target.

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: ai-worker
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: worker
        image: ai-worker:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # or amd.com/gpu for AMD
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: shm
        emptyDir:
          medium: Memory
```

---

## Performance Considerations

### 1. Memory Management

- **Quantization (Implemented — llama.cpp/GGUF engine only)**: GGUF models
  carry their own quantization (Q4_K_M, Q5_K_M, Q8_0, …), baked in at
  conversion time upstream; this is what lets a 7B–14B-class model fit an
  8–16 GB consumer card. The Burn engine has **no quantization** — it always
  loads FP32 safetensors, and `quantization != "none"` is rejected.
- **Memory Pooling (Roadmap)**: Not implemented today. The worker's
  `gpu_manager.rs` tracks allocations for OOM admission control only (a
  counter, not a reusable buffer pool); any real pooling would have to come
  from the Burn backend's own allocator, which ai-cluster does not control
  or verify.
- **Paged KV Cache (Roadmap)**: Planned implementation to reduce memory usage by 50-70% for long sequences

### 2. Latency Optimization

- **Tensor Core Utilization (Roadmap)**: No FP16/BF16/mixed-precision or
  tensor-core-specific code path exists yet — the Burn/wgpu backend runs
  default f32 settings. Aspirational, not implemented.
- **Continuous Batching (Roadmap)**: Planned to provide 2-3x throughput improvement
- **Speculative Decoding (Roadmap)**: Planned logic for 2-3x speedup for generation
- **Flash Attention (Roadmap)**: Planned integration for 2-4x faster attention computation

### 3. Throughput Scaling

> **Projected goals, not measured benchmarks** (same disclaimer as the
> README). The multi-GPU rows below additionally assume pipeline-parallel
> service wiring that does not exist yet (see
> [Parallelism Strategies](#parallelism-strategies) — TP/PP are implemented
> in `parallelism.rs` but not wired to gRPC, so today every model runs on a
> single GPU per worker). None of these numbers are specific to the
> llama.cpp/GGUF engine's real quantized throughput on consumer cards either
> — treat the whole table as a directional target, not a benchmark.

| GPUs | Model | Batch Size | Throughput (tokens/s) | Scaling Efficiency |
|------|-------|------------|----------------------|-------------------|
| 1 | DeepSeek-7B | 1 | 45 | 1.0x |
| 2 | DeepSeek-7B | 2 | 88 | 0.98x |
| 4 | DeepSeek-7B | 4 | 170 | 0.94x |
| 1 | Llama-3-8B | 1 | 52 | 1.0x |
| 2 | Llama-3-8B | 2 | 101 | 0.97x |
| 4 | Llama-3-70B | 1 | 12 | 1.0x (pipeline) |
| 8 | Llama-3-70B | 2 | 22 | 0.92x (pipeline) |

---

## Security Architecture

> Implemented today: secure-by-default bind hosts, coordinator API-key auth,
> worker gRPC shared-secret auth. Not implemented: TLS/mTLS (transport is
> plaintext gRPC/HTTP — deploy on a trusted network or behind your own TLS
> proxy), rate limiting, audit logging, secure erasure.

### 1. Authentication & Authorization (implemented, opt-in)

- **Coordinator**: `COORDINATOR_API_KEYS` gates the whole HTTP surface except
  `/health`/`/metrics` (`Authorization: Bearer <key>` or `x-api-key: <key>`).
  The coordinator refuses to start bound to a non-loopback address with no
  keys set.
- **Worker gRPC**: `WORKER_GRPC_AUTH_TOKEN` (or `grpc_auth_token` in
  `worker.toml`) is a shared secret checked on every gRPC call; the worker
  logs a warning if it binds non-loopback with no token set.
- **Bind hosts default to loopback**: worker gRPC (`grpc_bind_host`) and
  spawned `llama-server` children (`llamaserver_bind_host`) both default to
  `127.0.0.1`. Containers opt in via `WORKER_GRPC_BIND_HOST` /
  `LLAMASERVER_BIND_HOST`. See [configuration.md](configuration.md) and
  [deployment.md](deployment.md).

```
┌──────────┐      ┌──────────────┐      ┌──────────┐
│  Client  │─────▶│ Coordinator  │─────▶│ Worker   │
│          │      │ (API keys)   │      │ (gRPC    │
│          │      │              │      │  token)  │
└──────────┘      └──────────────┘      └──────────┘
     │                 │                     │
     │ Bearer/x-api-key│                     │
     │────────────────>│  gRPC + shared      │
     │                 │  secret metadata    │
     │                 │────────────────────>│
```

### 2. Network Security

- **TLS/mTLS**: not implemented — gRPC and HTTP traffic is plaintext.
  Deploy on a trusted network or terminate TLS with your own reverse proxy.
- **Network Policies**: Kubernetes network policies are part of the
  not-yet-shipped Kubernetes manifests.
- **API Rate Limiting**: planned, not implemented.

### 3. Data Security

- **Memory Isolation**: Processes run in separate memory spaces
- **Secure Erasure**: Memory zeroed after model unloading
- **Audit Logging**: All access logged for compliance

*(Roadmap)* 
- **Model Encryption**: Encrypted model weights at rest
- **Secrets Management**: Vault integration for securely managing API keys

---

## Monitoring & Observability

### 1. Metrics Collected

| Category | Metrics | Collection Interval |
|----------|---------|-------------------|
| System | CPU, Memory, Disk, Network | 15s |
| GPU | Utilization, Temperature, Power, Memory | 10s |
| Inference | Request Rate, Latency, Tokens/sec | 5s |
| Model | Load Time, Memory Usage, Cache Hit Rate | 30s |
| Cluster | Worker Count, Queue Size, Error Rate | 10s |

### 2. Grafana Dashboards

```
┌─────────────────────────────────────────────────────────┐
│  AI Cluster Overview Dashboard                          │
├─────────────────────────────────────────────────────────┤
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│   │ Active Reqs │ │ Queue Size  │ │ Error Rate  │       │
│   │     12      │ │      3      │ │    0.1%     │       │
│   └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │ Request Rate (requests/sec)                     │   │
│   │   ████▁▁▁▁████▁▁▁▁████▁▁▁▁████▁▁▁▁              │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│   │ GPU 0       │ │ GPU 1       │ │ GPU 2       │       │
│   │ 75% Util    │ │ 82% Util    │ │ 45% Util    │       │
│   │ 65°C        │ │ 68°C        │ │ 52°C        │       │
│   └─────────────┘ └─────────────┘ └─────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### 3. Distributed Tracing (Roadmap)

Jaeger integration is planned for providing end-to-end tracing functionality.

```
┌─────────────────────────────────────────────────────┐
│ Trace: inference-123456 (Planned View)              │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Coordinator: route_request ──────────────────┐      │
│   [2ms]                                      │      │
│                                              │      │
│ Worker: load_model                           │      │
│   [150ms]                                    │      │
│                                              │      │
│ Worker: infer ─────────────────────────────┐ │      │
│   [450ms]                                  │ │      │
│   ├─ tokenization [2ms]                    │ │      │
│   ├─ forward_pass [400ms]                  │ │      │
│   │   ├─ attention [150ms]                 │ │      │
│   │   ├─ mlp [120ms]                       │ │      │
│   │   └─ moe_routing [130ms]               │ │      │
│   └─ sampling [48ms]                       │ │      │
│                                            │ │      │
│ Coordinator: stream_response ──────────────┘ │      │
│   [5ms]                                      │      │
│                                              │      │
│ Total: 607ms                                 │      │
└─────────────────────────────────────────────────────┘
```

---

## Fault Tolerance

### 1. Worker Failure

```
┌───────────┐      ┌──────────┐      ┌──────────┐
│Coordinator│      │ Worker 1 │      │ Worker 2 │
└─────┬─────┘      └─────┬────┘      └──────┬───┘
      │                  │                  │
      │ Health Check     │                  │
      │─────────────────>│                  │
      │                  │                  │
      │ Health Check     │                  │
      │─────────────────>│ (no response)    │
      │                  │                  │
      │ Health Check     │                  │
      │─────────────────>│ (no response)    │
      │                  │                  │
      │ Mark UNHEALTHY   │                  │
      │──┐               │                  │
      │  │ 3 failures    │                  │
      │<─┘               │                  │
      │                  │                  │
      │ Redistribute     │                  │
      │ requests         │                  │
      │────────────────────────────────────>│
      │                  │                  │
```

### 2. Model Loading Failure

```python
# Retry logic with exponential backoff
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(WorkerError)
)
async def load_model_on_worker(worker, model_name):
    try:
        response = await worker.stub.LoadModel(request)
        if not response.success:
            raise WorkerError(f"Failed to load: {response.message}")
        return response
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            raise WorkerError("Worker unavailable")
        raise
```

### 3. Request Timeout

```python
async def infer_with_timeout(coordinator, request, timeout=30):
    try:
        # Create task with timeout
        result = await asyncio.wait_for(
            coordinator.infer(request),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        # Log timeout
        logger.error(f"Request {request.id} timed out")
        
        # Try fallback worker
        return await fallback_inference(coordinator, request)
```

---

## Conclusion

The AI Cluster architecture lets you run LLM inference on the consumer
NVIDIA/AMD GPUs you already own — a quantized GGUF model via the primary
llama.cpp engine when it fits one card, or the same design scaling out to
more cards/machines when it doesn't. Honest status, as of this doc:

- **Cross-Vendor Today**: AMD, NVIDIA, and CPU fallback, unified interfaces
- **Quantization Today**: via the llama.cpp/GGUF engine only (Burn stays FP32)
- **Parallelism Partial**: Data Parallelism works; Tensor/Pipeline Parallelism
  algorithms exist but aren't wired to inference yet (single GPU per worker
  on the Burn path); multi-GPU GGUF split is not yet exposed either
- **Deployment Today**: Docker Compose and native builds; Kubernetes manifests
  are a design sketch, not shipped
- **Secure by Default, Opt-in to Exposure**: loopback binds and API-key/gRPC-token
  auth ship today; TLS/mTLS and rate limiting do not — see
  [Security Architecture](#security-architecture)

For more information, see:
- [API Reference](api_reference.md)
- [Deployment Guide](deployment.md)
- [Configuration Guide](configuration.md)
- [Troubleshooting](troubleshooting.md)