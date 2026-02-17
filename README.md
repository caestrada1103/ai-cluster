# AI‑Cluster

  An end‑to‑end AI inference cluster that combines a Python coordinator
   with Rust workers built on the Burn framework.
  It supports **multi‑GPU inference** on both AMD (ROCm) and
  NVIDIA (CUDA) platforms, **model parallelism** (pipeline, tensor,
  data), **dynamic model loading**, **gRPC communication**, and
  **Prometheus metrics**.

  ## Architecture

  +--------------------------------+
  +----------------------------+
  |      Python Coordinator        |<------>|        Rust Worker(s)
    |
  | (grpcio + prometheus‑client)   |        | (burn + tonic + metrics)
    |
  +--------------------------------+
  +----------------------------+


  * The coordinator exposes a gRPC service (`InferenceService`) to
  submit inference jobs.
  * Workers expose a gRPC service that the coordinator calls.
    Each worker can load models at runtime and use GPU resources via
  ROCm or CUDA.
  * Prometheus metrics are exported by both the coordinator
  (`/metrics`) and the workers (`/metrics`).

  ## Getting Started

  ```bash
  # Build the Rust workspace
  cd worker
  cargo build --release

  # Install Python dependencies
  cd ..
  pip install -e .

  # Run locally
  python -m ai_cluster.coordinator --grpc-port 50051
  # In a separate terminal
  ./target/release/worker


  Deployment

  The system can be deployed using Docker Compose:

  docker compose up -d


  The docker-compose.yml creates two services:
  - coordinator – the Python process.
  - worker – the Rust binary.
  The worker container automatically detects AMD or NVIDIA GPUs, loads
  the requested model, and registers metrics with Prometheus.

  Features

  Feature: Multi‑GPU
  Description: AMD (ROCm) and NVIDIA (CUDA) GPU support via Burn’s
    Device abstraction.
  ────────────────────────────────────────
  Feature: Model Parallelism
  Description: Pipeline, tensor, and data parallelism are supported
    through Burn’s Parallel primitives.
  ────────────────────────────────────────
  Feature: Dynamic Model Loading
  Description: Models can be loaded on demand at runtime; the worker
    keeps an in‑memory registry.
  ────────────────────────────────────────
  Feature: gRPC Communication
  Description: Uses tonic (Rust) and grpcio (Python) for fast
    inter‑process RPC.
  ────────────────────────────────────────
  Feature: Prometheus Metrics
  Description: Exported via the prometheus crate in Rust and
    prometheus‑client in Python.

# 📁 Project Structure

```
AICluster/
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