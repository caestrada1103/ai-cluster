# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AICluster is a distributed LLM inference platform with two core components:
- **Coordinator** (`coordinator/`) — Python FastAPI service providing an OpenAI-compatible REST API, worker discovery, load balancing, and model registry.
- **Worker** (`worker/`) — Rust service using the Burn deep learning framework that runs GPU inference and exposes a gRPC endpoint.

Clients talk REST to the coordinator; the coordinator talks gRPC (protobuf) to workers. Protocol definitions live in `proto/cluster.proto` and generated bindings are in `coordinator/proto/` and built by `worker/build.rs`.

## Commands

### Docker Compose (full stack)
```bash
docker compose up -d --build   # Start coordinator + worker + Prometheus + Grafana + Open-WebUI
docker compose logs -f
docker compose down
```

### Coordinator (Python)
```bash
cd coordinator
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Worker (Rust) — choose the feature flag for your hardware
```bash
cd worker
cargo build --release --features wgpu    # Universal — Vulkan/DX12/Metal, auto-detects AMD/NVIDIA/Intel (default)
cargo build --release --features cuda    # NVIDIA — native CUDA, best NVIDIA perf
cargo build --release --features rocm    # AMD — native ROCm/HIP, best AMD perf
cargo build --release --features metal   # macOS — Metal via wgpu
cargo build --release --features ndarray # CPU-only fallback (no GPU required)
./target/release/ai-worker --port 50051
```

### Tests
```bash
# Rust worker tests
cd worker && cargo test

# Python unit tests (run from repo root)
pytest coordinator/

# Integration / client smoke tests
python tests/test_client.py
python tests/cluster_chat.py
```
> Note: `coordinator/tests/` contains a 44-test suite covering `models`, `config`, and `router` modules.

### Linting
```bash
# Python: Black (line-length 100) + Ruff + MyPy strict
black --line-length 100 coordinator/
ruff check coordinator/
mypy coordinator/

# Rust
cargo fmt
cargo clippy
```

## Architecture

```
Client (REST) → Coordinator (FastAPI) → Workers (Rust/Burn) → GPU
                      │
              Prometheus / Grafana
```

**Coordinator modules** (`coordinator/`):
- `main.py` — FastAPI app entry point, lifespan, CORS, Prometheus ASGI mount
- `api.py` — FastAPI routes (`/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/v1/workers`, `/health`, `/metrics`)
- `coordinator.py` — Core orchestration logic
- `router.py` — Load balancing strategies: `least_load`, `round_robin`, `random`, `affinity`
- `discovery.py` — Worker discovery: static list, mDNS, Consul
- `models.py` — Model registry and lifecycle
- `config.py` — `Settings` (pydantic-settings), reads env vars and `coordinator.yaml`
- `monitoring.py` — Prometheus metrics definitions and helpers

**Worker modules** (`worker/src/`):
- `main.rs` — CLI entry point (clap), tokio runtime, gRPC server startup
- `worker.rs` — gRPC service handlers
- `gpu_manager.rs` — GPU detection and VRAM management
- `model_loader.rs` — Safetensors loading + quantization (FP16/INT8/INT4)
- `backend.rs` — Burn backend selection (wgpu/CUDA/ROCm/ndarray)
- `config.rs` — Worker config struct, reads `worker.toml`
- `error.rs` — Shared error types (`thiserror`)
- `metrics.rs` — Prometheus metrics definitions
- `parallelism.rs` — Pipeline/tensor/expert parallelism; `AllReduce<B>` trait; TP/PP/EP fully implemented

**Configuration files** (`config/`):
- `coordinator.yaml` — Server, discovery, routing, security, health checks
- `worker.toml` — GPU settings, inference defaults, KV-cache, parallelism
- `models.toml` — Model registry: architectures, memory requirements, HuggingFace IDs
- `prometheus.yml` — Prometheus scrape targets
- `alerts.yml` — Alertmanager alert rules
- `logging.yaml` — Structured logging configuration

## Key Development Patterns

### Adding a new model
1. Convert weights: `python scripts/convert_model.py <hf-repo> --output ./models/`
2. Add entry to `config/models.toml` (architecture, memory, HF repo ID, quantization flags)
3. Load via API: `POST /v1/models/load {"model_name": "your-model"}`

### Changing the gRPC interface
1. Edit `proto/cluster.proto`
2. Regenerate Python bindings: run `grpc_tools.protoc` (see coordinator Dockerfile for flags)
3. Rust bindings regenerate automatically via `worker/build.rs` on `cargo build`

### Environment variables (`.env` / Docker)
| Variable | Default | Purpose |
|---|---|---|
| `GPU_COUNT` | 1 | Number of GPU workers to spawn |
| `GPU_INDEX` | 0 | Which GPU device index |
| `HF_TOKEN` | — | HuggingFace token for gated models |
| `RUST_LOG` | info | Worker log level |
| `RUST_BACKTRACE` | 1 | Rust panic backtrace (set to `full` for verbose) |
| `GPU_VRAM_GB` | 6 | VRAM hint for memory planning |
| `WORKER_ID` | — | Unique worker identifier (auto-assigned if empty) |
| `GRPC_BASE_PORT` | 50051 | Base port for gRPC; multi-GPU workers increment from here |
| `METRICS_BASE_PORT` | 9091 | Base port for Prometheus metrics endpoint |

## CI / GitHub Actions

`.github/workflows/ci.yml` runs on every push/PR to `master` and `feature` branches:
- **Rust job**: `cargo check`, `cargo clippy`, `cargo test --features wgpu`
- **Python job**: `ruff check`, `black --check`, `mypy --strict`, `pytest coordinator/`

## Worker Model Architecture

**`worker/models/llama.rs`** key types and methods:
- `KvEntry<B>` / `KvCache<B>` — per-layer KV cache types (pub, shared with `parallelism.rs`)
- `Llama::prefill()` — full-sequence forward, returns `(Vec<f32>, KvCache<B>)`; one GPU→CPU transfer total
- `Llama::decode_step()` — single-token decode using KV cache, O(seq_cached) per step
- `Llama::tokenize_prompt()` — special-token-aware tokenization (`</s>`, `<s>`, `<|user|>`, `<|assistant|>`)
- `TextGeneration::generate()` — single `spawn_blocking` + mpsc channel; model cloned once outside loop

**`worker/src/parallelism.rs`** key types and functions:
- `TpKvCache<B>` — `Vec<Vec<KvEntry<B>>>` per-layer-per-shard KV for TP generation
- `AllReduce<B>` trait + `LocalAllReduce` — pluggable all-reduce (future NCCL impl drops in)
- `tensor_parallel_llama_prefill()` — full-sequence TP forward capturing per-shard KV
- `tensor_parallel_llama_decode_step()` — single-token TP decode with KV cache extension
- `pipeline_parallel_llama_forward()` — layer-chunk partitioning across stages
- `ParallelStrategy::ExpertParallel` — stub variant; returns error (requires MoE architecture)

## Git Conventions

When generating commit messages use Conventional Commits format (`feat`/`fix`/`chore`/`docs`) and reference the specific files changed. Keep the subject line under 72 characters. Always summarize the key changes across all modified files in the commit body.

## Docker & GPU

This project uses Docker with NVIDIA GPU support and Vulkan. Dockerfiles must include appropriate NVIDIA base images (`nvidia/cuda`) and Vulkan SDK layers (`libvulkan-dev`, `mesa-vulkan-drivers`). Always refer to existing Dockerfiles for patterns before creating new ones.

- `docker/Dockerfile.coordinator` — coordinator image (Python/FastAPI)
- Three worker Dockerfile variants — `Dockerfile.worker` (wgpu/Vulkan, universal default), `Dockerfile.worker.amd` (burn/rocm, max AMD perf), `Dockerfile.worker.nvidia` (burn/cuda, max NVIDIA perf)
- AMD passthrough: mount `/dev/kfd` + `/dev/dri`, add `group_add: [video, render]`
- NVIDIA passthrough: use `deploy.resources.reservations.devices` (NVIDIA Container Toolkit)
- Intel GPU works out of the box with `Dockerfile.worker` via Mesa Intel ANV Vulkan driver
- When modifying Dockerfiles, ensure GPU passthrough and Vulkan layers are preserved
- Always update `.env.example` when adding new environment variables to docker-compose or config
- GPU setup helper scripts: `scripts/setup_cuda.sh` (NVIDIA toolkit) and `scripts/setup_rocm.sh` (AMD ROCm)

## Languages & Build

Primary languages: Python (coordinator), Rust (worker), YAML (configs/CI), Shell scripts, Markdown docs.

- Always use Python type hints; keep YAML files consistent with existing formatting and indentation
- After modifying Rust files: `cd worker && cargo check`
- After modifying Python files: `python -m py_compile <file>`
- After modifying proto files: regenerate bindings (see "Changing the gRPC interface" above)
