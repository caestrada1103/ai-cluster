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
- `parallelism.rs` — Tensor/pipeline/expert parallelism core functions; `AllReduce<B>` trait; standalone TP/PP functions compile and are correct but not yet wired to the gRPC service layer

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

**`common.rs`**: `build_causal_bias<B>()` (O(seq²) once per prefill, passed to all layers), `RotaryEmbedding::apply()` (panic guard on bounds), `top_k_top_p_sample()` (single-pass running sum), `swiglu()`, `repeat_kv()`.

**`mod.rs`**: `TextStream`, `TextGeneration` trait, `ModelInstance` (holds `Arc<Mutex<dyn TextGeneration>>`); re-exports `KvEntry<B>` / `KvCache<B>` from `llama.rs` for use by all model modules.

**`llama.rs`**: Reference implementation. `KvEntry<B>` = `(Tensor<B,4>, Tensor<B,4>)` per layer. `LlamaAttention::forward()` accepts pre-built `causal_bias`. `Llama::prefill()` → `(Vec<f32>, KvCache<B>)`; `decode_step()` O(seq_cached). `TextGeneration::generate()` — single `spawn_blocking` + mpsc channel, model cloned once.

**`qwen.rs`** (new): Qwen3-Coder-32B — identical architecture to Llama3 (GQA + RoPE + SwiGLU). Config: 64 layers, hidden 5120, 40/8 GQA heads, vocab 151936, ctx 131072, rope_theta 1e6. Special tokens: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`. Weight names identical to Llama3 HF layout. `QwenAttention` has `forward_prefill()` (returns `KvEntry`) and `forward_decode()`.

**`deepseek.rs`**: MoE with sparse top-k routing (CPU sort → GPU weight broadcast). Added: `deepseek_v3()` config (61 layers, hidden 7168, 128 heads MHA, 256 experts / 8 active, ctx 163840); `forward_prefill()` / `forward_decode()` on attention and layer; `prefill()` / `decode_step()` / `TextGeneration` on model. `DeepSeek::new()` now accepts `tokenizer_path: &Path`; EOS: `<|EOT|>` → `</s>`.

**`mistral.rs`**: Sliding window causal mask; query `i` attends to `[max(0,i-window+1), i]`.

**`model_loader.rs`**: Async safetensors load; spawn_blocking for dtype conversion. Architectures: `"llama"`, `"qwen"`, `"deepseek"` (detected via `config.json` `"architectures"` field). `create_qwen_record()` — same weight paths as Llama3. `create_deepseek_record()` — loads `N` experts from `model.layers.{i}.mlp.experts.{j}.*`. DeepSeek variant: name `"v3"` / `"67b"` / else.

**`gpu_manager.rs`**: O(1) memory tracking via `AtomicU64`; `nvidia-smi`/`rocm-smi` with 3s timeout.

**`worker.rs`**: `active_requests` = `Arc<DashMap<String, Instant>>`; `loaded_models` = `Arc<RwLock<HashMap<String, ModelInstance>>>`.

**`parallelism.rs`**: `TpKvCache<B>`, `AllReduce<B>` + `LocalAllReduce`. `tensor_parallel_llama_prefill/decode_step`, `pipeline_parallel_llama_forward`. `ParallelStrategy` enum (ExpertParallel stub). TP/PP standalone — not yet wired to gRPC.

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
