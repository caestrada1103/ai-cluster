# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Cursor, Copilot, etc.) when working with code in this repository.

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
# Run from the REPO ROOT — `cd coordinator && uvicorn main:app` breaks package imports.
pip install -r coordinator/requirements-dev.txt   # runtime+lint+test (runtime only: requirements.txt)
uvicorn coordinator.main:app --reload --host 0.0.0.0 --port 8000
```

### Worker (Rust)
```bash
cd worker
cargo build --release --features wgpu    # Universal — Vulkan/DX12/Metal, auto-detects AMD/NVIDIA/Intel (default)
cargo build --release --features cuda    # NVIDIA base kernels (runtime backend type is still Wgpu — native wiring planned)
cargo build --release --features rocm    # AMD base kernels (runtime backend type is still Wgpu — native wiring planned)
cargo build --release --features metal   # macOS — Metal via wgpu
# llama.cpp engine (GGUF models) — combine with a Burn backend feature:
cargo build --release --features wgpu,llamacpp                  # llama.cpp on CPU
cargo build --release --features wgpu,llamacpp,llamacpp-vulkan  # llama.cpp Vulkan offload
cargo build --release --features cuda,llamacpp,llamacpp-cuda    # llama.cpp CUDA offload
./target/release/ai-worker --port 50051
```
> There is no CPU-only/ndarray build; a GPU (or Vulkan software rasterizer) is required.

### Tests
```bash
# Rust worker tests (must pass a backend feature — CI uses wgpu)
cd worker && cargo test --features wgpu
cargo test --features wgpu config::tests::test_name   # single test by path
# llama.cpp engine (compiles llama.cpp via cmake; needs cmake + libclang)
cargo check --features llamacpp
cargo test --features llamacpp llamacpp_engine                  # unit tests
cargo test --features llamacpp -- --ignored llamacpp            # e2e (network, ~1 MiB GGUF)

# Python unit tests (run from repo root)
pytest coordinator/
pytest coordinator/tests/test_router.py                # single file
pytest coordinator/tests/test_router.py::test_name -v  # single test

# Integration / client smoke tests (require a running coordinator + worker)
python tests/test_client.py
python tests/cluster_chat.py
```
> Note: `coordinator/tests/` contains the unit suite covering `models`, `config`, `router`, and coordinator error paths (run `pytest coordinator/ -q` for the current count). Rust unit tests live inline (`worker/src/config.rs`, `worker/src/gpu_manager.rs`, `worker/models/common.rs`).

### Linting
```bash
# Python: Black (line-length 100) + Ruff + MyPy strict
black --line-length 100 coordinator/
ruff check coordinator/
mypy coordinator/

# Rust (both enforced in CI)
cargo fmt -- --check
cargo clippy -p ai-worker --features wgpu -- -D warnings
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
- `router.py` — Wired into worker selection: `least_load`, `round_robin`, `random`, `affinity` (session-keyed, TTL), `power_of_two`; per-worker circuit breakers
- `discovery.py` — Worker discovery: static list only (mDNS/broadcast/Consul are planned; selecting them fails fast)
- `models.py` — Model registry and lifecycle
- `config.py` — `Settings` (pydantic-settings), reads `COORDINATOR_*` env vars / `.env` only (no YAML config exists)
- `monitoring.py` — Prometheus metrics definitions and helpers

**Worker modules** (`worker/src/`):
- `main.rs` — CLI entry point (clap), tokio runtime, gRPC server startup
- `worker.rs` — gRPC service handlers
- `gpu_manager.rs` — GPU detection and VRAM management
- `model_loader.rs` — Safetensors loading as FP32 (quantization ≠ NONE is rejected; quantized inference planned); resolves HF repo via `LoadModelRequest.model_path`
- `llamacpp_engine.rs` — llama.cpp engine for GGUF models (feature `llamacpp`, crate `llama-cpp-2 =0.1.150`); implements `TextGeneration`
- `backend.rs` — `WorkerBackend` type alias (Wgpu; cuda/rocm features compile burn's native kernels but runtime selection is not wired — planned)
- `config.rs` — Worker config struct, reads `worker.toml`
- `error.rs` — Shared error types (`thiserror`)
- `metrics.rs` — Prometheus metrics definitions
- `parallelism.rs` — Tensor/pipeline/expert parallelism core functions; `AllReduce<B>` trait; standalone TP/PP functions compile and are correct but not yet wired to the gRPC service layer

**Configuration files** (`config/`):
- `worker.toml` — flat worker settings (ports, gpu_ids, concurrency, HF token fallback, llamacpp thread/gpu-layer defaults; unknown keys rejected)
- `models.toml` — Model registry: architectures, memory requirements, HuggingFace repo ids, per-model `engine` ("burn" | "llamacpp") + `[models.X.gguf]` source
- `prometheus.yml` — Prometheus scrape targets
- `alerts.yml` — Prometheus alert rules (written against the real metric names)
(The coordinator has no config file — `COORDINATOR_*` env vars only.)

## Key Development Patterns

### Adding a new model
1. Convert weights: `python scripts/convert_model.py <hf-repo> --output ./models/`
2. Add entry to `config/models.toml` (architecture, memory, HF repo ID, quantization flags)
3. Load via API: `POST /v1/models/load {"model_name": "your-model"}`

### Changing the gRPC interface
1. Edit `proto/cluster.proto`
2. Regenerate Python bindings: `python -m grpc_tools.protoc -I./proto --python_out=./coordinator/proto --grpc_python_out=./coordinator/proto ./proto/cluster.proto`, then re-apply the package import: `sed -i 's/^import cluster_pb2 as cluster__pb2$/import coordinator.proto.cluster_pb2 as cluster__pb2/' coordinator/proto/cluster_pb2_grpc.py`
3. Rust bindings regenerate automatically via `worker/build.rs` on `cargo build`

### Environment variables (`.env` / Docker)
| Variable | Default | Purpose |
|---|---|---|
| `GPU_INDEX` | 0 | Which GPU device index (compose replicas offset ports by it) |
| `GPU_IDS` | — | Comma-separated device indices for one worker process (`--gpu-ids`) |
| `HF_TOKEN` | — | HuggingFace token for gated models (wins over worker.toml `hf_token`) |
| `RUST_LOG` | info | Worker log level (wins over `LOG_LEVEL`) |
| `LOG_LEVEL` / `LOG_JSON` | info / off | clap-level log settings |
| `RUST_BACKTRACE` | 1 | Rust panic backtrace (set to `full` for verbose) |
| `GPU_VRAM_GB` | 8 (binary) / 6 (compose default) | VRAM hint when vendor tools can't report |
| `WORKER_ID` | — | Unique worker identifier (auto-assigned if empty) |
| `GRPC_PORT` / `METRICS_PORT` | 50051 / 9091 | Explicit ports the binary reads (CLI/env > worker.toml > default) |
| `GRPC_BASE_PORT` / `METRICS_BASE_PORT` | 50051 / 9091 | Docker entrypoint only: replica port = base + GPU_INDEX (the bare binary ignores BASE vars) |

## CI / GitHub Actions

`.github/workflows/ci.yml` runs on every push/PR to `master` and `feature` branches:
- **Rust job**: `cargo check`, `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --features wgpu`
- **Python job**: `ruff check`, `black --check`, `mypy` (strict; pydantic plugin; `coordinator/proto/` excluded), `pytest coordinator/`

## Worker Model Architecture

**`common.rs`**: `build_causal_bias<B>()` (O(seq²) once per prefill, shared by all model prefills), `RotaryEmbedding::apply()` (panic guard on bounds; cos/sin are [max_seq_len, head_dim/2]), `top_k_top_p_sample()` (real multinomial sampling via `rand::StdRng`, seedable from `InferenceRequest.seed`; temperature < 0.01 → greedy argmax in callers), `load_eos_ids()` (eos ids from (generation_)config.json), `swiglu()`, `repeat_kv()`.

**`mod.rs`**: `TextStream`, `TextGeneration` trait (`generate(..., seed: Option<u64>)`), `ModelInstance` (holds `Arc<Mutex<dyn TextGeneration>>`; increments `inference_count`; errors when no model is attached); re-exports `KvEntry<B>` / `KvCache<B>` from `llama.rs` for use by all model modules.

**`llama.rs`**: Reference implementation. `KvEntry<B>` = `(Tensor<B,4>, Tensor<B,4>)` per layer. `LlamaAttention::forward()` accepts pre-built `causal_bias`. `Llama::prefill()` → `(Vec<f32>, KvCache<B>)`; `decode_step()` O(seq_cached). `TextGeneration::generate()` — single `spawn_blocking` + mpsc channel, model cloned once.

**`qwen.rs`**: Qwen2/2.5 family — Llama-style GQA + RoPE + SwiGLU **plus** optional q/k/v biases (`attention_bias` from config.json) and explicit `head_dim` support. Qwen3 checkpoints are rejected at load (per-head q/k-norm unimplemented). Config is always built from the checkpoint's config.json. `QwenAttention` has `forward_prefill()` (returns `KvEntry`) and `forward_decode()`.

**`deepseek.rs`**: MoE with V1/V2-style sparse top-k routing (CPU sort → GPU weight broadcast; V3 sigmoid/group routing + MLA NOT implemented). `DeepSeekConfig` is built from the checkpoint's config.json (`n_routed_experts` supported); dense DeepSeek checkpoints (no `mlp.experts.*` keys) load through the Llama record path. EOS ids come from (generation_)config.json.

**`mistral.rs`**: Sliding window causal mask; query `i` attends to `[max(0,i-window+1), i]`.

**`model_loader.rs`**: Async safetensors load; spawn_blocking for dtype conversion (all weights land as FP32). Architectures: `"llama"`, `"qwen"`, `"deepseek"` (detected via `config.json` `"architectures"`). Downloads use `LoadModelRequest.model_path` (HF repo id resolved by the coordinator) with `model_name` as the registry key. Per-model loading lock + GPU reservation rollback on failure; `unload()` releases reservations. `create_deepseek_record()` loads `N` experts from `model.layers.{i}.mlp.experts.{j}.*` when present.

**`llamacpp_engine.rs`** (feature `llamacpp`): `LlamaCppEngine::load(path, n_gpu_layers, n_ctx, n_threads)`; shared `Arc<LlamaModel>`, per-call `LlamaContext` inside one `spawn_blocking` + mpsc (same streaming shape as `llama.rs`); sampler chain from request temperature/top_k/top_p (greedy < 0.01), seedable from `InferenceRequest.seed`; EOS from GGUF metadata. Routing: the coordinator sends `engine`/`gguf_repo_id`/`gguf_file`/`n_gpu_layers`/`n_ctx` in the existing `ModelConfig.metadata` map (zero proto change); `model_loader.rs::gguf_spec_from_metadata` parses it and `load_llamacpp_model` downloads the GGUF via hf-hub, reporting the file size as memory.

**`gpu_manager.rs`**: O(1) memory tracking via `AtomicU64` with tagged `allocate_memory`/`free_memory`; telemetry (util/temp/power) refreshed from `nvidia-smi` at scrape/health time (3s timeout); CPU adapters dropped whenever a real GPU exists.

**`worker.rs`**: `active_requests` = `Arc<DashMap<String, Instant>>` (RAII `ActiveGuard` cleanup); `loaded_models` = `Arc<RwLock<HashMap<String, ModelInstance>>>`; `infer` bounded by a `max_concurrent_requests` semaphore (RESOURCE_EXHAUSTED beyond it); finish reasons: Stop / Length / Timeout / Error.

**`parallelism.rs`**: `TpKvCache<B>`, `AllReduce<B>` + `LocalAllReduce`. `tensor_parallel_llama_prefill/decode_step`, `pipeline_parallel_llama_forward`. `ParallelStrategy` enum (ExpertParallel stub). TP/PP standalone — not yet wired to gRPC.

## Git Conventions

When generating commit messages use Conventional Commits format (`feat`/`fix`/`chore`/`docs`) and reference the specific files changed. Keep the subject line under 72 characters. Always summarize the key changes across all modified files in the commit body.

## Docker & GPU

This project uses Docker with NVIDIA GPU support and Vulkan. Dockerfiles must include appropriate NVIDIA base images (`nvidia/cuda`) and Vulkan SDK layers (`libvulkan-dev`, `mesa-vulkan-drivers`). Always refer to existing Dockerfiles for patterns before creating new ones.

- `docker/Dockerfile.coordinator` — coordinator image (Python/FastAPI)
- `docker/Dockerfile.worker` — ONE parameterized worker image: default wgpu/Vulkan; `--build-arg BACKEND=rocm|cuda` with matching `BUILDER_IMAGE`/`RUNTIME_IMAGE`/`*_EXTRA_PKGS` args for the vendor variants (see the file header and docker-compose.yml comments)
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
