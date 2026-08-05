# Docker Deployment for AI Cluster

Dockerfiles for the coordinator and worker, plus the compose stack at the repo root.

## Quick Start

```bash
cp .env.example .env
# Required in .env — without all three, no worker starts and the coordinator
# refuses to bind:
#   GRAFANA_ADMIN_PASSWORD=...
#   COORDINATOR_API_KEYS=...
#   COMPOSE_PROFILES=cuda-native   # or nvidia-vulkan / amd-vulkan / intel-vulkan / rocm-native
docker compose up -d --build
curl -H "Authorization: Bearer $COORDINATOR_API_KEYS" http://localhost:8000/health
```

No profile means no worker: the stack assumes no GPU vendor. Pick the one that
matches your hardware. See docs/deployment.md.

## Images

### ai-coordinator (`docker/Dockerfile.coordinator`)
- Multi-stage `python:3.14-slim` build; deps installed with **pip** from
  `coordinator/requirements.txt` (runtime-only).
- Regenerates the gRPC stubs during the build (grpc_tools.protoc + package-import fix).
- Runs as non-root; HTTP API + `/metrics` on port 8000; healthcheck on `/health`.

### ai-worker (`docker/Dockerfile.worker` — one file, three variants)

| Variant | Build | Backend feature | Base images |
|---|---|---|---|
| Universal (default) | `docker build -f docker/Dockerfile.worker .` | `wgpu` (Vulkan) | ubuntu:24.04 |
| AMD native | `--build-arg BACKEND=rocm --build-arg BUILDER_IMAGE=rocm/dev-ubuntu-24.04:6.2.2 --build-arg RUNTIME_IMAGE=rocm/dev-ubuntu-24.04:6.2.2 --build-arg BUILDER_EXTRA_PKGS="" --build-arg RUNTIME_EXTRA_PKGS="libvulkan1"` | `rocm` | rocm/dev-ubuntu-24.04 (amd64 only; 6.2.1 was never published) |
| NVIDIA native | `--build-arg BACKEND=cuda --build-arg BUILDER_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04 --build-arg RUNTIME_IMAGE=nvidia/cuda:13.0.3-runtime-ubuntu24.04 --build-arg BUILDER_EXTRA_PKGS="" --build-arg RUNTIME_EXTRA_PKGS="libcublas-13-0 libvulkan1 libx11-6 libxext6 libegl1" --build-arg LLAMASERVER_BACKEND=cuda` | `cuda` | nvidia/cuda 13.0.3 |

There is no `GPU_BACKEND` arg and no `hip` feature — the arg is `BACKEND`
and the AMD feature is `rocm`.

> `RUNTIME_EXTRA_PKGS` **replaces** the Dockerfile default rather than
> extending it, so the vendor-native rows must repeat the Vulkan loader (and,
> on NVIDIA, the injected ICD's own deps) or the worker enumerates no GPU.
> See docs/deployment.md.

> CUDA 13.0.3 (bumped from 12.6.3) — 12.6's nvcc predates Blackwell/`sm_121`,
> so it cannot build for GB10 (DGX Spark). Both `13.0.3-devel-ubuntu24.04` and
> `13.0.3-runtime-ubuntu24.04` publish `linux/amd64` **and** `linux/arm64`
> manifests. Note the runtime package name tracks the version:
> `libcublas-13-0`, not `libcublas-12-6`.

All three variants also bake in a `llama-server` binary (from a pinned
llama.cpp tag via `--build-arg LLAMASERVER_TAG=`) at `/usr/local/bin/llama-server`
with `LLAMASERVER_BINARY_PATH` preset, for `engine = "llamaserver"` agentic
models. Its GGML backend is `--build-arg LLAMASERVER_BACKEND=` — default
`vulkan` (portable); `cuda` on an NVIDIA `BUILDER_IMAGE` for CUDA offload
instead (the NVIDIA native row above sets this). Opt out of the compile
entirely with `--build-arg LLAMASERVER_SRC=llamaserver-none`. See
docs/deployment.md "llama-server for agentic serving".

> That stage needs **Node 20+** (copied in from `node:22-bookworm-slim`).
> llama.cpp's `tools/CMakeLists.txt` adds the embedded Web UI unconditionally
> whenever `LLAMA_BUILD_SERVER=ON` and links it into `llama-server`, so
> `-DLLAMA_BUILD_UI=OFF` does NOT skip it; building the UI needs npm because
> its `vite` pin requires Node 20.19+/22.12+. Do not remove the Node stage.

## GPU passthrough

- **AMD**: `devices: [/dev/kfd, /dev/dri]` + `group_add: [video, render]` (see `worker-amd-vulkan`/`worker-rocm-native`).
- **NVIDIA**: NVIDIA Container Toolkit + `deploy.resources.reservations.devices` (see `worker-nvidia-vulkan`/`worker-cuda-native`).
- **Intel**: default image works via Mesa ANV (see `worker-intel-vulkan`).

## Environment variables

Coordinator (see docs/configuration.md for the full table): `COORDINATOR_HOST`,
`COORDINATOR_PORT`, `COORDINATOR_DISCOVERY_METHOD` (static), `COORDINATOR_STATIC_WORKERS`,
`COORDINATOR_CORS_ORIGINS`.

Worker: `GPU_INDEX` (replica offset), `GRPC_BASE_PORT`/`METRICS_BASE_PORT`
(replica listens on base+index), optional explicit `GRPC_PORT`/`METRICS_PORT`,
`WORKER_ID`, `GPU_VRAM_GB`, `HF_TOKEN`, `RUST_LOG`, `RUST_BACKTRACE`, `GPU_IDS`,
`LOG_LEVEL`, `LOG_JSON`.

## Ports & networking (bridge network `ai-cluster-net`)

- coordinator: 8000 (API + metrics)
- worker: 50051+GPU_INDEX (gRPC), 9091+GPU_INDEX (metrics/health), 8081/8082
  (llama-server registry ports for `engine = "llamaserver"` models — published so
  the coordinator can proxy agentic HTTP inference to them; firewall to the LAN)
- prometheus: host 9099 → container 9090; grafana: 3000; open-webui: 8080

## Health checks

- coordinator: `GET /health` on 8000
- worker: `GET /health` on the metrics port (the HEALTHCHECK computes base+index itself)

## Troubleshooting

```bash
docker compose logs -f worker
docker compose exec worker vulkaninfo --summary   # universal image: GPU must be listed
docker compose exec coordinator curl -s http://worker:9091/health
```
