# Docker Deployment for AI Cluster

Dockerfiles for the coordinator and worker, plus the compose stack at the repo root.

## Quick Start

```bash
cp .env.example .env            # set GRAFANA_ADMIN_PASSWORD (required) and HF_TOKEN
docker compose up -d --build
curl http://localhost:8000/health
```

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
| AMD native | `--build-arg BACKEND=rocm --build-arg BUILDER_IMAGE=rocm/dev-ubuntu-24.04:6.2.1 --build-arg RUNTIME_IMAGE=rocm/dev-ubuntu-24.04:6.2.1 --build-arg BUILDER_EXTRA_PKGS="" --build-arg RUNTIME_EXTRA_PKGS=""` | `rocm` | rocm/dev-ubuntu-24.04 |
| NVIDIA native | `--build-arg BACKEND=cuda --build-arg BUILDER_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04 --build-arg RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04 --build-arg BUILDER_EXTRA_PKGS="" --build-arg RUNTIME_EXTRA_PKGS="libcublas-12-6"` | `cuda` | nvidia/cuda 12.6.3 |

There is no `GPU_BACKEND` arg and no `hip` feature — the arg is `BACKEND`
and the AMD feature is `rocm`.

All three variants also bake in a `llama-server` binary (Vulkan, from a pinned
llama.cpp tag via `--build-arg LLAMASERVER_TAG=`) at `/usr/local/bin/llama-server`
with `LLAMASERVER_BINARY_PATH` preset, for `engine = "llamaserver"` agentic
models. Opt out (skip the llama.cpp compile) with
`--build-arg LLAMASERVER_SRC=llamaserver-none`. See
docs/deployment.md "llama-server for agentic serving".

## GPU passthrough

- **AMD**: `devices: [/dev/kfd, /dev/dri]` + `group_add: [video, render]` (see the active worker service).
- **NVIDIA**: NVIDIA Container Toolkit + `deploy.resources.reservations.devices` (see the commented block).
- **Intel**: default image works via Mesa ANV.

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
