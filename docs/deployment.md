# Deployment Guide

## 1. Docker Compose (recommended)

```bash
git clone https://github.com/caestrada1103/ai-cluster.git
cd ai-cluster
cp .env.example .env
# REQUIRED: set GRAFANA_ADMIN_PASSWORD in .env (compose refuses to start Grafana without it)
# Set HF_TOKEN for gated models (Llama 3, ...)
docker compose up -d --build
curl http://localhost:8000/health
```

Services and host ports:

| Service | Port | Notes |
|---|---|---|
| coordinator | 8000 | REST API + `/metrics` (Prometheus scrapes 8000 — there is no :9090 coordinator port) |
| worker | 50051 (gRPC), 9091 (`/metrics`, `/health`, `/live`) | replica N listens on base+N |
| prometheus | 9099 → container 9090 | |
| grafana | 3000 | admin / $GRAFANA_ADMIN_PASSWORD |
| open-webui | 8080 | chat UI against the coordinator |

### GPU variants

One parameterized `docker/Dockerfile.worker`:

- **Universal wgpu/Vulkan** (default): AMD passthrough — `/dev/kfd` + `/dev/dri`,
  `group_add: [video, render]`; NVIDIA — Container Toolkit injects the Vulkan ICD;
  Intel — works out of the box (Mesa ANV).
- **AMD native ROCm**: build arg `BACKEND=rocm` with the `rocm/dev-ubuntu-24.04:6.2.1` images.
- **NVIDIA native CUDA**: build arg `BACKEND=cuda` with the `nvidia/cuda:12.6.3-*-ubuntu24.04` images.

See the commented blocks in `docker-compose.yml` for copy-paste service definitions.
Cargo features are `wgpu` (default) / `rocm` / `cuda` — there is no `hip` feature.

### Multi-GPU hosts

Run one worker service per GPU (see the `worker-gpu-N` commented blocks):
set `GPU_INDEX=N` per service (ports become 50051+N / 9091+N), give each its own
GPU passthrough block, add each address to `COORDINATOR_STATIC_WORKERS`, and add
each metrics target to `config/prometheus.yml`.

## 2. Native (no Docker)

Prerequisites: Python 3.10+, Rust 1.70+, `protobuf-compiler`, GPU drivers
(ROCm 6.0+ / CUDA 12.1+ / any Vulkan driver for wgpu).

```bash
# Coordinator — run from the REPO ROOT (module path matters)
python3 -m venv venv && source venv/bin/activate
pip install -r coordinator/requirements.txt
uvicorn coordinator.main:app --host 0.0.0.0 --port 8000

# Worker (second terminal)
cd worker
cargo build --release --features wgpu     # or: rocm / cuda
./target/release/ai-worker --port 50051 --gpu-ids 0
```

Point the coordinator at workers with
`COORDINATOR_STATIC_WORKERS=localhost:50051` (comma-separated for more), or
register at runtime: `curl -X POST localhost:8000/v1/workers/manual -H "Content-Type: application/json" -d '["localhost:50051"]'`.

## 3. Multi-machine cluster

- Run the coordinator on one host; a worker per GPU host.
- `COORDINATOR_STATIC_WORKERS=host-a:50051,host-b:50051`.
- Open TCP 50051 (gRPC) worker-side and 9091 for Prometheus scrapes; 8000 coordinator-side.
- Traffic is plaintext gRPC/HTTP — deploy on a trusted network or behind your own TLS proxy
  (built-in TLS/auth is planned, not implemented).

## 4. Health checks

- Coordinator: `GET /health` → `{"status": "healthy", "workers": N}` (always 200; `starting` before ready).
- Worker (metrics port): `GET /health` → `OK`, `GET /live` → `ALIVE`.
- There are no `/health/live|ready|startup` coordinator routes.

## 5. Upgrades

```bash
git pull
docker compose build
docker compose up -d      # brief downtime; rolling/canary orchestration is not built in
```

## Planned (not shipped)

Kubernetes manifests/Helm, systemd unit files, backup/restore + canary/blue-green
scripts, Vault/Redis/MinIO/Elastic integrations, TLS/mTLS, Jaeger tracing.
