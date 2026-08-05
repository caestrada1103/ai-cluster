# Deployment Guide

The project's target scenario is a single machine with one or more consumer
NVIDIA/AMD GPUs (~8–16 GB VRAM), including cards that would otherwise sit
idle, running a quantized GGUF model through the **llama.cpp engine**
(`--features llamacpp`). That engine is **opt-in**, not the default cargo
build — every command below shows how to add it. Multi-machine clusters
(section 3) are supported but are the "also scales to" case, not the
everyday deployment.

## 1. Docker Compose (recommended)

```bash
git clone https://github.com/caestrada1103/ai-cluster.git
cd ai-cluster
cp .env.example .env
# REQUIRED: set GRAFANA_ADMIN_PASSWORD in .env (compose refuses to start Grafana without it)
# REQUIRED: set COORDINATOR_API_KEYS in .env (a comma-separated secret list, e.g.
#   `openssl rand -hex 32`) — the coordinator's REST API port (8000) IS published to
#   the host by this compose file, and the coordinator now REFUSES TO START with a
#   non-loopback bind and no keys configured (secure-by-default; see coordinator/main.py).
# REQUIRED: set COMPOSE_PROFILES in .env to match your GPU (see "GPU variants" below) —
#   with no profile set, `docker compose up` starts everything EXCEPT a worker.
# Set HF_TOKEN for gated models (Llama 3, ...)
docker compose up -d --build
curl -H "Authorization: Bearer <your COORDINATOR_API_KEYS value>" http://localhost:8000/health
```

Services and host ports (default — nothing else is published; see "Exposing
worker/Open-WebUI ports" below for the opt-in override):

| Service | Port | Notes |
|---|---|---|
| coordinator | 8000 | REST API + `/metrics`, gated by `COORDINATOR_API_KEYS` (Prometheus scrapes 8000 — there is no :9090 coordinator port) |
| worker | none by default | gRPC (50051)/metrics (9091)/llama-server (8081/8082) are reachable by the coordinator CONTAINER over the private compose network, but not published to the host |
| prometheus | 9099 → container 9090 | |
| grafana | 3000 | admin / $GRAFANA_ADMIN_PASSWORD |
| open-webui | none by default | chat UI against the coordinator; not published to the host |

### Exposing worker/Open-WebUI ports (opt in)

The base `docker-compose.yml` is secure-by-default: no worker port or
Open-WebUI's port is published to the host. That is correct for the common single-host case (coordinator and
worker(s) only need to reach each other over the private compose network).
Layer `docker-compose.expose-ports.yml` when something OUTSIDE this compose
project needs to reach one of these directly — e.g. a LAN cluster where the
coordinator runs on a different host (section 3), or you want to browse to
Open-WebUI from another machine:

```bash
docker compose -f docker-compose.yml -f docker-compose.expose-ports.yml \
  --profile cuda-native up -d
```

Every worker service in `docker-compose.yml` already sets
`WORKER_GRPC_BIND_HOST=${WORKER_GRPC_BIND_HOST:-0.0.0.0}` and
`LLAMASERVER_BIND_HOST=${LLAMASERVER_BIND_HOST:-0.0.0.0}` — that's what lets
the coordinator container reach a worker container over the private network
even before this overlay publishes anything to the host; layering
`docker-compose.expose-ports.yml` only changes what's reachable from OUTSIDE
docker. Set `WORKER_GRPC_AUTH_TOKEN` in `.env` before doing this beyond a
trusted LAN behind a firewall/VPN — the worker gRPC port has no other
authentication (see `worker/src/grpc_auth.rs`).

### GPU variants

One parameterized `docker/Dockerfile.worker` builds five worker profiles in
`docker-compose.yml` — no vendor is assumed by default, so pick the one
profile matching your hardware via `COMPOSE_PROFILES` in `.env` (or
`--profile <name>` on the CLI). Whichever one is active, it joins the compose
network as both its own service name and the `worker` alias, so
`COORDINATOR_STATIC_WORKERS=["worker:50051"]` keeps working unchanged. Only
activate **one** single-GPU profile at a time (see "Multi-GPU hosts" below
for running several workers at once).

| Profile | Service | `BACKEND` | Vendor | GPU passthrough | Perf |
|---|---|---|---|---|---|
| `amd-vulkan` | `worker-amd-vulkan` | `wgpu` (default) | AMD | `/dev/kfd` + `/dev/dri`, `group_add: [video, render]` | good |
| `nvidia-vulkan` | `worker-nvidia-vulkan` | `wgpu` (default) | NVIDIA | NVIDIA Container Toolkit (`deploy.resources.reservations.devices`) | good |
| `intel-vulkan` | `worker-intel-vulkan` | `wgpu` (default) | Intel | `/dev/dri`, `group_add: [video, render]` (Mesa ANV — works out of the box) | good |
| `rocm-native` | `worker-rocm-native` | `rocm` | AMD | `/dev/kfd` + `/dev/dri`, `group_add: [video, render]` | best |
| `cuda-native` | `worker-cuda-native` | `cuda` | NVIDIA (incl. DGX Spark, sm_121) | NVIDIA Container Toolkit | best |

The three `*-vulkan` profiles all build the exact same universal wgpu/Vulkan
image — Vulkan auto-detects the vendor at runtime — and differ only in which
device nodes get passed through. `rocm-native`/`cuda-native` build the
vendor's native Burn backend instead (`BACKEND=rocm|cuda`) for the best
per-vendor performance. Cargo features are `wgpu` (default) / `rocm` / `cuda`
— there is no `hip` feature.

`cuda-native` builds on `nvidia/cuda:13.0.3-devel-ubuntu24.04` /
`13.0.3-runtime-ubuntu24.04` (bumped 2026-08 from 12.6.3, whose nvcc predates
sm_121/Grace-Blackwell support) — both amd64 and arm64 manifests are
published for these tags, so the same block builds on an RTX 3080 desktop
(x86_64) or a DGX Spark (aarch64). `worker/build.rs` links NCCL (a
multi-GPU collective-comm library) only when it's found on the link-search
path — linking it unconditionally broke single-GPU CUDA hosts, notably DGX
Spark, whose CUDA 13 toolkit ships `cudart`/`cublas` but no `libnccl`,
failing only at final link after a long build.

A worker's `max_loaded_models = 1` (see `config/worker.toml`) is meant for
hosts with unified CPU+GPU memory like DGX Spark, where the operator wants
only one model resident at a time — loading a new one evicts the
oldest-loaded first. `0` (the default) is unlimited, for deployments with
headroom to keep several models warm.

#### Stage 1 build layout

The repo root is a cargo workspace (virtual manifest, `worker` as its only
member). Stage 1 of `docker/Dockerfile.worker` reproduces that layout inside
`/build` (root `Cargo.toml`/`Cargo.lock` plus `worker/` and `proto/`) instead
of flattening `worker/Cargo.toml` to `/build/Cargo.toml` — the latter breaks
`version.workspace = true`-style inheritance, since there is no workspace root
above it, and the build fails with `error inheriting 'edition' from workspace
root manifest`. The build then runs from `/build/worker`, since a virtual
workspace root rejects `cargo build --features`; output still lands in the
workspace's shared `/build/target`, which is what the runtime stage copies
from.

#### Vendor-native RUNTIME_EXTRA_PKGS override

`RUNTIME_EXTRA_PKGS` (Stage 3) defaults to `libvulkan1 mesa-vulkan-drivers
vulkan-tools`, because Stage 2's `llama-server` builds against
`LLAMASERVER_BACKEND` (default `vulkan`; `cuda` on NVIDIA builder images —
see section 4 below), and the main `ai-worker` binary's own GPU
detection (`gpu_manager.rs`) always goes through Vulkan regardless of that
choice, so every variant still needs the loader. Both `worker-cuda-native`
and `worker-rocm-native` in `docker-compose.yml` set this build arg to their
own vendor-specific package list, which *replaces* the default rather than
extending it, so each must list the loader itself (`libvulkan1`) too. `mesa-vulkan-drivers` and
`vulkan-tools` are deliberately left off both vendor-native variants — they
are software/debug packages the vendor's own stack doesn't need.

On `worker-cuda-native`, `libvulkan1` alone is not enough. The NVIDIA
Container Toolkit injects the ICD `libGLX_nvidia.so.0` and its EGL
counterpart, but not their own link-time dependencies: `libX11.so.6` /
`libXext.so.6` (`libx11-6`, `libxext6`), and the GLVND dispatcher
`libEGL.so.1` (`libegl1`), without which the ICD's `vk_icdGetInstanceProcAddr`
never resolves. The loader then reports `Found no drivers!` and enumerates
zero devices — silently, since an unloadable driver looks the same as none.
That broke two things at once: `llama-server --list-devices` found nothing,
and `gpu_manager.rs`, which detects GPUs through the same Vulkan path,
registered the worker as `CPU Fallback` with 0 GB VRAM. Hence
`libcublas-13-0 libvulkan1 libx11-6 libxext6 libegl1`, verified by
`llama-server --list-devices` printing `Vulkan0: NVIDIA GB10` and by the
worker registering `NVIDIA GB10 (vulkan)` with real VRAM.

`worker-rocm-native` was left at `libvulkan1` only — AMD's Vulkan ICD (RADV,
Mesa-based) does not share NVIDIA's GLVND/EGL packaging and has no reason to
need `libx11-6`/`libxext6`/`libegl1`, but this repo has no AMD GPU to verify
that empirically, and `rocm-native` is unbuildable on this host regardless
(see below) — left untested rather than guessed.

#### ROCm base image tag

`rocm/dev-ubuntu-24.04:6.2.1` is not a published tag (`docker manifest
inspect` returns `no such manifest`). The pin in `docker-compose.yml`
(`worker-rocm-native`), `docker/Dockerfile.worker`'s header comment, and
`docker/README.md`'s variants table now point at `6.2.2`, the closest
existing tag, confirmed via `docker manifest inspect`. That tag (like the
rest of the `rocm/dev-ubuntu-24.04` repository) publishes `linux/amd64` only,
so `rocm-native` remains unbuildable on aarch64 hosts such as DGX Spark for
that separate, pre-existing reason.

#### Per-GPU quick start

```bash
# RTX 3080 (NVIDIA, universal Vulkan image — good perf, simplest path)
echo "COMPOSE_PROFILES=nvidia-vulkan" >> .env
docker compose up -d --build

# RTX 3080 (NVIDIA, native CUDA build — best perf)
echo "COMPOSE_PROFILES=cuda-native" >> .env
docker compose up -d --build

# RX 9060XT (AMD, universal Vulkan image — good perf, simplest path)
echo "COMPOSE_PROFILES=amd-vulkan" >> .env
docker compose up -d --build

# RX 9060XT (AMD, native ROCm build — best perf)
echo "COMPOSE_PROFILES=rocm-native" >> .env
docker compose up -d --build

# DGX Spark (Grace-Blackwell, aarch64, sm_121) — native CUDA build recommended
echo "COMPOSE_PROFILES=cuda-native" >> .env
docker compose up -d --build
# or, as a vendor-neutral fallback (also multi-arch, builds fine on aarch64):
echo "COMPOSE_PROFILES=nvidia-vulkan" >> .env
docker compose up -d --build
```

#### Enabling the llama.cpp/GGUF engine (recommended for consumer GPUs)

The default image builds the Burn/wgpu path only. To also compile the
llama.cpp engine — the one that actually loads quantized GGUF models — add
the `WORKER_FEATURES` build arg to whichever worker service matches your
profile (e.g. `worker-nvidia-vulkan:` for `COMPOSE_PROFILES=nvidia-vulkan`):

```yaml
# docker-compose.yml, e.g. under the worker-nvidia-vulkan: service, build.args:
build:
  context: .
  dockerfile: docker/Dockerfile.worker
  args:
    WORKER_FEATURES: "llamacpp,llamacpp-vulkan"   # Vulkan offload, NVIDIA or AMD
    # or, under worker-cuda-native: (already BACKEND: cuda), just add:
    # WORKER_FEATURES: "llamacpp,llamacpp-cuda"
```

`llamacpp-vulkan`/`llamacpp-cuda` select llama.cpp's own GPU kernels;
`llamacpp` alone still runs (CPU-only GGUF inference). See
[docs/configuration.md](configuration.md) for pointing a registry entry at
a GGUF file once the worker is built this way.

### Multi-GPU hosts

Run one worker service per GPU (see the `worker-gpu-N` commented blocks,
`profiles: ["multi-gpu"]`, brought up with
`COMPOSE_PROFILES=multi-gpu docker compose up -d`): set `GPU_INDEX=N` per
service (ports become 50051+N / 9091+N), give each its own GPU passthrough
block (base it on whichever single-GPU profile above matches that card's
vendor — mixed-vendor rigs just combine blocks of different `BACKEND`s), add
each address to `COORDINATOR_STATIC_WORKERS`, and add each metrics target to
`config/prometheus.yml`. Don't also activate one of the single-GPU profiles
in the same project — both would fight over the shared `worker` network
alias and the default GRPC/metrics ports.

## 2. Native (no Docker)

Prerequisites: Python 3.10+, Rust 1.70+, `protobuf-compiler`, GPU drivers
(ROCm 6.0+ / CUDA 12.1+, or 13.0+ for sm_121 / Grace-Blackwell hosts like DGX
Spark / any Vulkan driver for wgpu).

```bash
# Coordinator — run from the REPO ROOT (module path matters)
python3 -m venv venv && source venv/bin/activate
pip install -r coordinator/requirements.txt
# Loopback-only, single-machine (secure by default — no COORDINATOR_API_KEYS needed):
uvicorn coordinator.main:app --host 127.0.0.1 --port 8000
# Reachable from elsewhere on your LAN — the coordinator refuses to start with
# a non-loopback --host and no COORDINATOR_API_KEYS configured:
export COORDINATOR_API_KEYS=$(openssl rand -hex 32)   # or set it in .env
uvicorn coordinator.main:app --host 0.0.0.0 --port 8000

# Worker (second terminal)
cd worker
cargo build --release --features wgpu     # Burn engine only (default; FP32, no quantization)
# Recommended for consumer GPUs — also compile the llama.cpp/GGUF engine:
cargo build --release --features wgpu,llamacpp                  # llama.cpp on CPU
cargo build --release --features wgpu,llamacpp,llamacpp-vulkan  # llama.cpp Vulkan offload (NVIDIA or AMD)
cargo build --release --features cuda,llamacpp,llamacpp-cuda    # llama.cpp CUDA offload (NVIDIA)
# Loopback-only bind is the worker binary's default too (grpc_bind_host in
# worker.toml, or the built-in default when no config file is present).
./target/release/ai-worker --port 50051 --gpu-ids 0
# Coordinator on a DIFFERENT host: set grpc_bind_host = "0.0.0.0" (and
# llamaserver_bind_host, for engine="llamaserver" models) in worker.toml —
# or export WORKER_GRPC_BIND_HOST=0.0.0.0 / LLAMASERVER_BIND_HOST=0.0.0.0,
# which win over the file — and set WORKER_GRPC_AUTH_TOKEN, since the worker
# gRPC port has no other authentication.
```

`wgpu` is the default Burn backend feature (`rocm`/`cuda` are the native
Burn alternatives); `llamacpp` is always opt-in on top of one of those. See
[AGENTS.md](../AGENTS.md) for the full feature matrix and
[configuration.md](configuration.md) for pointing a model at a GGUF file.

Point the coordinator at workers with
`COORDINATOR_STATIC_WORKERS=localhost:50051` (comma-separated for more), or
register at runtime: `curl -X POST localhost:8000/v1/workers/manual -H "Content-Type: application/json" -d '["localhost:50051"]'`.

## 3. Multi-machine cluster

- Run the coordinator on one host; a worker per GPU host.
- `COORDINATOR_STATIC_WORKERS=host-a:50051,host-b:50051`.
- Open TCP 50051 (gRPC) worker-side and 9091 for Prometheus scrapes; 8000 coordinator-side.
- Traffic is plaintext gRPC/HTTP — deploy on a trusted network or behind your own TLS proxy
  (built-in TLS/auth is planned, not implemented).

## 4. llama-server for agentic serving (engine = "llamaserver")

Models with `engine = "llamaserver"` in `config/models.toml` (the 16 GB AMD
fleet's **Devstral Small 2 24B** and **Qwen3-Coder-30B-A3B**) are served
differently from the in-process `llamacpp` engine: the worker spawns and
**supervises one `llama-server` child process per such model**, and the
coordinator **proxies** agentic HTTP inference (OpenAI/Anthropic tool calling,
streaming `tool_calls`, `/v1/messages`) straight through to it. This is the only
path that gets real tool calling — the in-process engine cannot do it.

Unlike the in-process engine (compiled by cargo via `--features llamacpp`), the
`llama-server` binary is **not** built by cargo. Each worker host needs it
present, located via worker config `llamaserver_binary_path` (env
`LLAMASERVER_BINARY_PATH` wins; default `llama-server` on `PATH`).

### Docker (built in)

The worker image bakes it in — nothing to do. `docker/Dockerfile.worker` builds
only the `llama-server` target from a **pinned** llama.cpp release, statically
linked, into `/usr/local/bin/llama-server` and sets `LLAMASERVER_BINARY_PATH`.

The GGML backend it's built with is `--build-arg LLAMASERVER_BACKEND=`, default
`vulkan` (portable — matches the universal wgpu image, and runs as-is since
that image already ships the Vulkan loader). Pass `LLAMASERVER_BACKEND=cuda`
when also building against an NVIDIA `BUILDER_IMAGE` (e.g. `worker-cuda-native`
in `docker-compose.yml`, which sets this by default) so the bundled
`llama-server` offloads via CUDA instead of Vulkan. This is independent of the
main `ai-worker` binary's own GPU detection, which always uses Vulkan (see
"Vendor-native RUNTIME_EXTRA_PKGS override" above) regardless of
`LLAMASERVER_BACKEND`.

To build a pure-Burn / CPU-only image *without* the (heavy) llama.cpp compile,
pass `--build-arg LLAMASERVER_SRC=llamaserver-none`.

### Bare-metal Linux

Build `llama-server` once from the same pinned tag, then point the worker at it:

```bash
# Build deps (Ubuntu 24.04; glslc + spirv-headers compile the Vulkan shaders;
# libcurl is required by llama.cpp's default build unless you add -DLLAMA_CURL=OFF):
sudo apt-get install -y build-essential cmake git libvulkan-dev glslc spirv-headers libcurl4-openssl-dev
# Runtime deps (Vulkan loader + AMD/Intel Mesa drivers):
sudo apt-get install -y libvulkan1 mesa-vulkan-drivers
# Node.js 20.19+/22.12+ (NOT Ubuntu 24.04's apt nodejs, which is 18.x — too
# old) — required by `-DLLAMA_BUILD_SERVER=ON` below, see the note underneath:
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

git clone --depth 1 --branch b9941 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server -j

# Point the worker at the in-tree binary (simplest — keeps its shared libs):
export LLAMASERVER_BINARY_PATH="$PWD/build/bin/llama-server"
```

> **Node.js is required, not optional.** `-DLLAMA_BUILD_SERVER=ON` pulls in
> llama.cpp's embedded Web UI unconditionally (`tools/CMakeLists.txt` does
> `add_subdirectory(ui)` whenever `LLAMA_BUILD_SERVER=ON` — it does **not**
> gate on `-DLLAMA_BUILD_UI`, and `tools/server/CMakeLists.txt` links the
> resulting `llama-ui` static lib into the `llama-server` binary). The UI's
> `package.json` pins vite 7, which needs Node 20.19+/22.12+. Without a
> working npm, CMake falls through to downloading prebuilt UI assets from a
> Hugging Face bucket instead — a network-dependent fallback with its own
> failure modes — so installing Node as above is the reliable path. This is
> exactly what `docker/Dockerfile.worker`'s Stage 2 does (copies a Node 22
> binary in via multi-stage `COPY`) for the same reason.

- NVIDIA hosts can build with `-DGGML_CUDA=ON` instead for CUDA offload; the
  Vulkan build above already covers AMD, NVIDIA, and Intel.
- To relocate it to a single file on `PATH`, add `-DBUILD_SHARED_LIBS=OFF`
  (the Docker image does this) so it links statically, then copy
  `build/bin/llama-server` anywhere and drop the `LLAMASERVER_BINARY_PATH`
  export.

> **Minimum version — do not use a build older than mid-March 2026.** Builds
> from around then had a tool-calling bug that corrupted the `arguments` of
> streamed `tool_calls` (llama.cpp issue #20198, fixed in PR #20213). The pinned
> tag **b9941** (released 2026-07-09) is well past the fix. If agents see
> garbled tool-call arguments, your `llama-server` is too old — rebuild at the
> pinned tag or newer.

### Port reachability (coordinator → worker)

Each llamaserver model listens on its own coordinator-assigned `llamaserver_port`
(`config/models.toml`: **8081** Devstral, **8082** Qwen3-Coder). The worker's
`llamaserver_bind_host` defaults to **`127.0.0.1`**, secure by default —
`/slots` can leak another slot's prompt text, and this port has no auth at all.
The coordinator proxies to `http://<worker_host>:<port>`; those TCP ports must
therefore be **reachable from the coordinator host**:

- **Same-host Docker Compose (the default, no action needed):** the shipped
  `config/worker.toml` sets `llamaserver_bind_host = "0.0.0.0"` explicitly
  (documented there) because the worker and coordinator are separate
  containers on the private compose network — that is NOT the same as LAN
  exposure, since `docker-compose.yml` no longer publishes 8081/8082 to the
  host by default.
- **LAN cluster (coordinator on a DIFFERENT host):** publish the ports by
  layering `docker-compose.expose-ports.yml`
  (`docker compose -f docker-compose.yml -f docker-compose.expose-ports.yml
  --profile <name> up -d`) — see section 1 — and set `WORKER_GRPC_AUTH_TOKEN`.
- **Bare metal:** set `llamaserver_bind_host = "0.0.0.0"` in `worker.toml`
  and open the ports in the worker host's firewall to the coordinator only.
- **Trusted-LAN only:** that proxy hop is still plaintext and unauthenticated
  at the HTTP layer — never expose the llamaserver ports to an untrusted
  network. Also disable `/slots` unless you specifically need it
  (`llamaserver_enable_slots_endpoint = false` is the default).

### Windows

Windows `llama-server` provisioning is not yet supported.

## 5. Health checks

- Coordinator: `GET /health` → `{"status": "healthy", "workers": N}` (always 200; `starting` before ready).
- Worker (metrics port): `GET /health` → `OK`, `GET /live` → `ALIVE`.
- There are no `/health/live|ready|startup` coordinator routes.

## 6. Upgrades

```bash
git pull
docker compose build
docker compose up -d      # brief downtime; rolling/canary orchestration is not built in
```

## Planned (not shipped)

Kubernetes manifests/Helm, systemd unit files, backup/restore + canary/blue-green
scripts, Vault/Redis/MinIO/Elastic integrations, TLS/mTLS, Jaeger tracing.
