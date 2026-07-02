# Configuration Guide

Three real configuration surfaces:

1. **Coordinator** — environment variables only (prefix `COORDINATOR_`, or `.env`).
2. **Worker** — CLI flags / env vars, with `config/worker.toml` (flat) as fallback.
3. **Model registry** — `config/models.toml`.

## Coordinator (COORDINATOR_* environment variables)

Source of truth: `coordinator/config.py` (pydantic-settings). Unknown `.env`
keys are ignored, so the shared `.env` can also carry worker variables.

| Variable | Default | Description |
|---|---|---|
| `COORDINATOR_HOST` | `0.0.0.0` | Bind address |
| `COORDINATOR_PORT` | `8000` | HTTP port |
| `COORDINATOR_DISCOVERY_METHOD` | `static` | Only `static` is implemented (`mdns`/`broadcast`/`consul` are planned and fail fast) |
| `COORDINATOR_STATIC_WORKERS` | `[]` | Comma-separated `host:port` list (or JSON array) |
| `COORDINATOR_DISCOVERY_INTERVAL` | `30` | Discovery loop seconds (min 5) |
| `COORDINATOR_HEALTH_CHECK_INTERVAL` | `30` | Health loop seconds (min 5) |
| `COORDINATOR_HEALTH_CHECK_TIMEOUT` | `5` | GetStatus RPC timeout seconds |
| `COORDINATOR_MAX_FAILURES` | `3` | Consecutive failures before UNHEALTHY |
| `COORDINATOR_REQUEST_TIMEOUT` | `300` | End-to-end inference timeout seconds |
| `COORDINATOR_MAX_QUEUE_SIZE` | `1000` | Request queue bound |
| `COORDINATOR_MODELS_CONFIG` | `config/models.toml` | Registry path |
| `COORDINATOR_CORS_ORIGINS` | `*` | `*`, comma-separated list, or JSON array |
| `COORDINATOR_MAX_CONCURRENT_REQUESTS_PER_WORKER` | `10` | Per-worker in-flight cap |
| `COORDINATOR_ROUTING_STRATEGY` | `least_load` | `least_load` / `round_robin` / `random` / `affinity` / `power_of_two` |
| `COORDINATOR_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Failures before a worker's circuit opens |
| `COORDINATOR_CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | `30.0` | Seconds before half-open |
| `COORDINATOR_AFFINITY_TTL_SECONDS` | `600.0` | Session stickiness TTL (affinity strategy) |

There is no `coordinator.yaml` — YAML config was removed in favor of env-only.

## Worker

Precedence for `worker_id` / `grpc_port` / `metrics_port` / `gpu_ids`:
**CLI flag or env var → `config/worker.toml` → built-in default.**

### CLI flags / env vars (clap)

| Flag | Env | Default | Description |
|---|---|---|---|
| `--worker-id` | `WORKER_ID` | auto (`worker-<gpu>`) | Unique id reported to the coordinator |
| `--port` | `GRPC_PORT` | 50051 | gRPC port |
| `--metrics-port` | `METRICS_PORT` | 9091 | Prometheus port |
| `--gpu-ids` | `GPU_IDS` | `[0]` | Comma-separated device indices |
| `--config` | `CONFIG_FILE` | `config/worker.toml` | TOML path |
| `--log-level` | `LOG_LEVEL` | `info` | debug/info/warn/error (RUST_LOG wins when set) |
| `--log-json` | `LOG_JSON` | off | JSON log output |

Additional env: `RUST_LOG`, `RUST_BACKTRACE`, `HF_TOKEN` (gated model downloads),
`GPU_VRAM_GB` (VRAM hint when the vendor tool can't report it — code default 8).
Docker replicas also use `GPU_INDEX`, `GRPC_BASE_PORT`, `METRICS_BASE_PORT`
(entrypoint computes port = base + index).

### config/worker.toml (flat — unknown keys are a hard error)

```toml
# worker_id = "my-worker"
grpc_port = 50051
metrics_port = 9091
gpu_ids = [0]
model_cache_dir = "./models"
download_dir = "./data/downloads"
max_concurrent_loads = 2
max_concurrent_requests = 32
request_timeout_secs = 120
# hf_token = "hf_..."          # HF_TOKEN env var wins
# hf_cache_dir = "/models/hf"
```

## Model registry: config/models.toml

Real keys per model (everything else is ignored):

```toml
[models.my-model]                # registry key used in API calls
family = "llama"                 # llama | qwen | deepseek | mistral | phi | gemma
description = "…"
parameters = "8B"
min_memory_gb = 16
recommended_gpus = 1
max_gpus = 2

[models.my-model.architecture]
num_layers = 32
hidden_size = 4096
num_attention_heads = 32
num_kv_heads = 8
vocab_size = 128256
max_seq_len = 8192
intermediate_size = 14336
rms_norm_eps = 1e-5
rope_theta = 500000.0
is_moe = false

[models.my-model.quantization]
supported = ["none"]             # quantized inference is planned
default = "none"

[models.my-model.parallelism]
supported = ["single"]
default = "auto"

[models.my-model.hf]
repo_id = "org/real-hf-repo"     # what the worker downloads (model_path)
```

Only `llama`, `qwen`, and `deepseek` architectures are loadable by the worker
today; `mistral`/`phi`/`gemma`/`mixtral` entries are planned placeholders.

## Monitoring configs

- `config/prometheus.yml` — scrape targets `coordinator:8000` and `worker:9091`.
- `config/alerts.yml` — alert rules against the real metric names.
- `monitoring/grafana-*` — Grafana provisioning (datasource + overview dashboard).

## Validation

- Coordinator: settings validate at startup (pydantic) — bad values fail fast with a clear error.
- Worker: `ai-worker --config <path>` fails fast on unknown/invalid TOML keys.
- Alerts: `promtool check rules config/alerts.yml`.
