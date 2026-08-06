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
| `COORDINATOR_HOST` | `0.0.0.0` | Bind address. Refuses to start when this is not loopback (`127.0.0.1`/`::1`/`localhost`) AND `COORDINATOR_API_KEYS` is unset — secure by default. |
| `COORDINATOR_PORT` | `8000` | HTTP port |
| `COORDINATOR_API_KEYS` | *(unset)* | Comma-separated API keys gating the whole HTTP surface (except `/health`/`/metrics`); `Authorization: Bearer <key>` or `x-api-key: <key>`. Empty = open (only safe with a loopback `COORDINATOR_HOST`). |
| `COORDINATOR_API_KEY_FILE` | *(unset)* | Path to a TOML file assigning a `role`/`models` identity to individual keys, layered on top of `COORDINATOR_API_KEYS`. See "Per-key identity" below. |
| `COORDINATOR_DISCOVERY_METHOD` | `static` | Only `static` is implemented (`mdns`/`broadcast`/`consul` are planned and fail fast) |
| `COORDINATOR_STATIC_WORKERS` | `[]` | Comma-separated `host:port` list (or JSON array) |
| `COORDINATOR_DISCOVERY_INTERVAL` | `30` | Discovery loop seconds (min 5) |
| `COORDINATOR_HEALTH_CHECK_INTERVAL` | `30` | Health loop seconds (min 5) |
| `COORDINATOR_HEALTH_CHECK_TIMEOUT` | `5` | GetStatus RPC timeout seconds |
| `COORDINATOR_MAX_FAILURES` | `3` | Consecutive failures before UNHEALTHY |
| `COORDINATOR_REQUEST_TIMEOUT` | `300` | End-to-end inference timeout seconds |
| `COORDINATOR_MAX_QUEUE_SIZE` | `1000` | Request queue bound |
| `COORDINATOR_MAX_REQUEST_BODY_BYTES` | `25000000` | Max HTTP request body size in bytes |
| `COORDINATOR_MODELS_CONFIG` | `config/models.toml` | Registry path |
| `COORDINATOR_ALLOW_UNREGISTERED_MODEL_PULL` | `false` | Allow `POST /v1/models/load` for a `model_name` absent from the registry (otherwise sent to the worker as an arbitrary HF repo id) |
| `COORDINATOR_ALLOW_MANUAL_WORKER_REGISTRATION` | `false` | Enable `POST /v1/workers/manual` — also always requires a valid `COORDINATOR_API_KEYS` credential regardless of global auth state |
| `COORDINATOR_MANUAL_WORKER_ALLOWED_HOSTS` | `[]` | Optional host/CIDR allowlist for `POST /v1/workers/manual`; empty accepts any well-formed address once the feature above is enabled |
| `COORDINATOR_CORS_ORIGINS` | `[]` | `*`, comma-separated list, or JSON array — beyond the always-on loopback allowance |
| `COORDINATOR_MAX_CONCURRENT_REQUESTS_PER_WORKER` | `10` | Per-worker in-flight cap |
| `COORDINATOR_ROUTING_STRATEGY` | `least_load` | `least_load` / `round_robin` / `random` / `affinity` / `power_of_two` |
| `COORDINATOR_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Failures before a worker's circuit opens |
| `COORDINATOR_CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | `30.0` | Seconds before half-open |
| `COORDINATOR_AFFINITY_TTL_SECONDS` | `600.0` | Session stickiness TTL (affinity strategy) |
| `COORDINATOR_LLAMASERVER_SLOT_AFFINITY` | `true` | Pin a resolved caller to a consistent llama-server conversation slot (`id_slot`) on `engine="llamaserver"` models with 2+ slots. Only takes effect once `COORDINATOR_API_KEY_FILE` is set. See "llama-server knobs" below. |

There is no `coordinator.yaml` — YAML config was removed in favor of env-only.

### Security notes

- **API keys**: env-only (`COORDINATOR_API_KEYS` and/or
  `COORDINATOR_API_KEY_FILE`), checked via `Authorization: Bearer <key>` or
  `x-api-key: <key>`. The valid-key set is the union of both sources,
  re-read on every request (no restart needed after an edit). `/health` and
  `/metrics` are always exempt. Comparison is constant-time over UTF-8 bytes
  (not `hmac.compare_digest` directly, which requires ASCII-only `str`
  arguments and would 500 on a non-ASCII candidate key instead of 401ing). A
  CORS preflight (`OPTIONS`) always bypasses the key check — browsers don't
  attach credentials to preflights, and gating it would make browsers
  misreport a 401 as a CORS failure.

- **Middleware order**: `add_middleware` prepends, so registration order is
  reversed at request time. Auth is registered before CORS so a preflight
  reaches CORS first; the request-body-size cap is registered last (so it
  runs outermost) to reject oversized bodies before auth/CORS spend cycles
  on them.
- **Request body size cap** (`COORDINATOR_MAX_REQUEST_BODY_BYTES`): enforced
  two ways — a `Content-Length` pre-check, and a streaming byte counter for
  chunked or lying-header bodies. Applies to the `engine="llamaserver"` raw
  proxy path too. Default 25 MB comfortably covers large prompts/context
  without allowing unbounded memory use per request.
- **`POST /v1/workers/manual`** is off by default because it lets a caller
  point the coordinator's gRPC client at an arbitrary host:port — enabling
  it always requires a valid `COORDINATOR_API_KEYS` credential regardless of
  the global auth setting, to avoid becoming an SSRF/traffic-hijack vector.
  A registered worker self-reports `loaded_models`, so this endpoint can
  make other model names appear served.
- **`COORDINATOR_ALLOW_UNREGISTERED_MODEL_PULL`** is off by default because,
  when on, an unrecognized `model_name` is forwarded to the worker as a raw
  HuggingFace repo id.
- **Context-compression budget**: sized as `n_ctx - max_tokens - 512`, where
  512 is headroom for chat-template/formatting overhead.

#### Per-key identity (`COORDINATOR_API_KEY_FILE`)

`COORDINATOR_API_KEYS` alone still behaves exactly as before: every key it
lists resolves to an admin, unrestricted caller — this is the compatibility
guarantee for a deployment that sets nothing new. Pointing
`COORDINATOR_API_KEY_FILE` at a TOML file layers a role and an optional
model scope onto individual keys instead:

```toml
[keys.ci-runner]
key = "3f9a..."
role = "user"                   # "admin" or "user"; default "user"
models = ["qwen2.5-0.5b-gguf"]  # omitted or [] = unrestricted

[keys.ops]
key = "b711..."
role = "admin"
```

The table label (`ci-runner`, `ops`) is the caller id recorded in the audit
log. Once the file is configured: a key defined in it uses that entry; a key
present only in `COORDINATOR_API_KEYS` is demoted to `role = "user"` with an
unrestricted model list; this holds even when the file declares zero keys.
The file is validated strictly and fails closed — bad TOML, a missing or
empty `key`, an invalid `role`, a non-string/empty `models` entry, a
duplicate key across labels, or an unknown top-level table all raise, and
the coordinator responds 503 to every request rather than falling back to
open or partially-resolved access. The parsed file is cached on
`(path, mtime, size)`, so an edit is picked up without a restart but a
request doesn't re-parse the TOML every time. Key comparison is the same
constant-time UTF-8-byte comparison used for `COORDINATOR_API_KEYS`.

#### Model scoping

A key's `models` list (from `COORDINATOR_API_KEY_FILE`) restricts which
model names it may address on `/v1/completions`, `/v1/chat/completions`,
`/v1/messages`, `/v1/messages/count_tokens`, `/v1/embeddings`, `/infill`,
`POST /v1/models/load`, and `DELETE /v1/models/{model_name}`. A request
naming any other model gets **403, never 404** — and the message is
identical whether or not that model actually exists in the registry, so the
response never reveals which models are registered to a key that's probing
for them. `GET /v1/models` returns only the models a key may address (an
unrestricted key, or auth off entirely, still sees everything). An empty or
absent `models` list means unrestricted, same as a plain
`COORDINATOR_API_KEYS` entry.

#### Admin-only routes

`POST /v1/models/load`, `DELETE /v1/models/{model_name}`, and
`POST /v1/workers/manual` additionally require `role = "admin"`; a
non-admin key gets 403. `/v1/workers/manual` keeps its existing gates too
(feature flag off by default, plus its own independent credential check) —
the admin check runs after those, so its existing 401/403 responses for a
disabled or unauthenticated request are unchanged.

#### Audit log

Management actions (`model.load`, `model.unload`, `model.autoload`,
`model.evicted`, `worker.register`) each emit one single-line JSON record
at INFO on the `coordinator.audit` logger: `ts` (UTC ISO-8601), `action`,
`caller`, `outcome` (`success`, `failure`, or `denied`), and optionally
`model`, `worker`, `detail`. Prompts, request/response bodies, and key
material are never logged; `caller`/`model`/`worker` are capped at 128
characters and `detail` at 200, each truncated with a trailing `...` rather
than rejected. `model.evicted` is attributed to the caller whose load
triggered the worker-side eviction — that's the only point at which an
eviction is observable to the coordinator at all.

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
`GPU_VRAM_GB` (explicit VRAM override, in GB — takes precedence over any
detected value; unset by default, including in `docker-compose.yml`),
`GPU_MEMORY_HEADROOM_PERCENT` (default `15`; on unified-memory hardware
where the vendor "free" figure is unreliable, the available-memory estimate
is this percent less than total system RAM, held back as headroom for the
OS/other processes), `WORKER_GRPC_AUTH_TOKEN` (shared-secret gRPC auth; wins
over `worker.toml`'s `grpc_auth_token`), `WORKER_GRPC_BIND_HOST` /
`LLAMASERVER_BIND_HOST` (win over `worker.toml`'s `grpc_bind_host` /
`llamaserver_bind_host`; `docker-compose.yml` sets both to `0.0.0.0` per
worker service — see [deployment.md](deployment.md)), `LLAMASERVER_BINARY_PATH`,
`RPC_SERVER_BIND_HOST` / `RPC_SERVER_BINARY_PATH` (win over `worker.toml`'s
`rpc_server_bind_host` / `rpc_server_binary_path`; see "Distributed (Level 2,
ggml-RPC)" below — unlike the other binds, never set this to `0.0.0.0`).
Docker replicas also use `GPU_INDEX`, `GRPC_BASE_PORT`, `METRICS_BASE_PORT`
(entrypoint computes port = base + index).

### Memory detection

Total VRAM comes from the CUDA driver (`cuMemGetInfo`, via `dlopen`) or a
vendor tool (`nvidia-smi`) where available. On unified-memory hardware (e.g.
DGX Spark) the vendor "free" figure counts reclaimable page cache as used
and can understate available memory by tens of GB, so the *available*
figure falls back to `GPU_MEMORY_HEADROOM_PERCENT` less than total system
RAM instead of trusting that number. `GPU_VRAM_GB` overrides total memory
outright when set, ahead of any detection. See "Hardening notes" below for
the DGX Spark `nvidia-smi`/NVML failure mode this works around.

### config/worker.toml (flat — unknown keys are a hard error)

```toml
# worker_id = "my-worker"
grpc_bind_host = "127.0.0.1"  # secure by default — 0.0.0.0 is an explicit opt-in
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
# grpc_auth_token = "..."      # WORKER_GRPC_AUTH_TOKEN env var wins
# llamaserver_bind_host = "127.0.0.1"          # secure by default
# llamaserver_enable_slots_endpoint = false    # /slots can leak another slot's prompt text
# llamaserver_port_min = 1024                  # allowed llamaserver.port range
# llamaserver_port_max = 65535
# max_n_ctx = 262144                           # ceiling on caller-supplied n_ctx
# rpc_server_enabled = false                   # opt this node into lending GPU(s) (feature llamacpp-rpc)
# rpc_server_bind_host = "127.0.0.1"           # secure by default — set to the interconnect address, never 0.0.0.0
# rpc_server_binary_path = "ggml-rpc-server"   # RPC_SERVER_BINARY_PATH env var wins
# rpc_server_port = 50151                      # base port; a peer lending N GPUs uses port..port+N-1
```

### llama-server (`engine = "llamaserver"`) knobs

- **`instances`** (`[models.X.llamaserver] instances = N`, default `1`;
  `parallel` is still accepted as an alias, `instances` wins if both are
  set) — the number of concurrent conversation slots, mapped to
  `llama-server`'s `--parallel`. Also settable per request:
  `POST /v1/models/load {"model_name": "...", "instances": 3}`. Measured on
  hardware, `llama-server` does **not** divide `-c` across slots — every
  slot gets the *full* `-c` value, which is exactly the registry's `n_ctx`
  (see the `n_ctx` note below). Raising `instances` grows how many
  concurrent conversations can use that context, not any single slot's
  ceiling. The worker reads the GGUF header and reserves memory for
  `n_ctx * instances` up front, refusing the load with a
  resource-exhausted error (rolling back any prior reservation) if it will
  not fit, rather than spawning `llama-server` and finding out later.
- **Slot affinity** (`COORDINATOR_LLAMASERVER_SLOT_AFFINITY`, default on):
  when a model has 2+ slots, the coordinator derives a stable slot index
  from the caller id (`sha256`, so it survives a restart) and injects
  `id_slot` into the forwarded body, giving a resolved caller a consistent
  slot across requests. It's a routing preference, not a reservation:
  `llama-server` queues a request naming a busy slot until that slot frees
  rather than rejecting it, and the coordinator never overrides an
  `id_slot` the client already sent. Injection only happens when all of the
  following hold: the setting is on, `COORDINATOR_API_KEY_FILE` is
  configured, the caller resolved to a real identity (not the unrestricted
  fallback), the request path is `/v1/chat/completions` or
  `/v1/completions` (the two routes that funnel into llama-server's shared
  completions handler), the model has 2 or more slots, and the client
  didn't already send `id_slot`. Every other request is still forwarded
  byte-for-byte; this is the one case where that raw-bytes passthrough is
  relaxed (the body is re-serialized to add the field). The coordinator
  never consults `GET /slots` for this — slot count comes from the
  registry's `instances`/`parallel` value, since `/slots` is disabled by
  default on the worker (`llamaserver_enable_slots_endpoint = false`) as it
  can leak another slot's cached prompt text. Two practical consequences: a
  caller pinned to one slot has its own concurrent requests served serially
  by that slot; and because the index is a hash modulo the slot count, two
  callers can collide onto the same slot (there are more possible callers
  than slots, and nothing reserves one). Treat it as a consistency hint, not
  a guarantee — and note the mapping shifts if `instances` changes.
- **`n_ctx`, `-c`, and `--parallel` — measured, not assumed.** An earlier
  version of this doc claimed `llama-server` divides its own `-c` value
  evenly across `--parallel` slots. That is **false** for the current
  build. Measured directly:
  ```
  -c 8192   (default --parallel): n_slots=4, n_ctx_slot=8192,   kv_unified=true
  -c 196608 --parallel 1:         n_slots=1, n_ctx_slot=196608, kv_unified=false
  ```
  Every slot got the *full* `-c` value, not `-c / parallel`. With more than
  one slot, `kv_unified=true` — slots share one KV pool instead of each
  owning a private one. Since that measurement, the worker passes the
  registry's `n_ctx` straight through as `-c`, unmultiplied — every slot
  gets exactly `n_ctx` tokens regardless of `instances`, so `-c` never
  exceeds the model's own `n_ctx_train` on account of `instances` alone.
  `verify_props` cross-checks the spawned server's reported context
  against this same expectation as a diagnostic (warns, never fails the
  load) — it flags a mismatch with what the worker *asked for*, not an
  independent correctness check of that request.
- **`n_gpu_layers`** (`-ngl`): partial-offload knob for consumer GPUs that
  can't fit every layer in VRAM; negative = offload all layers.
- **`n_cpu_moe`** (`--n-cpu-moe <N>`): keeps the first `N` layers' MoE
  expert weights on CPU while `-ngl` still places everything else on GPU —
  the way to fit a large MoE model into less VRAM than `n_gpu_layers` alone
  achieves. Tune by watching VRAM at load time and lowering `N` (more
  experts on GPU, faster) until it fits; see
  https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide.
- **`extra_args`** is whitespace-split and appended verbatim to
  `llama-server`'s argv (flag injection is possible, not shell injection).
  Every flag with filesystem/network/credential semantics is excluded from
  the allowlist (e.g. `--path`, `--log-file`, `--host`, `--port`,
  `--api-key-file`, `--lora`, `--slot-save-path`, `--models-dir`,
  `--hf-token`, `--chat-template-file`, `--ssl-key-file`), as are flags that
  duplicate a typed field above (`-np`/`--parallel`, `-c`/`--ctx-size`,
  `-m`/`--model`, `-ngl`, `-ncmoe`, `-ctk`/`-ctv`) so a conflicting flag
  can't silently override a value already used to size the KV-cache
  reservation. Unknown flags are rejected outright. Reasoning/thinking
  controls — `-rea`/`--reasoning on|off|auto`, `--reasoning-budget N`
  (token budget for thinking; `-1` unrestricted, `0` ends thinking
  immediately), `--reasoning-budget-message`, `--reasoning-preserve`, and
  `--reasoning-format` are on the allowlist; a reasoning model otherwise
  returns an empty `message.content` until it finishes thinking.
  **Prefer `--reasoning-budget N` over `--reasoning off`** — it keeps
  reasoning while bounding it, instead of disabling it outright. Watch for
  the failure mode either way: reasoning on with a low `max_tokens` can
  consume the whole budget on thinking and return empty `content` with
  `finish_reason: "length"` before any answer is produced. See
  [troubleshooting.md](troubleshooting.md).

## Model registry: config/models.toml

### GGUF / llama.cpp models (recommended)

This is the primary, recommended way to configure a model for the project's
actual goal — running inference on consumer GPUs (~8–16 GB VRAM, NVIDIA or
AMD), including idle cards, using a quantized checkpoint. Set
`engine = "llamacpp"` and point `[models.<key>.gguf]` at a pre-quantized GGUF
file on Hugging Face; no `[.architecture]` block is needed since llama.cpp
reads architecture from the GGUF's own metadata. Requires the worker binary
to be built with `--features llamacpp` (see [deployment.md](deployment.md)).

```toml
[models."qwen2.5-7b-instruct-gguf"]      # registry key used in API calls
family = "qwen"
description = "Qwen2.5 7B Instruct — Q4_K_M GGUF, fits an 8 GB consumer GPU"
parameters = "7B"
min_memory_gb = 6
recommended_gpus = 1
max_gpus = 1
engine = "llamacpp"

[models."qwen2.5-7b-instruct-gguf".gguf]
repo_id = "Qwen/Qwen2.5-7B-Instruct-GGUF"    # HF repo containing the GGUF file
file = "qwen2.5-7b-instruct-q4_k_m.gguf"     # exact filename; already quantized
n_gpu_layers = -1                            # -1 = offload all layers to the GPU
n_ctx = 8192
```

Quantization here is a property of the GGUF file itself (Q4_K_M/Q5_K_M/Q8_0/…
— whatever the file was exported as), not a separate config knob. Splitting
one model across multiple consumer GPUs is upstream llama.cpp functionality
that isn't yet exposed through the worker wrapper (today it loads one GGUF
model per worker process); `recommended_gpus`/`max_gpus` on larger entries
describe the target shape once that lands. See real examples (including a
multi-GPU-target one) in `config/models.toml`.

### Burn / safetensors models (FP32 reference engine)

`engine` defaults to `"burn"` when omitted, and only applies to the worker's
default cargo build (no `--features llamacpp` needed). The Burn engine loads
full-precision safetensors and needs an `[.architecture]` block; it does not
quantize (`quantization != "none"` is rejected) and runs on a single GPU per
worker. Only `llama`, `qwen`, and `deepseek` architectures are loadable by
the worker today; `mistral`/`phi`/`gemma`/`mixtral` entries are planned
placeholders.

```toml
[models.my-model]
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
supported = ["none"]             # Burn engine is FP32-only; not applicable to GGUF models above
default = "none"

[models.my-model.parallelism]
supported = ["single"]
default = "auto"

[models.my-model.hf]
repo_id = "org/real-hf-repo"     # what the worker downloads (model_path)
```

### Sizing examples (measured on real hardware)

- **16 GB AMD fleet, Devstral 24B**: Q4_K_M weights are ~14.3 GB, leaving
  only ~1.0–1.7 GB for KV cache + compute. `cache_type_k`/`cache_type_v =
  "q8_0"` halves the fp16-default KV footprint, which is what makes 16384
  tokens of context fit at all; `q4_0` fits more but visibly hurts quality
  on an already VRAM-starved model. To raise `n_ctx` further without a
  bigger card, partially offload layers to system RAM via `n_gpu_layers`,
  or move to a 24 GB+ card.
- **16 GB AMD fleet, Qwen3-Coder-30B-A3B**: 18.6 GB of Q4_K_M weights alone
  don't fit a 16 GB card even before KV cache, so MoE experts must be
  partially CPU-offloaded via `n_cpu_moe`. `n_ctx` passes straight through
  as `-c` (see the `n_ctx`/`instances` notes above), so raising `instances`
  doesn't change `-c` at all — it only reserves `n_ctx` worth of KV memory
  per added slot, and each slot shares one `kv_unified` pool sized to `-c`
  rather than owning a private copy of it.
- **DGX Spark tier (GB10, 121.6 GiB unified memory), Qwen3.6-35B-A3B**:
  hybrid-attention — only ~1 layer in 4 holds a real KV cache, the rest use
  a fixed ~63 MiB recurrent buffer that doesn't grow with context. Measured
  KV: 1360 MiB at 131072 total context, 2720 MiB at 262144 (linear). At
  `n_ctx=262144, instances=4` the worker asks `llama-server` for
  `-c = 262144` — exactly this model's native training context, not a
  multiple of it (the worker no longer multiplies `-c` by `instances`).
  KV ≈10.6 GiB by the same linear rate (2720 MiB per slot × 4 slots) +
  ~22.4 GB Q4_K_XL weights ≈ ~33 GB against 121 GiB unified memory (~88 GiB
  headroom) — memory isn't the ceiling here. Qwen3.6 is a reasoning model:
  with thinking on, `message.content` stays empty
  until the chain-of-thought (in `reasoning_content`) finishes — a short
  prompt can burn the whole `max_tokens` budget on thinking and return
  empty content with `finish_reason="length"`. Prefer `--reasoning-budget
  N` at the process level (keeps reasoning, bounds it) over `--reasoning
  off` (drops it outright); per-request, `{"chat_template_kwargs":
  {"enable_thinking": false}}` disables it for that call.

### Distributed (Level 2, ggml-RPC)

`[models.<key>.distributed]` (see `qwen2.5-coder-32b-gguf` in
`config/models.toml`) splits one GGUF model's pipeline layers across a
"lead" node (owns the real llama.cpp context) and one or more "rpc_server"
peer nodes that only lend local GPU memory over the network:

```toml
[models."qwen2.5-coder-32b-gguf".distributed]
enabled = true
lead = "amd-node-1"                    # worker_id, must be a registered worker
peers = ["amd-node-2", "amd-node-3"]   # worker_ids, loaded before the lead
split = "auto"                         # "auto" or an explicit weight list, e.g. [0.5, 0.3, 0.2]
rpc_port = 50151                       # base ggml-RPC port each peer binds from

[models."qwen2.5-coder-32b-gguf".distributed.gpu_ids]
amd-node-1 = [0]     # GPU ids that node contributes; omit a node to use every GPU it reports
amd-node-2 = [0]
amd-node-3 = [0]
```

`distributed = true` requires `engine = "llamacpp"` or `engine =
"llamaserver"` (any other engine is rejected at config-load time). The DGX
Spark tier's MiniMax-M2.7 entry uses `engine = "llamaserver"`, since it
needs the worker-supervised `llama-server` process for real tool calling.

Build the worker with the opt-in feature: `cargo build --release --features
wgpu,llamacpp-rpc`.

**Coordinator orchestration** (`ClusterCoordinator._load_distributed_model`
in `coordinator/coordinator.py`): resolves `lead`/`peers` to registered,
`HEALTHY` workers (a missing or unhealthy worker_id fails the whole load
with no RPC sent at all); loads every peer FIRST, then the lead; computes
the tensor split from `distributed_split` when given, else derives it from
each contributed GPU's reported `total_memory` (largest-remainder/Hamilton
apportionment, so weights always sum to exactly 1.0); unloads in the
reverse order (lead first, then peers) since the lead depends on every peer
while serving. One failed peer or a failed lead aborts the whole load and
unloads whatever peers already succeeded — upstream llama.cpp has no
partial-load recovery, so this never leaves a half-loaded model around.
Inference for a distributed model is auto-pinned to its lead (the peers
hold no servable model).

**Worker roles**, set via the `distributed_role` metadata key:

- **`rpc_server` (peer)** — lends this node's GPU(s), never loads a
  servable model itself. The worker spawns one `ggml-rpc-server` child per
  GPU id requested (ports `rpc_bind_port`, `rpc_bind_port + 1`, …), health
  via a raw TCP connect (ggml-RPC has no HTTP surface), and kills it on
  unload. Requires `rpc_server_enabled = true` in that node's
  `worker.toml` — an explicit second opt-in beyond what the coordinator
  requests — and a non-`0.0.0.0` `rpc_server_bind_host`.
- **`lead`** — owns the real `llama-server` process and dials out to the
  peers. `rpc_peers` and `tensor_split` are threaded into
  `llama-server --rpc`/`--tensor-split`; both come from `distributed_role =
  "lead"` metadata, never from `llamaserver.extra_args` — a network target
  is a typed field here, not a free-form flag.

**gRPC metadata contract** (`ModelConfig.grpc_metadata_rpc_server`/
`grpc_metadata_lead` in `coordinator/models.py`, carried in the existing
`ModelConfig.metadata` map — no proto change):

| Key | Sent to | Meaning |
|---|---|---|
| `distributed_role` | peer, lead | `"rpc_server"` or `"lead"` |
| `rpc_bind_port` | peer | base port for that peer's own `ggml-rpc-server` process(es); a node lending *k* GPUs binds `rpc_bind_port .. rpc_bind_port+k-1` |
| `rpc_reserve_bytes` | peer | this node's share of the weights to reserve, split evenly across `gpu_ids`; omit to conservatively reserve each GPU's full available memory |
| `rpc_devices` | peer | comma-separated `-d` value per GPU, e.g. `"CUDA0,CUDA1"` — required whenever more than one `gpu_id` is requested, since there is no portable way to guess a backend's device-naming scheme |
| `rpc_peers` | lead | comma-separated `"host:port"` list, one entry per peer GPU, GPU-index-aligned with `tensor_split`'s peer portion |
| `tensor_split` | lead | comma-separated weights. **Peers come first, the lead's own GPUs last** |

**Leave `tensor_split` unset unless you have a reason.** llama.cpp then
places layers by live available memory, which measured better than any
hand-tuned value on identical nodes and adapts to mixed hardware. Note the
ordering above is easy to invert — with `--rpc`, llama.cpp enumerates RPC
peers *first* and the local GPU *last*, so `--tensor-split 0.55,0.45` sends
`0.55` to the peer and `0.45` to the lead, not the other way around.
Getting it backwards can fail the load outright with
`ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer`. Measured,
at full context on two identical 121 GiB nodes:

| `--tensor-split` (peer,lead) | Lead free | Peer free |
|---|---|---|
| `0.40,0.60` | failed to allocate | — |
| `0.45,0.55` | 6 GiB | 19 GiB |
| `0.55,0.45` | 27 GiB | 0 GiB |
| omitted (auto) | **17 GiB** | **10 GiB** |

Roughly 2 GiB of headroom shifts per `0.01` of split. Omitting
`--tensor-split` measured best on both balance and speed here, and it's the
only option of the four that adapts automatically to non-identical
hardware — hence the recommendation above.

Measured on two DGX Sparks over a 200 Gb/s ConnectX-7 link: splitting a
35B-A3B GGUF cost -3.0% generation and gained +44% prefill versus one node.
A 175 GiB model that fits no single node served at 18.19 tok/s with working
tool calls. Both sides are unit-tested, but the coordinator-driven path has
not yet been exercised on real hardware — those figures come from invoking
llama.cpp directly. See `pending-work/18-dual-spark-when-cable-arrives.md`
(internal, gitignored) for the full write-up and the llama.cpp RPC defects
to plan around (partial-graph failure kills the whole server, no
`--split-mode row` support, `--split-mode tensor` crashes MoE models).

ggml-RPC has no auth or encryption — bind a peer's port to the trusted
interconnect interface only, never a public one.

## Monitoring configs

- `config/prometheus.yml` — scrape targets `coordinator:8000` and `worker:9091`.
- `config/alerts.yml` — alert rules against the real metric names.
- `monitoring/grafana-*` — Grafana provisioning (datasource + overview dashboard).

## Validation

- Coordinator: settings validate at startup (pydantic) — bad values fail fast with a clear error.
- Worker: `ai-worker --config <path>` fails fast on unknown/invalid TOML keys.
- Alerts: `promtool check rules config/alerts.yml`.

## Hardening notes

- **`resolve_head_dim` division guard**: a checkpoint's `config.json` can
  set `num_attention_heads = 0`; computing `hidden_size / num_attention_heads`
  eagerly would panic (div-by-zero) before the model-load error path could
  roll back its GPU memory reservation. The division only runs once the
  divisor is known non-zero, and an RAII guard releases the reservation on
  any unwind, not just the `Err` arm.
- **Eviction race**: two concurrent loads of different models, each
  evicting under only a per-model lock, could pick the same victim. A
  single loader-wide eviction lock serializes the "choose victim(s), evict,
  insert" sequence and re-checks capacity at insert time; the (slow)
  download itself stays outside the lock.
- **GGUF shard path safety**: `hf-hub`'s cache path join does not sanitize
  `..`/absolute paths in a filename (checked against hf-hub 0.4.3), so a
  shard name like `../../../etc/passwd` could escape the cache directory.
  Every shard filename — including ones the loader derives at runtime, not
  just the configured `file` — is validated as a safe relative path before
  use.
- **Multi-GPU adapter de-duplication**: `wgpu`'s `AdapterInfo` has no PCI
  bus/slot address, so name/vendor/device-id dedup can't always tell two
  identical physical cards apart. Enumerating from a single backend at a
  time avoids merging genuinely distinct hardware (observed bug: an NVIDIA
  GB10 double-enumerated as `IntegratedGpu`/Vulkan and `Other`/GL under the
  same PCI ids) — the tradeoff is that a card double-enumerated *within*
  one backend by competing ICDs isn't caught.
- **NVIDIA memory detection on DGX Spark**: `nvidia-smi
  --query-gpu=memory.total` and `nvmlDeviceGetMemoryInfo` both fail
  (`[N/A]` / `NVML_ERROR_NOT_SUPPORTED`) on a GB10 host even though the GPU
  works. `cuMemGetInfo`, loaded via `dlopen` on the CUDA driver API (no
  build-time link dependency, so non-NVIDIA builds are unaffected),
  correctly reports the real total/free memory — matching `llama-server
  --list-devices`'s own independent query. It uses the CUDA *primary*
  context (refcounted per process/device) so it can't interfere with the
  in-process Burn or llama.cpp engines' own CUDA usage.
