# API Reference

Base URL: `http://<coordinator-host>:8000`

Secure by default: the coordinator refuses to start on a non-loopback bind
with no `COORDINATOR_API_KEYS` set. When keys are configured, every route
except `/health` and `/metrics` requires `Authorization: Bearer <key>` or
`x-api-key: <key>`; with no keys set (loopback-only use), the API is open.
See [configuration.md](configuration.md) and [deployment.md](deployment.md).
Rate limiting is still planned, not implemented.
Interactive docs: `/docs` (Swagger UI), `/redoc`, schema at `/openapi.json`.

Errors use FastAPI's standard shape — `{"detail": "<message>"}` — with status
codes 200, 401 (missing/bad API key), 403 (admin-required route with a
non-admin key, or a model outside the key's scope — see
[configuration.md](configuration.md)), 404, 413 (request body over
`COORDINATOR_MAX_REQUEST_BODY_BYTES`, 25 MB default), 422 (validation), 500,
503 (no workers / not ready), 504 (timeout).

## Endpoints

### GET /health

```json
{"status": "healthy", "workers": 1}
```
Returns `{"status": "starting"}` before the coordinator finishes startup. Always 200.

### GET /v1/workers

Returns a bare JSON array:

```json
[
  {
    "id": "worker-0",
    "address": "worker:50051",
    "state": "healthy",
    "gpus": [{"id": 0, "name": "Radeon RX 7600 (Vulkan)", "memory_gb": 8.0, "available_gb": 6.2}],
    "loaded_models": ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
    "active_requests": 0
  }
]
```

### POST /v1/workers/manual (admin-only)

Body: a JSON array of `host:port` strings. Connects each and reports per-address status.
Requires an admin API key when `COORDINATOR_API_KEY_FILE` is configured (see
[configuration.md](configuration.md)), on top of its existing
feature-flag/credential gates.

```bash
curl -X POST http://localhost:8000/v1/workers/manual \
  -H "Content-Type: application/json" -d '["192.168.1.20:50051"]'
```

### GET /v1/models

OpenAI-style list with cluster extensions. When the calling key has a model
scope (`COORDINATOR_API_KEY_FILE`), the list is filtered to only the models
that key may address; an unrestricted key, or auth off entirely, sees every
registered model.

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3-8b",
      "object": "model",
      "created": 0,
      "owned_by": "custom",
      "family": "llama",
      "parameters": "8B",
      "min_memory_gb": 16.0,
      "loaded_on": [],
      "supports_quantization": ["none"]
    }
  ]
}
```

`supports_quantization` always reads `["none"]` today, including for
GGUF/llama.cpp-engine models — it describes the `quantization` request field
below (a Burn-engine load-time knob), not whether the model file itself is
quantized. A registry entry with `engine = "llamacpp"` runs a GGUF file
that's already quantized (e.g. Q4_K_M) regardless of this field; see
[configuration.md](configuration.md).

### POST /v1/models/load (admin-only)

Requires an admin API key when `COORDINATOR_API_KEY_FILE` is configured; a
`models`-scoped key is also refused (403) for a `model_name` outside its
scope. See [configuration.md](configuration.md).

Request: `{"model_name": "<registry key or HF repo id>", "worker_id": null, "quantization": "none", "instances": null}`
— the `quantization` field only accepts `"none"` today (other values are
422/rejected by workers). It applies to the Burn engine's FP32 loader only;
it is not how GGUF models get quantized — for those, quantization comes from
the GGUF file picked in the registry entry (`engine = "llamacpp"`), and this
field stays `"none"` regardless. `instances` (integer, >= 1) overrides the
registry's `[models.X.llamaserver] instances` value for this load only — it
only applies to `engine = "llamaserver"` models (422 otherwise) and sets the
number of concurrent conversation slots; see
[configuration.md](configuration.md).
Response: `{"status": "loaded"|"failed", "model_name": "...", "worker_id": "...", "memory_used_gb": null, "message": null}`

### DELETE /v1/models/{model_name} (admin-only)

Unloads the model and frees GPU memory on every worker holding it (or one
worker via `?worker_id=`). 404 when not loaded anywhere. Requires an admin
API key when `COORDINATOR_API_KEY_FILE` is configured; a `models`-scoped key
is also refused (403) for a model outside its scope.

```json
{"status": "unloaded", "model_name": "llama3-8b", "workers": ["worker-0"]}
```

### POST /v1/completions (buffered)

Request fields (everything else is rejected with 422):

| Field | Type | Default | Notes |
|---|---|---|---|
| model | string | required | registry key or HF repo id; `model@worker_id` targets a worker |
| prompt | string | required | |
| max_tokens | int | 512 | 1–32768 |
| temperature | float | 0.7 | 0–2; < 0.01 → greedy argmax |
| top_p | float | 0.95 | 0–1 |
| top_k | int | 40 | 0 disables |
| stream | bool | false | ignored — this endpoint always buffers |
| worker_id | string | null | force a specific worker |
| session_id | string | null | sticky-session key for affinity routing |

Response (NOT OpenAI-shaped — `/v1/chat/completions` is the OpenAI-compatible one):

```json
{
  "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "text": "…generated text…",
  "tokens_generated": 50,
  "processing_time_ms": 2140.5,
  "worker_id": "worker-0"
}
```

### POST /v1/chat/completions (OpenAI-compatible; used by Open-WebUI)

Request: `model`, `messages` (`[{"role": "system|user|assistant", "content": "…"}]`),
`max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `session_id`.

Non-streaming response:

```json
{
  "id": "…request id…",
  "object": "chat.completion",
  "created": 1719800000,
  "model": "llama3-8b",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "…"}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 42, "total_tokens": 42}
}
```

With `"stream": true`, Server-Sent Events are emitted live as the worker
generates (`Content-Type: text/event-stream`):

```
data: {"id": "…", "object": "chat.completion.chunk", "created": 1719800000, "model": "llama3-8b", "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": null}]}

data: {"id": "…", "object": "chat.completion.chunk", …, "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}]}

data: [DONE]
```

### GET /metrics

Prometheus exposition (coordinator). Key series: `coordinator_requests_total{model,status}`,
`coordinator_request_duration_seconds`, `coordinator_active_requests`,
`router_routed_requests_total{strategy,model}`, `router_queue_size{priority}`,
`router_circuit_breaker_open`. Workers expose their own `/metrics` on port 9091+
(`worker_requests_total`, `worker_request_duration_seconds`,
`worker_gpu_*`, `worker_model_memory_bytes`, `worker_errors_total`, …).

## curl quick reference

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/workers
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}'
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Hello", "max_tokens": 32}'
curl -X DELETE "http://localhost:8000/v1/models/TinyLlama%2FTinyLlama-1.1B-Chat-v1.0"
```

## Planned (not implemented)

Rate limiting, batch endpoint (`/v1/completions/batch`), per-worker/per-model
detail endpoints, WebSocket streaming, client SDKs (`pip install
ai-cluster-client` does not exist yet).
