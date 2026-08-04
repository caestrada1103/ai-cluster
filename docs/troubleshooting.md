# Troubleshooting

## Coordinator won't start

- **`ModuleNotFoundError` / import errors** — start from the REPO ROOT:
  `uvicorn coordinator.main:app --host 127.0.0.1 --port 8000` (running `uvicorn main:app`
  from inside `coordinator/` breaks package imports).
- **`RuntimeError: Refusing to start: COORDINATOR_HOST='0.0.0.0' is not
  loopback-only and COORDINATOR_API_KEYS is unset`** — this is intentional
  (C4, secure by default): a non-loopback bind with no API keys means every
  route is reachable with zero credentials. Either use `--host 127.0.0.1`
  for local-only, or set `COORDINATOR_API_KEYS` (comma-separated secrets)
  before binding to `0.0.0.0`/a LAN address. See `.env.example`.
- **Settings validation error at startup** — the message names the bad
  `COORDINATOR_*` variable; see docs/configuration.md for the full table.
  `COORDINATOR_DISCOVERY_METHOD` other than `static` fails fast (mdns/broadcast/consul are planned).

## No workers show up

```bash
curl http://localhost:8000/v1/workers          # what the coordinator sees
docker compose logs worker | tail -50          # worker-side errors
# Manual registration to test connectivity:
curl -X POST http://localhost:8000/v1/workers/manual \
  -H "Content-Type: application/json" -d '["worker:50051"]'
```
- Check `COORDINATOR_STATIC_WORKERS` matches the worker's address:port.
- Multi-GPU replicas listen on `GRPC_BASE_PORT + GPU_INDEX` (50052 for index 1, …).

## GPU not detected in the worker

```bash
# In the container/host:
vulkaninfo --summary          # universal image — a real GPU must be listed
nvidia-smi                    # NVIDIA
rocm-smi                      # AMD
```
- AMD: compose must mount `/dev/kfd` + `/dev/dri` and add `group_add: [video, render]`.
- NVIDIA: install the NVIDIA Container Toolkit; the default image needs
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` (set in the Dockerfile).
- Docker Desktop on Windows/WSL2 + NVIDIA: no Vulkan ICD is injected — the
  universal image falls back to CPU. Build the CUDA variant
  (`--build-arg BACKEND=cuda`, see docker/Dockerfile.worker header).
- Wrong reported VRAM? Set `GPU_VRAM_GB` (used when vendor tools can't report).

## Model load fails

- `Unsupported architecture: X` — this is the **Burn engine** loader, which
  only loads `llama`, `qwen`, `deepseek` safetensors checkpoints.
  Mistral/Phi/Gemma/Mixtral registry entries are planned placeholders there.
  If you need a wider range of architectures or a quantized model, use a
  registry entry with `engine = "llamacpp"` instead — GGUF architecture
  support comes from upstream llama.cpp, not this per-family loader (worker
  must be built with `--features llamacpp`; see
  [deployment.md](deployment.md)).
- `Qwen3 checkpoints are not supported yet` — use a Qwen2.5 checkpoint (Burn engine only).
- `quantization ... is not implemented` — this is the Burn engine's
  load-time `quantization` field; request `"quantization": "none"` (Burn
  always loads FP32). To actually run a quantized model, pick a GGUF
  registry entry (`engine = "llamacpp"`) instead — quantization there comes
  from the GGUF file, not this field.
- 401/403 from HuggingFace — set `HF_TOKEN` in `.env` (gated repos like Llama 3).
- `Out of memory on GPU N` — unload something first:
  `curl -X DELETE http://localhost:8000/v1/models/<name>` — or check
  `GET /v1/workers` for `available_gb`.
- Disk: weights cache under the worker's `model_cache_dir` (Docker volume `./models`).

## Inference problems

- **504 Gateway Timeout** — generation exceeded `COORDINATOR_REQUEST_TIMEOUT`
  (default 300 s) or the worker's `request_timeout_secs`. Long prompts on CPU
  fallback are the usual cause — verify a real GPU is selected (worker log line
  `Selected WGPU Device`).
- **503 "No available workers"** — every worker is unhealthy, at its
  concurrency cap, or lacks memory for the model. Check `GET /v1/workers`.
- **RESOURCE_EXHAUSTED from the worker** — `max_concurrent_requests` (worker.toml) reached.
- **Prompt rejected: `prompt is N tokens but the model's max_seq_len is M`** —
  shorten the prompt; the worker refuses instead of silently truncating.
- **Streaming**: only `POST /v1/chat/completions` streams (SSE);
  `/v1/completions` always buffers. There is no WebSocket endpoint.

## Agentic serving (`engine = "llamaserver"`)

- **`failed to spawn llama-server binary ...`** — the worker can't find
  `llama-server`. The Docker worker image ships it at
  `/usr/local/bin/llama-server` (env `LLAMASERVER_BINARY_PATH` preset); on bare
  metal install it and put it on `PATH` or set `LLAMASERVER_BINARY_PATH`
  (see [deployment.md](deployment.md) "llama-server for agentic serving"). A
  pure-Burn image built with `--build-arg LLAMASERVER_SRC=llamaserver-none` has
  only a stub that exits 127 — rebuild the default worker image to serve
  llamaserver models.
- **`llama-server ... did not become healthy` / `exited during startup`** — the
  child died before its `/health` returned 200. Usual causes: the model OOM'd the
  GPU (lower `n_ctx`/`parallel` or add MoE CPU-offload via `llamaserver.extra_args`
  in `config/models.toml`), the `llamaserver_port` is already in use, or a bad
  `extra_args` flag. Check `docker compose logs worker` for the child's stderr.
- **Coordinator returns 404/502/504 for a llamaserver model** — the coordinator
  proxies to `http://<worker_host>:<llamaserver_port>`; it must be able to reach
  that port over the LAN. Publish 8081/8082 on the worker container (Docker) or
  open them in the worker's firewall (bare metal). 404 with a "load the model"
  message means it isn't loaded anywhere yet.
- **Agent sees garbled/empty tool-call `arguments`** — your `llama-server` build
  predates the mid-March-2026 fix (llama.cpp issue #20198 / PR #20213). Rebuild
  at the pinned tag (**b9941** or newer); the Docker image already pins it.

## Monitoring gaps

- Coordinator metrics: `http://localhost:8000/metrics` (there is NO :9090 coordinator port).
- Worker metrics: `http://localhost:9091/metrics` (+GPU_INDEX per replica).
- Prometheus UI: `http://localhost:9099`, Grafana: `http://localhost:3000`.
- Flat GPU panels usually mean the vendor tool (nvidia-smi) isn't available in
  the container — telemetry refreshes from it at scrape time.

## Converting models

```bash
# model id is POSITIONAL; there are no --model/--device flags
python scripts/convert_model.py deepseek-ai/deepseek-llm-7b-base --output ./models --quantize fp16
python scripts/convert_model.py --list-models
# heavy deps: pip install -r scripts/requirements-scripts.txt
```

## Collecting diagnostics

```bash
docker compose ps
docker compose logs --tail=200 coordinator worker
curl -s localhost:8000/health && curl -s localhost:8000/v1/workers
curl -s localhost:9091/metrics | grep -E "worker_(errors|requests)_total"
```
(There is no `scripts/diagnose.sh` — the commands above are the supported path.)
