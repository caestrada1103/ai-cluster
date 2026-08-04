# Client Setup — Agentic Serving

Practical, copy-pasteable configuration for pointing common coding clients at the AI Cluster
coordinator over your LAN. This assumes the coordinator is already running and at least one
`engine = "llamaserver"` model (e.g. `devstral-small-2-24b-instruct-gguf` or
`qwen3-coder-30b-a3b-instruct-gguf`, see `config/models.toml`) is loaded on a worker — for the
server side (starting the coordinator/worker, building `llama-server`, firewalling ports) see
[docs/deployment.md](deployment.md). This page only covers the client half.

Throughout, replace `<host>` with the coordinator machine's LAN IP or hostname and
`<model>` with the exact model name from `GET /v1/models` (e.g. `devstral-small-2-24b-instruct-gguf`).
The coordinator listens on port `8000` by default. If the coordinator has
`COORDINATOR_API_KEYS` set (see [docs/deployment.md](deployment.md) and
`coordinator/auth.py`), every client below needs a valid key via `Authorization: Bearer <key>`
or `x-api-key: <key>`; if it's unset, any value (or none) works.

To sanity-check the server before configuring a client, run
`python scripts/validate_agentic.py --base-url http://<host>:8000`.

## Quick reference

| Client | Protocol used | Needs native tool calling? |
|---|---|---|
| Claude Code CLI | Anthropic `POST /v1/messages` (+ `/v1/messages/count_tokens`) | Yes — Anthropic `tools`/`tool_use` |
| Cline / Roo Code | OpenAI `POST /v1/chat/completions` ("OpenAI Compatible" provider) | No — works via XML-formatted tool instructions in the prompt |
| aider | OpenAI `POST /v1/chat/completions` | No — aider's own diff/edit format, no `tools` field |
| Continue | OpenAI `POST /v1/chat/completions` (chat); autocomplete via `POST /v1/infill` | No for chat; autocomplete uses the coordinator's `/v1/infill` FIM proxy |
| Open WebUI | OpenAI `POST /v1/chat/completions`, `GET /v1/models` | No |
| OpenCode / Goose / Crush | OpenAI `POST /v1/chat/completions` with `tools` | Yes — OpenAI `tools`/`tool_calls` |

"Needs native tool calling" means the client requires the server to actually execute
llama.cpp's tool-calling grammar and return `tool_calls` / `tool_use` blocks — only
`engine = "llamaserver"` models support this (see
`pending-work/13-agentic-serving-llama-server.md` for the underlying contract).
Clients marked "No" work against any
model the coordinator serves, including the in-process `llamacpp`/`burn` engines, because
they never send a `tools` field.

---

## Claude Code CLI

Claude Code talks the **Anthropic** wire format (`POST /v1/messages`), which the coordinator
proxies only for `engine = "llamaserver"` models (see `coordinator/api.py::create_message`).
Point it at the coordinator with three environment variables:

```bash
export ANTHROPIC_BASE_URL=http://<host>:8000
export ANTHROPIC_AUTH_TOKEN=<coordinator API key, or any non-empty placeholder if auth is off>
export ANTHROPIC_MODEL=<model>   # e.g. devstral-small-2-24b-instruct-gguf — must match GET /v1/models exactly
claude
```

- `ANTHROPIC_AUTH_TOKEN` must be **some** non-empty value even when the coordinator has no
  `COORDINATOR_API_KEYS` configured — Claude Code needs a credential present to skip its normal
  OAuth login flow; the coordinator ignores it when auth is off.
- `ANTHROPIC_MODEL` must be the exact registry name from `GET /v1/models`, not an official
  Anthropic model ID (`claude-opus-4-8` etc.) — the coordinator resolves the `model` field
  against `config/models.toml` and 404s on an unknown name.

**If you see `400` errors** (most often on the first request or right after a tool call):
Claude Code sends `anthropic-beta` feature-opt-in headers by default that llama-server's
`/v1/messages` implementation may not recognize. Disable them:

```bash
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
```

Re-run the failing request. If `400`s persist, capture the response body — llama-server's
error detail usually names the unsupported field — and compare against
`scripts/validate_agentic.py`'s check `f` (`POST /v1/messages`), which exercises this exact
path independent of the CLI.

## Cline / Roo Code

Both are VS Code extensions and both talk OpenAI's `/v1/chat/completions`. Neither requires
server-side native tool calling — they format tool instructions as XML directly in the prompt
and parse the model's XML reply themselves, so they work against **any** coordinator model
(not just `engine = "llamaserver"`), though llamaserver models still give better instruction
following.

In the extension's settings:

1. **API Provider**: `OpenAI Compatible`
2. **Base URL**: `http://<host>:8000/v1`
3. **API Key**: your coordinator API key, or any placeholder if auth is off (the field usually
   can't be left empty)
4. **Model**: type the model name manually (e.g. `devstral-small-2-24b-instruct-gguf`) — it
   won't appear in a dropdown since the coordinator isn't a recognized provider preset

## aider

aider speaks OpenAI's API and never sends a `tools` field (it uses its own diff/whole-file
edit formats), so it works with any coordinator model:

```bash
export OPENAI_API_BASE=http://<host>:8000/v1
export OPENAI_API_KEY=<coordinator API key, or any placeholder if auth is off>
aider --model openai/<model>   # e.g. openai/devstral-small-2-24b-instruct-gguf
```

The `openai/` prefix tells aider to use the generic OpenAI-compatible backend instead of
trying to resolve `<model>` against its built-in model list.

## Continue

Continue (VS Code / JetBrains) supports an `openai` provider type pointed at a custom
`apiBase`. Add to `config.json` (or the equivalent `config.yaml` provider block):

```json
{
  "models": [
    {
      "title": "AI Cluster",
      "provider": "openai",
      "model": "<model>",
      "apiBase": "http://<host>:8000/v1",
      "apiKey": "<coordinator API key, or any placeholder if auth is off>"
    }
  ]
}
```

This wires up **chat**. For Continue's autocomplete (tab-completion), the coordinator proxies
llama-server's fill-in-the-middle endpoint at **`POST /v1/infill`**: point
`tabAutocompleteModel` at provider `llama.cpp` with `apiBase: "http://<coordinator-host>:8000/v1"`
(Continue appends `/infill` to that base — verify against your Continue version). If more than
one llamaserver model is loaded, include `"model": "<name>"` in the request options; with exactly
one loaded, the coordinator resolves it automatically.

## Open WebUI

Open WebUI already ships in this repo's `docker-compose.yml` pointed at the coordinator; for
a standalone instance, add the coordinator as an OpenAI connection:

1. **Settings → Connections → OpenAI API**
2. **API Base URL**: `http://<host>:8000/v1`
3. **API Key**: your coordinator API key, or any placeholder if auth is off

Open WebUI calls `GET /v1/models` to populate its model picker and
`POST /v1/chat/completions` for chat — both work against any coordinator model.

## OpenCode / Goose / Crush

These agentic coding CLIs use OpenAI's `tools` / `tool_calls` mechanism (not Anthropic's), so
— like Claude Code — they only work against `engine = "llamaserver"` models, and specifically
need `POST /v1/chat/completions` with a non-empty `tools` array to round-trip correctly. Point
each at `http://<host>:8000/v1` as its OpenAI-compatible base URL with the coordinator API key
(or a placeholder if auth is off), and set the model to the exact registry name. Consult each
tool's own docs for its specific config file/env var names — the request shape is identical to
Cline/aider above, only the `tools` field usage differs. `scripts/validate_agentic.py` check
`e` exercises the same tool-calling path these clients depend on; run it first if a client's
tool calls aren't round-tripping to rule out a server-side issue.

---

## Troubleshooting

- **`401`**: the coordinator has `COORDINATOR_API_KEYS` set and your client sent no key, the
  wrong key, or put it in the wrong header. Every client above accepts `Authorization: Bearer
  <key>` — most map it directly from their "API Key" field. Verify with
  `scripts/validate_agentic.py --api-key <key>` (checks `g`).
- **`404` on a request with a `model` field**: the name doesn't match `GET /v1/models` exactly,
  or (for Claude Code / OpenCode / Goose / Crush) the model isn't loaded on any worker — load
  it with `POST /v1/models/load {"model_name": "<model>"}` first.
- **`501` from `POST /v1/messages`**: the model exists but its `engine` isn't `llamaserver`
  (only llamaserver-engine models serve the Anthropic surface) — pick one of the models listed
  in `config/models.toml` under `engine = "llamaserver"`.
- **Streaming looks stalled / hangs**: confirm the coordinator process can reach the worker's
  llama-server port directly (`curl http://<worker-host>:<llamaserver_port>/health`) — the
  proxy adds one hop and never buffers, so a stall usually means the upstream llama-server
  itself isn't responding.
- **General**: `python scripts/validate_agentic.py --base-url http://<host>:8000 --model
  <model>` runs the same checks (models list, chat stream/non-stream, tools round-trip,
  `/v1/messages`, auth, timing) any of these clients depend on, independent of client-specific
  quirks.

See [docs/deployment.md](deployment.md) for the server side: starting the coordinator and
worker, building/provisioning the `llama-server` binary, and firewalling the llamaserver ports
to the trusted LAN.
