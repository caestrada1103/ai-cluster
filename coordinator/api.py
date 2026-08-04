"""API routes for the AI Cluster coordinator.

Provides the FastAPI router mounted at ``/v1`` in main.py.
Endpoints:
    POST /completions  - Run inference
    GET  /models       - List available models
    POST /models/load  - Load a model onto a worker
    GET  /workers      - List connected workers
"""
import asyncio
import ipaddress
import json
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, TypeVar, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from coordinator import auth, proxy
from coordinator.models import ModelConfig, ModelRegistry

logger = logging.getLogger(__name__)

_BodyT = TypeVar("_BodyT", bound=BaseModel)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """Schema for a single chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Body for the POST /chat/completions endpoint (OpenAI compatible)."""

    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(512, ge=1, le=32768)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(40, ge=0)
    stream: Optional[bool] = False
    session_id: Optional[str] = Field(
        None, description="Sticky-session key for affinity routing (optional)"
    )
    compress_context: Optional[bool] = Field(
        None,
        description=(
            "Override the server's context-compression default for this request only "
            "(true forces it on, false forces it off, omitted uses the server default)"
        ),
    )


class CompletionRequest(BaseModel):
    """Body for the POST /completions endpoint."""

    model: str = Field(..., description="Model name, e.g. 'deepseek-7b'")
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(512, ge=1, le=32768)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=0)
    stream: bool = False
    worker_id: Optional[str] = Field(
        None, description="Optional worker ID to force routing to a specific GPU"
    )
    session_id: Optional[str] = Field(
        None, description="Sticky-session key for affinity routing (optional)"
    )
    compress_context: Optional[bool] = Field(
        None,
        description=(
            "Override the server's context-compression default for this request only "
            "(true forces it on, false forces it off, omitted uses the server default)"
        ),
    )


class CompletionResponse(BaseModel):
    """Response from the POST /completions endpoint."""

    request_id: str
    text: str
    tokens_generated: int
    processing_time_ms: float
    worker_id: Optional[str] = None


class LoadModelRequest(BaseModel):
    """Body for the POST /models/load endpoint."""

    model_name: str
    worker_id: Optional[str] = None
    quantization: str = "none"  # only "none" is accepted by workers today; others are planned


class LoadModelResponse(BaseModel):
    """Response from the POST /models/load endpoint."""

    status: str
    model_name: str
    worker_id: Optional[str] = None
    memory_used_gb: Optional[float] = None
    message: Optional[str] = None


class ModelInfo(BaseModel):
    """Schema for a single model entry, compatible with OpenAI API."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "custom"

    # Custom AI Cluster extensions
    family: Optional[str] = None
    parameters: Optional[str] = None
    min_memory_gb: Optional[float] = None
    loaded_on: List[Dict[str, Any]] = []
    supports_quantization: List[str] = []


class ModelsResponse(BaseModel):
    """Schema for the /models response, compatible with OpenAI API."""

    object: str = "list"
    data: List[ModelInfo]


class WorkerInfoResponse(BaseModel):
    """Schema for a single worker entry."""

    id: str
    address: str
    state: str
    gpus: List[Dict[str, Any]] = []
    loaded_models: List[str] = []
    active_requests: int = 0


# ---------------------------------------------------------------------------
# Helper to get the coordinator from the request
# ---------------------------------------------------------------------------


def _get_coordinator(request: Request) -> Any:  # type: ignore[type-arg]
    """Retrieve the ClusterCoordinator stored in app state."""
    coordinator = getattr(request.app.state, "coordinator", None)
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    return coordinator


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _parse_model_and_worker(model_string: str) -> tuple[str, Optional[str]]:
    """Parse 'model@worker_id' syntax."""
    if "@" in model_string:
        parts = model_string.split("@", 1)
        return parts[0], parts[1]
    return model_string, None


# ---------------------------------------------------------------------------
# Engine dispatch + transparent llama-server proxy (Plan 13 Task 2)
# ---------------------------------------------------------------------------


async def _read_json_body(request: Request) -> tuple[bytes, Dict[str, Any]]:  # type: ignore[type-arg]
    """Read the RAW request bytes once and parse them as a JSON object.

    The proxy path forwards these raw bytes to llama-server unchanged (so
    OpenAI ``tools``/``tool_calls`` and other unknown fields survive), while the
    in-process path re-validates the same dict through pydantic. Reading the
    body here — instead of declaring a pydantic body parameter — is exactly what
    lets an agentic request with ``content: null`` tool messages reach the proxy
    rather than 422-ing at FastAPI's validation layer first.
    """
    raw = await request.body()
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return raw, parsed


def _parse_body(model_cls: Type[_BodyT], data: Dict[str, Any]) -> _BodyT:
    """Validate a raw body dict through a pydantic model, 422 on failure.

    Used only on the in-process path — mirrors the validation FastAPI would have
    done had the route declared a typed body parameter.
    """
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc


def _lookup_engine_model(model_string: Any) -> Optional[ModelConfig]:
    """Resolve a request's ``model`` field to a registry entry (None if unknown).

    Strips the AICluster ``model@worker`` suffix before the lookup so engine
    dispatch sees the bare model name.
    """
    if not isinstance(model_string, str):
        return None
    model_name, _ = _parse_model_and_worker(model_string)
    return ModelRegistry.get_model(model_name)


def _worker_host(address: str) -> str:
    """Host portion of a worker's gRPC address ('host:port' -> 'host')."""
    return address.rsplit(":", 1)[0] if ":" in address else address


# ---------------------------------------------------------------------------
# C3: POST /v1/workers/manual address validation
# ---------------------------------------------------------------------------

#: Hard cap on addresses accepted in a single POST /v1/workers/manual call.
_MAX_MANUAL_WORKERS_PER_REQUEST = 16

_HOSTNAME_RE = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*$")


def _split_host_port(address: str) -> tuple[str, int]:
    """Split 'host:port' or '[ipv6]:port' into (host, port); ValueError on any other shape."""
    if address.startswith("["):
        end = address.find("]")
        if end == -1 or address[end + 1 : end + 2] != ":":
            raise ValueError(f"invalid address '{address}': expected '[ipv6-host]:port'")
        host, port_str = address[1:end], address[end + 2 :]
    else:
        if address.count(":") != 1:
            raise ValueError(f"invalid address '{address}': expected 'host:port'")
        host, port_str = address.rsplit(":", 1)
    if not host:
        raise ValueError(f"invalid address '{address}': empty host")
    if not port_str.isdigit():
        raise ValueError(f"invalid address '{address}': port must be numeric")
    port = int(port_str)
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid address '{address}': port {port} out of range 1-65535")
    return host, port


def _is_well_formed_host(host: str) -> bool:
    """True for a syntactically valid IPv4/IPv6 literal or DNS hostname."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    return bool(_HOSTNAME_RE.match(host)) and len(host) <= 253


def _host_allowed(host: str, allowed: List[str]) -> bool:
    """Empty allowlist == accept any well-formed host (the auth+opt-in gate
    on the route is the primary control then). Otherwise the host must
    exactly match an allowed entry, OR — if the entry parses as a network —
    fall inside that CIDR."""
    if not allowed:
        return True
    for entry in allowed:
        if entry == host:
            return True
        try:
            if ipaddress.ip_address(host) in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def _validate_manual_worker_address(address: str, allowed_hosts: List[str]) -> None:
    """Raise ValueError for anything that isn't a plausible, allowed worker address."""
    host, _port = _split_host_port(address)
    if not _is_well_formed_host(host):
        raise ValueError(f"invalid address '{address}': '{host}' is not a valid host")
    if not _host_allowed(host, allowed_hosts):
        raise ValueError(f"host '{host}' in address '{address}' is not in the configured allowlist")


def _proxy_response(result: proxy.ProxyResponse) -> Response:
    """Adapt a proxy result into a FastAPI response.

    Preserves the upstream status code and Content-Type; streams SSE verbatim
    (never buffered) and buffers ordinary JSON replies. Content-Type is set via
    ``media_type`` so it is not duplicated by the forwarded header set.
    """
    headers = dict(result.headers)
    media_type = headers.pop("content-type", None)
    if isinstance(result, proxy.StreamingProxyResponse):
        return StreamingResponse(
            result.body,
            status_code=result.status_code,
            media_type=media_type,
            headers=headers or None,
        )
    return Response(
        content=result.content,
        status_code=result.status_code,
        media_type=media_type,
        headers=headers or None,
    )


def _not_loaded_detail(model_name: str) -> str:
    """404 message for a llamaserver model that no worker holds (auto-load off)."""
    return (
        f"Model '{model_name}' (engine=llamaserver) is not loaded on any worker. "
        "Load it first with POST /v1/models/load."
    )


async def _resolve_llamaserver_worker(coordinator: Any, model_cfg: ModelConfig) -> Any:
    """Find — or, when auto-load is on, load — a worker serving ``model_cfg``.

    Plan 13 Task 5. The already-loaded common case takes the fast path
    (``find_worker_for_model``) untouched. When no worker reports the model
    loaded, ``settings.llamaserver_autoload`` decides: True (default) triggers
    the coordinator's single-flight auto-load and returns the freshly-loaded
    worker (a load failure/timeout surfaces as 503); False preserves the
    Phase-1 behavior — 404 pointing at ``POST /v1/models/load``.
    """
    worker = await coordinator.find_worker_for_model(model_cfg.name)
    if worker is not None:
        return worker

    settings = getattr(coordinator, "settings", None)
    if not bool(getattr(settings, "llamaserver_autoload", True)):
        raise HTTPException(status_code=404, detail=_not_loaded_detail(model_cfg.name))

    try:
        return await coordinator.ensure_llamaserver_model_loaded(model_cfg.name)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Auto-load of model '{model_cfg.name}' failed: {exc}",
        ) from exc


#: H5 — llama-server has no admission control of its own for these fields;
#: the in-process path already bounds max_tokens (CompletionRequest/
#: ChatCompletionRequest: `le=32768`) via pydantic, but the proxy path
#: forwards raw bytes and skips that entirely. This ceiling is looser
#: (agentic/long-context llamaserver models legitimately want more headroom)
#: but still bounded rather than "anything the client sends, verbatim".
_PROXY_MAX_TOKENS_CEILING = 131_072


def _validate_proxy_envelope(data: Dict[str, Any]) -> None:
    """Minimal sanity checks on a proxied request body BEFORE it is
    forwarded verbatim (H5).

    Deliberately NOT a full re-model of the OpenAI/Anthropic schema — that
    would break forward compatibility with ``tools``/``tool_calls``/
    ``thinking``/... fields, which is the entire point of proxying raw
    bytes. Only the couple of fields llama-server has no bound of its own
    for are checked.
    """
    max_tokens = data.get("max_tokens")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, float)):
            raise HTTPException(status_code=422, detail="'max_tokens' must be a number")
        if not (1 <= max_tokens <= _PROXY_MAX_TOKENS_CEILING):
            raise HTTPException(
                status_code=422,
                detail=f"'max_tokens' must be between 1 and {_PROXY_MAX_TOKENS_CEILING}",
            )
    stream = data.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise HTTPException(status_code=422, detail="'stream' must be a boolean")


async def _proxy_to_llamaserver(
    request: Request,  # type: ignore[type-arg]
    coordinator: Any,
    model_cfg: ModelConfig,
    raw_body: bytes,
    stream: bool,
    *,
    data: Optional[Dict[str, Any]] = None,
    upstream_path: Optional[str] = None,
) -> Response:
    """Forward a request to the worker-local llama-server serving ``model_cfg``.

    Resolves a serving worker (auto-loading on demand — Plan 13 Task 5 — via
    :func:`_resolve_llamaserver_worker`), builds
    ``http://<worker_host>:<port><path>`` and passes the raw body straight
    through. ``path`` defaults to the incoming ``request.url.path``; callers pass
    ``upstream_path`` to override it (``/infill`` is served at llama-server's
    root, not under the coordinator's ``/v1`` mount — see :func:`create_infill`).
    ``data`` (H5), when given, is sanity-checked via
    :func:`_validate_proxy_envelope` before anything is dialed.
    """
    if data is not None:
        _validate_proxy_envelope(data)
    worker = await _resolve_llamaserver_worker(coordinator, model_cfg)
    if model_cfg.llamaserver_port is None:  # defensive: validated at registry load
        raise HTTPException(
            status_code=500,
            detail=f"Model '{model_cfg.name}' has no llamaserver_port configured",
        )
    host = _worker_host(worker.address)
    path = upstream_path if upstream_path is not None else request.url.path
    url = f"http://{host}:{model_cfg.llamaserver_port}{path}"
    headers = proxy.filter_request_headers(request.headers)
    result = await proxy.proxy_request(request.method, url, raw_body, headers, stream)
    return _proxy_response(result)


def _require_llamaserver_model(model_string: Any, *, surface: str = "this endpoint") -> ModelConfig:
    """Resolve a request body's ``model`` to a llamaserver registry entry.

    404 when the model is unknown; 501 when it exists but is not a
    llamaserver-engine model (these surfaces — Anthropic ``/v1/messages``,
    ``/v1/embeddings``, ``/infill`` — have no in-process path; only llama-server
    serves them). ``surface`` names the endpoint in the 501 message.
    """
    model_cfg = _lookup_engine_model(model_string)
    if model_cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_string}'")
    if model_cfg.engine != "llamaserver":
        raise HTTPException(
            status_code=501,
            detail=(
                f"Model '{model_cfg.name}' (engine={model_cfg.engine}) does not support "
                f"{surface}; only engine=llamaserver models do."
            ),
        )
    return model_cfg


async def _resolve_single_loaded_llamaserver(coordinator: Any) -> ModelConfig:
    """Resolve ``/infill`` when the body omits ``model`` (Plan 13 Task 6).

    Uses the sole llamaserver model currently loaded across the fleet; raises a
    400 telling the client to specify ``model`` when zero or more than one are
    loaded (the request is genuinely ambiguous).
    """
    loaded = await coordinator.loaded_llamaserver_models()
    if len(loaded) == 1:
        cfg = ModelRegistry.get_model(loaded[0])
        if cfg is not None:
            return cfg
    if not loaded:
        hint = "no engine=llamaserver model is currently loaded"
    else:
        hint = f"multiple llamaserver models are loaded ({', '.join(loaded)})"
    raise HTTPException(
        status_code=400,
        detail=f'POST /infill requires a "model" field: {hint}.',
    )


@router.post("/completions", response_model=None)
async def create_completion(request: Request) -> Any:  # type: ignore[type-arg]
    """Run inference on the cluster (OpenAI text-completion endpoint).

    Resolves the requested model's engine FIRST (Plan 13 Task 2): a model with
    ``engine == "llamaserver"`` is proxied verbatim to its worker-local
    llama-server (raw body + SSE passthrough, context compression skipped —
    Phase 2); every other engine runs the existing in-process path below.
    """
    coordinator = _get_coordinator(request)

    raw_body, data = await _read_json_body(request)
    model_cfg = _lookup_engine_model(data.get("model"))
    if model_cfg is not None and model_cfg.engine == "llamaserver":
        return await _proxy_to_llamaserver(
            request, coordinator, model_cfg, raw_body, bool(data.get("stream", False)), data=data
        )

    body = _parse_body(CompletionRequest, data)
    model_name, target_worker = _parse_model_and_worker(body.model)
    worker_id = body.worker_id or target_worker

    from coordinator.context_compression import maybe_compress_prompt

    prompt = await maybe_compress_prompt(
        body.prompt, coordinator=coordinator, override_enabled=body.compress_context
    )

    try:
        result = await coordinator.infer(
            model_name=model_name,
            prompt=prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            stream=False,  # /v1/completions is buffered; only /v1/chat/completions streams
            worker_id=worker_id,
            session_id=body.session_id,
        )
        return CompletionResponse(**result)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        # H4: an unregistered model_name with pull-through disabled raises
        # ValueError from coordinator._load_model_on_worker — a client
        # error (bad model name), not a server error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        # M12: never echo a raw exception string to the client (it can leak
        # internal paths/library internals/stack details) — the real error
        # is already captured server-side by logger.exception above.
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Internal server error") from None


# ---------------------------------------------------------------------------
# Chat prompt templates for the in-process engine path (item 4)
#
# Plan 13 Phase 3 will delete this whole section once the in-process chat path
# is deprecated in favor of engine="llamaserver" (which uses llama.cpp's real,
# per-model Jinja chat templates + native tool calling). Until then, a single
# hardcoded Zephyr-style template was used for EVERY in-process model
# regardless of family, which is wrong for anything that isn't Zephyr/similar:
# verified on hardware with a Qwen model, replies terminated correctly but
# were then followed by a spurious "<|user|>" turn replaying the prompt,
# because the model was never trained on Zephyr's "</s>"-separated turns and
# the coordinator had no stop sequence to cut the reply at the right point.
#
# Pragmatic fix (NOT a full Jinja/minja templating engine): pick a
# template per the registry's ModelConfig.family, and always truncate the
# final text at the first occurrence of that template's stop markers — this
# bounds the damage even for families/templates that are only an
# approximation of the model's real fine-tuning format.
# ---------------------------------------------------------------------------

_ChatPromptBuilder = Any  # Callable[[List[ChatMessage]], str] — Any avoids a forward-ref headache


def _build_chatml_prompt(messages: List[ChatMessage]) -> str:
    """ChatML (Qwen, and many others fine-tuned on the same format)."""
    prompt = ""
    for msg in messages:
        role = msg.role.lower()
        prompt += f"<|im_start|>{role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def _build_llama3_prompt(messages: List[ChatMessage]) -> str:
    """Llama-3/3.1-Instruct header-block format."""
    prompt = "<|begin_of_text|>"
    for msg in messages:
        role = msg.role.lower()
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{msg.content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def _build_mistral_prompt(messages: List[ChatMessage]) -> str:
    """Mistral-Instruct [INST]/[/INST] format; system content folds into the
    next user turn (Mistral has no separate system role)."""
    parts: List[str] = []
    pending_system = ""
    for msg in messages:
        role = msg.role.lower()
        if role == "system":
            pending_system += msg.content + "\n\n"
        elif role == "user":
            content = pending_system + msg.content if pending_system else msg.content
            pending_system = ""
            parts.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            parts.append(f" {msg.content}</s>")
    return "".join(parts)


def _build_gemma_prompt(messages: List[ChatMessage]) -> str:
    """Gemma <start_of_turn>/<end_of_turn> format; system content folds into
    the next user turn (Gemma has no separate system role)."""
    prompt = ""
    pending_system = ""
    for msg in messages:
        role = msg.role.lower()
        if role == "system":
            pending_system += msg.content + "\n\n"
            continue
        gemma_role = "model" if role == "assistant" else "user"
        content = msg.content
        if gemma_role == "user" and pending_system:
            content = pending_system + content
            pending_system = ""
        prompt += f"<start_of_turn>{gemma_role}\n{content}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt


def _build_phi_prompt(messages: List[ChatMessage]) -> str:
    """Phi-3-style format — structurally like the old fallback but with
    Phi's actual end-of-turn token (``<|end|>``) instead of Zephyr's
    ``</s>``."""
    prompt = ""
    for msg in messages:
        role = msg.role.lower()
        if role in ("system", "user", "assistant"):
            prompt += f"<|{role}|>\n{msg.content}<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt


def _build_deepseek_prompt(messages: List[ChatMessage]) -> str:
    """DeepSeek-V2/V3-style format (best-effort — DeepSeek checkpoints vary
    more than most families; the stop sequences below are the real safety
    net for this one)."""
    prompt = ""
    for msg in messages:
        role = msg.role.lower()
        if role == "system":
            prompt += f"{msg.content}\n\n"
        elif role == "user":
            prompt += f"<｜User｜>{msg.content}"
        elif role == "assistant":
            prompt += f"<｜Assistant｜>{msg.content}<｜end▁of▁sentence｜>"
    prompt += "<｜Assistant｜>"
    return prompt


def _build_zephyr_prompt(messages: List[ChatMessage]) -> str:
    """The original hardcoded Zephyr-ish fallback template, used only when
    the model's family has no dedicated template above (or the model is
    unregistered)."""
    prompt = ""
    for msg in messages:
        role = msg.role.lower()
        content = msg.content
        if role == "system":
            prompt += f"<|system|>\n{content}</s>\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}</s>\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}</s>\n"
    prompt += "<|assistant|>\n"
    return prompt


# family.value (coordinator/models.py ModelFamily) -> (builder, stop sequences)
_FAMILY_CHAT_TEMPLATES: Dict[str, tuple[_ChatPromptBuilder, List[str]]] = {
    "qwen": (_build_chatml_prompt, ["<|im_end|>", "<|im_start|>"]),
    "llama": (_build_llama3_prompt, ["<|eot_id|>", "<|start_header_id|>"]),
    "mistral": (_build_mistral_prompt, ["</s>", "[INST]"]),
    "gemma": (_build_gemma_prompt, ["<end_of_turn>", "<start_of_turn>"]),
    "phi": (_build_phi_prompt, ["<|end|>", "<|user|>"]),
    "deepseek": (
        _build_deepseek_prompt,
        ["<｜end▁of▁sentence｜>", "<｜User｜>"],
    ),
}

_ZEPHYR_FALLBACK_STOP = ["<|user|>", "<|system|>"]


def _select_chat_template(
    model_cfg: Optional[ModelConfig], model_name: str
) -> tuple[_ChatPromptBuilder, List[str]]:
    """Pick a (prompt builder, stop sequences) pair for ``model_name``.

    Model-aware when the registry knows the model's ``family``; otherwise
    (unregistered model, or a family with no dedicated template yet) falls
    back to the historical one-size-fits-all template, WITH stop sequences
    added so a spurious replayed turn still gets truncated. Logs a warning
    pointing at ``engine="llamaserver"`` as the real fix — see the module
    docstring for this section.
    """
    family = model_cfg.family.value if model_cfg is not None else None
    template = _FAMILY_CHAT_TEMPLATES.get(family) if family else None
    if template is not None:
        return template
    logger.warning(
        "Model '%s' (family=%s) has no model-aware chat template; falling back to the "
        "generic template. This can still produce lower-quality output (e.g. a spurious "
        "extra turn after the real answer) for models that don't match it. For best "
        'quality and tool-calling support, set engine="llamaserver" for this model in '
        "config/models.toml instead of relying on the in-process engine's flattened prompt.",
        model_name,
        family,
    )
    return _build_zephyr_prompt, _ZEPHYR_FALLBACK_STOP


def _truncate_at_stop(text: str, stop_sequences: List[str]) -> str:
    """Cut ``text`` at the earliest occurrence of any ``stop_sequences`` entry.

    Applied to the model's generated text (never the input prompt) so a
    spurious replayed turn — the model continuing past where it should have
    stopped, e.g. emitting ``<|user|>`` and restating the prompt — never
    reaches the client, regardless of which template above produced the
    prompt.
    """
    cut: Optional[int] = None
    for stop in stop_sequences:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1 and (cut is None or idx < cut):
            cut = idx
    return text if cut is None else text[:cut]


def _build_flat_response(result: Dict[str, Any], model: str, prompt: str) -> Dict[str, Any]:
    """Build a standard OpenAI-compatible chat completion response.

    ``prompt_tokens`` for the in-process engine path: the gRPC
    ``InferenceResponse`` (proto/cluster.proto) has no prompt-token-count
    field today — the worker never counts or sends one, so the coordinator
    cannot report the model's true prompt token count without a worker/proto
    change (out of scope here — see AGENTS.md ownership boundaries). Instead
    we estimate from the flattened prompt text using the same coarse,
    offline, model-agnostic heuristic already used for the context-compression
    budget check (``coordinator/context_compression/tokenizer.py``), so
    ``total_tokens`` is at least internally consistent
    (``prompt_tokens + completion_tokens``) instead of hardcoded to 0. The
    proxied ``llamaserver`` path is unaffected — llama-server counts real
    tokens itself.
    """
    from coordinator.context_compression.tokenizer import estimate_tokens

    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = result["tokens_generated"]
    return {
        "id": result["request_id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _stream_chat_completion(
    coordinator: Any, ctx: Any, model: str, timeout: float, stop_sequences: List[str]
) -> AsyncGenerator[str, None]:
    """Stream chunks live from the request's token queue as the worker produces them.

    ``stop_sequences`` (item 4) are checked against the FULL text accumulated
    so far on every chunk — not just the newly-arrived piece — so a stop
    marker split across two queue messages is still caught. Once one is
    found, only the text up to the marker is emitted, the stream is closed as
    ``finish_reason: "stop"``, and no further chunks are read from the queue
    (the in-flight worker request keeps running to completion in the
    background, same as today's early-break-on-``finished`` path).
    """
    deadline = time.time() + timeout
    accumulated = ""
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                error_chunk = {"error": {"message": "request timed out", "type": "timeout"}}
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                break
            try:
                response = await asyncio.wait_for(
                    ctx.token_queue.get(), timeout=min(remaining, 1.0)
                )
            except asyncio.TimeoutError:
                if ctx.error:
                    error_chunk = {"error": {"message": ctx.error, "type": "internal_error"}}
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                continue  # still generating — poll again

            new_accumulated = _truncate_at_stop(accumulated + response.text, stop_sequences)
            delta = new_accumulated[len(accumulated) :]
            hit_stop = len(new_accumulated) < len(accumulated) + len(response.text)
            accumulated = new_accumulated

            chunk = {
                "id": ctx.id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        # Emit the text even on the final chunk (proto allows text+finished)
                        "delta": {"content": delta},
                        "finish_reason": "stop" if (response.finished or hit_stop) else None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            if response.finished or hit_stop:
                yield "data: [DONE]\n\n"
                break
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {"error": {"message": str(e), "type": "internal_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        coordinator.active_requests.pop(ctx.id, None)


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: Request,  # type: ignore[type-arg]
) -> Union[Dict[str, Any], Response, StreamingResponse]:
    """OpenAI-compatible chat completions endpoint used by Open-WebUI + agents.

    Resolves the model's engine FIRST (Plan 13 Task 2): ``engine ==
    "llamaserver"`` models are proxied verbatim to a worker-local llama-server
    so OpenAI ``tools``/streaming ``tool_calls`` pass through unmodified (raw
    body + SSE passthrough, context compression skipped — Phase 2). Every other
    engine runs the in-process path below, which selects a chat template by
    the registry's ``family`` (item 4 — see the "Chat prompt templates"
    section above) instead of one hardcoded Zephyr-style template for every
    model.
    """
    coordinator = _get_coordinator(request)

    raw_body, data = await _read_json_body(request)
    model_cfg = _lookup_engine_model(data.get("model"))
    if model_cfg is not None and model_cfg.engine == "llamaserver":
        return await _proxy_to_llamaserver(
            request, coordinator, model_cfg, raw_body, bool(data.get("stream", False)), data=data
        )

    body = _parse_body(ChatCompletionRequest, data)
    logger.info(f"Received chat completion request for model: {body.model}")

    model_name, worker_id = _parse_model_and_worker(body.model)

    from coordinator.context_compression import maybe_compress_chat_messages

    messages = await maybe_compress_chat_messages(
        body.messages, coordinator=coordinator, override_enabled=body.compress_context
    )

    # Convert chat history to a raw prompt using a template selected by the
    # model's family (falls back to the legacy generic template, with stop
    # sequences, for unregistered models or families with no dedicated
    # template yet — see _select_chat_template).
    build_prompt, stop_sequences = _select_chat_template(model_cfg, model_name)
    prompt = build_prompt(messages)

    try:
        if body.stream:
            # Return immediately and stream tokens as the worker produces them.
            ctx = await coordinator.submit_request(
                model_name=model_name,
                prompt=prompt,
                max_tokens=body.max_tokens or 512,
                temperature=body.temperature or 0.7,
                top_p=body.top_p or 0.95,
                top_k=body.top_k or 40,
                stream=True,
                worker_id=worker_id,
                session_id=body.session_id,
            )
            timeout = coordinator.settings.request_timeout
            return StreamingResponse(
                _stream_chat_completion(coordinator, ctx, body.model, timeout, stop_sequences),
                media_type="text/event-stream",
            )

        result = await coordinator.infer(
            model_name=model_name,
            prompt=prompt,
            max_tokens=body.max_tokens or 512,
            temperature=body.temperature or 0.7,
            top_p=body.top_p or 0.95,
            top_k=body.top_k or 40,
            stream=False,
            worker_id=worker_id,
            session_id=body.session_id,
        )
        result = dict(result, text=_truncate_at_stop(result["text"], stop_sequences))
        return _build_flat_response(result, body.model, prompt)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        # H4: an unregistered model_name with pull-through disabled raises
        # ValueError from coordinator._load_model_on_worker — a client
        # error (bad model name), not a server error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        # M12: never echo a raw exception string to the client (it can leak
        # internal paths/library internals/stack details) — the real error
        # is already captured server-side by logger.exception above.
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/messages", response_model=None)
async def create_message(request: Request) -> Response:  # type: ignore[type-arg]
    """Anthropic-format messages endpoint (Plan 13 Task 2), llamaserver-only.

    The in-process engines have no Anthropic templating, so only ``engine ==
    "llamaserver"`` models are served here: the coordinator proxies the raw body
    — ``system``, ``tools``, ``thinking`` and all — to a worker-local
    llama-server, which natively serves ``POST /v1/messages``. Anthropic clients
    request streaming via ``"stream": true`` in the body, which is sniffed here
    to decide SSE passthrough. Unknown model → 404; any other engine → 501.
    """
    coordinator = _get_coordinator(request)
    raw_body, data = await _read_json_body(request)
    model_cfg = _require_llamaserver_model(
        data.get("model"), surface="the Anthropic /v1/messages API"
    )
    return await _proxy_to_llamaserver(
        request, coordinator, model_cfg, raw_body, bool(data.get("stream", False)), data=data
    )


@router.post("/messages/count_tokens", response_model=None)
async def count_message_tokens(request: Request) -> Response:  # type: ignore[type-arg]
    """Anthropic ``POST /v1/messages/count_tokens`` (Plan 13 Task 2).

    llamaserver-only (unknown model → 404, other engine → 501) and never
    streaming — the raw body is proxied verbatim to the worker-local
    llama-server, whose token count is returned unchanged.
    """
    coordinator = _get_coordinator(request)
    raw_body, data = await _read_json_body(request)
    model_cfg = _require_llamaserver_model(
        data.get("model"), surface="the Anthropic /v1/messages/count_tokens API"
    )
    return await _proxy_to_llamaserver(request, coordinator, model_cfg, raw_body, False, data=data)


@router.post("/embeddings", response_model=None)
async def create_embeddings(request: Request) -> Response:  # type: ignore[type-arg]
    """OpenAI-compatible embeddings endpoint (Plan 13 Task 6), llamaserver-only.

    Resolves the model from the JSON body's ``model`` field and proxies the raw
    body — buffered, NEVER SSE — to the worker-local llama-server's
    ``/v1/embeddings``. Unknown model → 404; any other engine → 501. Auto-load
    (Task 5) applies via :func:`_proxy_to_llamaserver`.
    """
    coordinator = _get_coordinator(request)
    raw_body, data = await _read_json_body(request)
    model_cfg = _require_llamaserver_model(data.get("model"), surface="the /v1/embeddings API")
    return await _proxy_to_llamaserver(request, coordinator, model_cfg, raw_body, False, data=data)


@router.post("/infill", response_model=None)
async def create_infill(request: Request) -> Response:  # type: ignore[type-arg]
    """llama.cpp FIM ``/infill`` proxy (Plan 13 Task 6), llamaserver-only.

    The body MAY carry ``model``: when present it is resolved like the other
    routes; when absent, ``/infill`` falls back to the single llamaserver model
    currently loaded across the fleet (zero or multiple → 400 telling the client
    to specify ``model``). The raw body is forwarded verbatim — llama-server
    ignores unknown fields, so the ``model`` key is left in place. Streaming is
    honored only when the body sets ``"stream": true`` (llama-server's ``/infill``
    supports SSE), sniffed like ``/v1/messages``.

    NOTE: llama-server serves FIM at its ROOT ``/infill``, not under ``/v1``. This
    route is registered on the coordinator's ``/v1`` router (mounted in main.py,
    outside this task's ownership), so the upstream path is pinned to ``/infill``
    regardless of the coordinator-facing path.
    """
    coordinator = _get_coordinator(request)
    raw_body, data = await _read_json_body(request)
    model_field = data.get("model")
    if model_field is not None:
        model_cfg = _require_llamaserver_model(model_field, surface="the /infill API")
    else:
        model_cfg = await _resolve_single_loaded_llamaserver(coordinator)
    return await _proxy_to_llamaserver(
        request,
        coordinator,
        model_cfg,
        raw_body,
        bool(data.get("stream", False)),
        data=data,
        upstream_path="/infill",
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:  # type: ignore[type-arg]
    """List all available models in OpenAI-compatible format."""
    coordinator = _get_coordinator(request)
    custom_models = await coordinator.list_models()

    # Convert custom format to OpenAI compatible format
    openai_models = []
    for model in custom_models:
        openai_models.append(
            ModelInfo(
                id=model["name"],
                family=model.get("family"),
                parameters=model.get("parameters"),
                min_memory_gb=model.get("min_memory_gb"),
                loaded_on=model.get("loaded_on", []),
                supports_quantization=model.get("supports_quantization", []),
            )
        )

    return ModelsResponse(data=openai_models)


@router.post("/models/load", response_model=LoadModelResponse)
async def load_model(body: LoadModelRequest, request: Request) -> LoadModelResponse:  # type: ignore[type-arg]
    """Load a model onto a worker."""
    coordinator = _get_coordinator(request)

    from coordinator.models import Quantization

    try:
        quantization = Quantization(body.quantization)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid quantization '{body.quantization}'. "
            f"Valid values: {[q.value for q in Quantization]} (only 'none' is loadable today)",
        ) from exc

    try:
        if body.worker_id is not None:
            worker_info = coordinator.workers.get(body.worker_id)
            if worker_info is None:
                raise HTTPException(status_code=404, detail=f"Worker {body.worker_id} not found")
        else:
            first_worker_id = next(iter(coordinator.workers), None)
            if first_worker_id is None:
                raise HTTPException(status_code=503, detail="No workers available")
            worker_info = coordinator.workers[first_worker_id]

        success = await coordinator._load_model_on_worker(
            worker_info,
            body.model_name,
            quantization=quantization,
        )

        if success:
            return LoadModelResponse(
                status="loaded",
                model_name=body.model_name,
                worker_id=worker_info.id,
            )
        return LoadModelResponse(
            status="failed",
            model_name=body.model_name,
            message="Model loading failed on the worker",
        )
    except HTTPException:
        raise
    except ValueError as exc:
        # H4: model_name absent from the registry with pull-through disabled.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        # M12: see create_completion's identical comment.
        logger.exception("Model load failed")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.delete("/models/{model_name}")
async def unload_model(
    model_name: str, request: Request, worker_id: Optional[str] = None  # type: ignore[type-arg]
) -> Dict[str, Any]:
    """Unload a model and free its GPU memory (all workers, or one via ?worker_id=)."""
    coordinator = _get_coordinator(request)
    try:
        unloaded_from = await coordinator.unload_model(model_name, worker_id=worker_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        # M12: see create_completion's identical comment.
        logger.exception("Model unload failed")
        raise HTTPException(status_code=500, detail="Internal server error") from None
    return {"status": "unloaded", "model_name": model_name, "workers": unloaded_from}


@router.get("/workers", response_model=List[WorkerInfoResponse])
async def list_workers(request: Request) -> List[Dict[str, Any]]:  # type: ignore[type-arg]
    """List all connected workers."""
    coordinator = _get_coordinator(request)
    result: List[Dict[str, Any]] = await coordinator.list_workers()
    return result


@router.post("/workers/manual")
async def add_manual_worker(
    addresses: List[str], request: Request  # type: ignore[type-arg]
) -> Dict[str, List[Dict[str, str]]]:
    """Manually add a worker by its host:port address.

    C3: a worker registered here self-reports ``loaded_models`` and can be
    selected by ``find_worker_for_model`` for real routed traffic (prompt
    exfiltration + poisoned completions), and the address is passed straight
    to ``grpc.aio.insecure_channel`` / the llamaserver HTTP proxy (SSRF). So
    this route:
      1. is disabled unless ``COORDINATOR_ALLOW_MANUAL_WORKER_REGISTRATION``
         is set — most deployments never need runtime registration
         (``COORDINATOR_STATIC_WORKERS`` covers the documented setups);
      2. independently requires a valid ``COORDINATOR_API_KEYS`` credential
         on THIS request regardless of whether the global
         ``APIKeyAuthMiddleware`` happens to be a no-op elsewhere (e.g. a
         dev deployment intentionally left open for everything else);
      3. shape-validates every address (host:port, optional
         ``COORDINATOR_MANUAL_WORKER_ALLOWED_HOSTS`` CIDR/host allowlist) and
         caps the list length, all BEFORE any address is dialed.
    """
    coordinator = _get_coordinator(request)
    settings = coordinator.settings

    if not settings.allow_manual_worker_registration:
        raise HTTPException(
            status_code=403,
            detail=(
                "Manual worker registration is disabled. Set "
                "COORDINATOR_ALLOW_MANUAL_WORKER_REGISTRATION=true to enable it — see "
                ".env.example."
            ),
        )

    valid_keys = auth.load_api_keys()
    if not valid_keys:
        raise HTTPException(
            status_code=403,
            detail=(
                "Manual worker registration requires COORDINATOR_API_KEYS to be "
                "configured, independent of whether auth is enabled for the rest of "
                "the API."
            ),
        )
    candidate = auth._extract_candidate_key(request.headers)
    if candidate is None or not auth._matches_any(candidate, valid_keys):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {"message": "invalid or missing API key", "type": "authentication_error"}
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if len(addresses) > _MAX_MANUAL_WORKERS_PER_REQUEST:
        raise HTTPException(
            status_code=422,
            detail=(
                f"at most {_MAX_MANUAL_WORKERS_PER_REQUEST} addresses per request "
                f"(got {len(addresses)})"
            ),
        )

    for address in addresses:
        try:
            _validate_manual_worker_address(address, settings.manual_worker_allowed_hosts)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    results: List[Dict[str, str]] = []
    for address in addresses:
        worker = await coordinator._connect_worker(address)
        if worker:
            results.append({"address": address, "status": "connected", "id": worker.id})
        else:
            results.append({"address": address, "status": "failed"})
    return {"results": results}
