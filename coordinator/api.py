"""API routes for the AI Cluster coordinator.

Provides the FastAPI router mounted at ``/v1`` in main.py.
Endpoints:
    POST /completions  - Run inference
    GET  /models       - List available models
    POST /models/load  - Load a model onto a worker
    GET  /workers      - List connected workers
"""
import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, TypeVar, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from coordinator import proxy
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


async def _proxy_to_llamaserver(
    request: Request,  # type: ignore[type-arg]
    coordinator: Any,
    model_cfg: ModelConfig,
    raw_body: bytes,
    stream: bool,
) -> Response:
    """Forward a request to the worker-local llama-server serving ``model_cfg``.

    Plan 13 Task 2: pick a worker that already reports the model loaded (no
    auto-load in Phase 1), build ``http://<worker_host>:<port><same_path>`` and
    pass the raw body straight through. Returns 404 (pointing at the load
    endpoint) when the model is loaded on no worker.
    """
    worker = await coordinator.find_worker_for_model(model_cfg.name)
    if worker is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model_cfg.name}' (engine=llamaserver) is not loaded on any worker. "
                "Load it first with POST /v1/models/load."
            ),
        )
    if model_cfg.llamaserver_port is None:  # defensive: validated at registry load
        raise HTTPException(
            status_code=500,
            detail=f"Model '{model_cfg.name}' has no llamaserver_port configured",
        )
    host = _worker_host(worker.address)
    url = f"http://{host}:{model_cfg.llamaserver_port}{request.url.path}"
    headers = proxy.filter_request_headers(request.headers)
    result = await proxy.proxy_request(request.method, url, raw_body, headers, stream)
    return _proxy_response(result)


def _require_llamaserver_model(model_string: Any) -> ModelConfig:
    """Resolve an Anthropic body's ``model`` to a llamaserver registry entry.

    404 when the model is unknown; 501 when it exists but is not a
    llamaserver-engine model (the Anthropic ``/v1/messages`` surface has no
    in-process path — only llama-server serves it).
    """
    model_cfg = _lookup_engine_model(model_string)
    if model_cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_string}'")
    if model_cfg.engine != "llamaserver":
        raise HTTPException(
            status_code=501,
            detail=(
                f"Model '{model_cfg.name}' (engine={model_cfg.engine}) does not support the "
                "Anthropic /v1/messages API; only engine=llamaserver models do."
            ),
        )
    return model_cfg


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
            request, coordinator, model_cfg, raw_body, bool(data.get("stream", False))
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
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _build_flat_response(result: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Build a standard OpenAI-compatible chat completion response."""
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
            "prompt_tokens": 0,
            "completion_tokens": result["tokens_generated"],
            "total_tokens": result["tokens_generated"],
        },
    }


async def _stream_chat_completion(
    coordinator: Any, ctx: Any, model: str, timeout: float
) -> AsyncGenerator[str, None]:
    """Stream chunks live from the request's token queue as the worker produces them."""
    deadline = time.time() + timeout
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

            chunk = {
                "id": ctx.id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        # Emit the text even on the final chunk (proto allows text+finished)
                        "delta": {"content": response.text},
                        "finish_reason": "stop" if response.finished else None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            if response.finished:
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
    engine runs the in-process Zephyr-flattening path below.
    """
    coordinator = _get_coordinator(request)

    raw_body, data = await _read_json_body(request)
    model_cfg = _lookup_engine_model(data.get("model"))
    if model_cfg is not None and model_cfg.engine == "llamaserver":
        return await _proxy_to_llamaserver(
            request, coordinator, model_cfg, raw_body, bool(data.get("stream", False))
        )

    body = _parse_body(ChatCompletionRequest, data)
    logger.info(f"Received chat completion request for model: {body.model}")

    model_name, worker_id = _parse_model_and_worker(body.model)

    from coordinator.context_compression import maybe_compress_chat_messages

    messages = await maybe_compress_chat_messages(
        body.messages, coordinator=coordinator, override_enabled=body.compress_context
    )

    # Convert chat history to a raw prompt
    # A simple chat template, can be expanded later for specific models (llama3, chatml, etc)
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

    # Add generation token
    prompt += "<|assistant|>\n"

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
                _stream_chat_completion(coordinator, ctx, body.model, timeout),
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
        return _build_flat_response(result, body.model)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
    model_cfg = _require_llamaserver_model(data.get("model"))
    return await _proxy_to_llamaserver(
        request, coordinator, model_cfg, raw_body, bool(data.get("stream", False))
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
    model_cfg = _require_llamaserver_model(data.get("model"))
    return await _proxy_to_llamaserver(request, coordinator, model_cfg, raw_body, False)


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
    except Exception as exc:
        logger.exception("Model load failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
    except Exception as exc:
        logger.exception("Model unload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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
    """Manually add a worker by its host:port address."""
    coordinator = _get_coordinator(request)
    results: List[Dict[str, str]] = []
    for address in addresses:
        worker = await coordinator._connect_worker(address)
        if worker:
            results.append({"address": address, "status": "connected", "id": worker.id})
        else:
            results.append({"address": address, "status": "failed"})
    return {"results": results}
