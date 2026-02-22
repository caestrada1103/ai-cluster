"""API routes for the AI Cluster coordinator.

Provides the FastAPI router mounted at ``/v1`` in main.py.
Endpoints:
    POST /completions  - Run inference
    GET  /models       - List available models
    POST /models/load  - Load a model onto a worker
    GET  /workers      - List connected workers
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import time

logger = logging.getLogger(__name__)

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


class CompletionRequest(BaseModel):
    """Body for the POST /completions endpoint."""

    model: str = Field(..., description="Model name, e.g. 'deepseek-7b'")
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(512, ge=1, le=32768)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=0)
    stream: bool = False
    worker_id: Optional[str] = Field(None, description="Optional worker ID to force routing to a specific GPU")


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
    quantization: str = "fp16"


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

def _get_coordinator(request: Request):
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


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(body: CompletionRequest, request: Request):
    """Run inference on the cluster."""
    coordinator = _get_coordinator(request)

    model_name, target_worker = _parse_model_and_worker(body.model)
    worker_id = body.worker_id or target_worker

    try:
        result = await coordinator.infer(
            model_name=model_name,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            stream=body.stream,
            worker_id=worker_id,
        )
        return CompletionResponse(**result)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))


def _build_flat_response(result: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Build a standard OpenAI-compatible chat completion response."""
    return {
        "id": result["request_id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result["text"]
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": result["tokens_generated"],
            "total_tokens": result["tokens_generated"]
        }
    }

async def _stream_chat_completion(ctx: Any, model: str):
    """Generator for OpenAI-compatible Server-Sent Events (SSE)."""
    try:
        while True:
            # Wait for next token from the queue
            response = await ctx.token_queue.get()
            
            # Build SSE chunk
            chunk = {
                "id": ctx.id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": response.text if not response.finished else ""
                    },
                    "finish_reason": "stop" if response.finished else None
                }]
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

@router.post("/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completions endpoint used by Open-WebUI."""
    logger.info(f"Received chat completion request for model: {body.model}")
    coordinator = _get_coordinator(request)

    model_name, worker_id = _parse_model_and_worker(body.model)

    # Convert chat history to a raw prompt
    # A simple chat template, can be expanded later for specific models (llama3, chatml, etc)
    prompt = ""
    for msg in body.messages:
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
        result = await coordinator.infer(
            model_name=model_name,
            prompt=prompt,
            max_tokens=body.max_tokens or 512,
            temperature=body.temperature or 0.7,
            top_p=body.top_p or 0.95,
            top_k=body.top_k or 40,
            stream=body.stream or False,
            worker_id=worker_id,
        )
        
        if body.stream:
            request_context = coordinator.active_requests.get(result["request_id"])
            if not request_context:
                # Fallback to flat result if somehow lost from context
                return _build_flat_response(result, body.model)
            
            return StreamingResponse(
                _stream_chat_completion(request_context, body.model),
                media_type="text/event-stream"
            )

        return _build_flat_response(result, body.model)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request):
    """List all available models in OpenAI-compatible format."""
    coordinator = _get_coordinator(request)
    custom_models = await coordinator.list_models()
    
    # Convert custom format to OpenAI compatible format
    openai_models = []
    for model in custom_models:
        openai_models.append(ModelInfo(
            id=model["name"],
            family=model.get("family"),
            parameters=model.get("parameters"),
            min_memory_gb=model.get("min_memory_gb"),
            loaded_on=model.get("loaded_on", []),
            supports_quantization=model.get("supports_quantization", [])
        ))
        
    return ModelsResponse(data=openai_models)


@router.post("/models/load", response_model=LoadModelResponse)
async def load_model(body: LoadModelRequest, request: Request):
    """Load a model onto a worker."""
    coordinator = _get_coordinator(request)

    # Pick a specific worker or let the coordinator decide
    workers = await coordinator.list_workers()
    if not workers:
        raise HTTPException(status_code=503, detail="No workers available")

    target_worker = None
    if body.worker_id:
        for w in workers:
            if w["id"] == body.worker_id:
                target_worker = w
                break
        if target_worker is None:
            raise HTTPException(status_code=404, detail=f"Worker {body.worker_id} not found")

    try:
        # Delegate to coordinator's internal loading mechanism
        worker_info = coordinator.workers.get(
            body.worker_id or next(iter(coordinator.workers))
        )
        if worker_info is None:
            raise HTTPException(status_code=503, detail="No workers available")

        from coordinator.models import Quantization

        success = await coordinator._load_model_on_worker(
            worker_info,
            body.model_name,
            quantization=Quantization(body.quantization),
        )

        if success:
            return LoadModelResponse(
                status="loaded",
                model_name=body.model_name,
                worker_id=worker_info.id,
            )
        else:
            return LoadModelResponse(
                status="failed",
                model_name=body.model_name,
                message="Model loading failed on the worker",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Model load failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/workers", response_model=List[WorkerInfoResponse])
async def list_workers(request: Request):
    """List all connected workers."""
    coordinator = _get_coordinator(request)
    return await coordinator.list_workers()


@router.post("/workers/manual")
async def add_manual_worker(addresses: List[str], request: Request):
    """Manually add a worker by its host:port address."""
    coordinator = _get_coordinator(request)
    results = []
    for address in addresses:
        worker = await coordinator._connect_worker(address)
        if worker:
            results.append({"address": address, "status": "connected", "id": worker.id})
        else:
            results.append({"address": address, "status": "failed"})
    return {"results": results}
