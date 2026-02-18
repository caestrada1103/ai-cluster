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
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    """Body for the POST /completions endpoint."""

    model: str = Field(..., description="Model name, e.g. 'deepseek-7b'")
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(512, ge=1, le=32768)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=0)
    stream: bool = False


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
    """Schema for a single model entry."""

    name: str
    family: str
    parameters: str
    min_memory_gb: float
    loaded_on: List[Dict[str, Any]] = []
    supports_quantization: List[str] = []


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

@router.post("/completions", response_model=CompletionResponse)
async def create_completion(body: CompletionRequest, request: Request):
    """Run inference on the cluster."""
    coordinator = _get_coordinator(request)

    try:
        result = await coordinator.infer(
            model_name=body.model,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            stream=body.stream,
        )
        return CompletionResponse(**result)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/models", response_model=List[ModelInfo])
async def list_models(request: Request):
    """List all available models and their loading status."""
    coordinator = _get_coordinator(request)
    return await coordinator.list_models()


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
