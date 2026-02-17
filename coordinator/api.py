from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx
import json

router = APIRouter(prefix="/api")

# ------------------------------------------------------------------
# Minimal client request model
# ------------------------------------------------------------------
class InferenceRequest(BaseModel):
    worker_id: str
    model_name: str
    input: dict  # Generic JSON payload

# ------------------------------------------------------------------
# Example route: query worker for a simple status
# ------------------------------------------------------------------
@router.post("/infer")
async def infer(req: InferenceRequest):
    # Dispatch to worker via gRPC – placeholder logic
    try:
        import grpc
        from worker_pb2_grpc import WorkerServiceStub
        from worker_pb2 import InferenceRequest as
GrpcInferenceRequest
    except Exception:
        raise HTTPException(status_code=500, detail="GRPC stub not
generated")

    async with grpc.aio.insecure_channel(f"worker:{req.worker_id}")
as channel:
        stub = WorkerServiceStub(channel)
        grpc_req = GrpcInferenceRequest(
            model_name=req.model_name,
            input_json=json.dumps(req.input),
        )
        grpc_resp = await stub.Infer(grpc_req)
        return {"output": grpc_resp.output_json}