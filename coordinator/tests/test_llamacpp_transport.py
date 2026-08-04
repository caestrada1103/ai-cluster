"""Tests for the llama.cpp engine gRPC transport (ModelConfig.metadata map).

The design carries engine routing in the EXISTING proto field
`ModelConfig.metadata` (map<string, string>), so no proto regeneration is
needed. These tests prove (a) the metadata survives a protobuf round-trip and
(b) _load_model_on_worker actually sends it.
"""

from types import SimpleNamespace
from typing import Any, List, Optional, cast

import pytest

import coordinator.proto.cluster_pb2 as pb
from coordinator.coordinator import ClusterCoordinator, WorkerInfo
from coordinator.models import ModelRegistry
from coordinator.tests.conftest import make_settings


def _bare_coordinator() -> ClusterCoordinator:
    """A ClusterCoordinator that skips the heavy __init__.

    `_load_model_on_worker` reads `settings.model_load_timeout` for the gRPC
    deadline, so that one attribute has to be supplied by hand.
    """
    coordinator = ClusterCoordinator.__new__(ClusterCoordinator)
    coordinator.settings = make_settings()
    return coordinator


_GGUF_REGISTRY = {
    "models": {
        "transport-test-gguf": {
            "family": "qwen",
            "parameters": "0.5B",
            "min_memory_gb": 1,
            "recommended_gpus": 1,
            "max_gpus": 1,
            "engine": "llamacpp",
            "gguf": {
                "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                "n_gpu_layers": -1,
                "n_ctx": 4096,
            },
        }
    }
}


class _FakeLoadStub:
    """Captures the LoadModelRequest the coordinator sends."""

    def __init__(self) -> None:
        self.request: Optional[Any] = None

    async def LoadModel(self, request: Any, timeout: Optional[int] = None) -> Any:
        self.request = request
        return SimpleNamespace(success=True, memory_used=0, message="ok")


def _fake_worker(stub: _FakeLoadStub) -> WorkerInfo:
    """Duck-typed WorkerInfo carrying only the attributes the loader touches."""
    return cast(
        WorkerInfo,
        SimpleNamespace(
            id="worker-test", gpus=[SimpleNamespace(id=0)], stub=stub, loaded_models={}
        ),
    )


def test_metadata_round_trips_through_proto() -> None:
    ModelRegistry.load_from_dict(_GGUF_REGISTRY)
    model = ModelRegistry.get_model("transport-test-gguf")
    assert model is not None
    config_pb = pb.ModelConfig(architecture="qwen", metadata=model.grpc_metadata())
    payload = pb.ModelConfig.FromString(config_pb.SerializeToString())
    assert payload.metadata["engine"] == "llamacpp"
    assert payload.metadata["gguf_repo_id"] == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    assert payload.metadata["gguf_file"] == "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    assert payload.metadata["n_gpu_layers"] == "-1"
    assert payload.metadata["n_ctx"] == "4096"


@pytest.mark.asyncio
async def test_load_model_on_worker_sends_llamacpp_metadata() -> None:
    ModelRegistry.load_from_dict(_GGUF_REGISTRY)
    stub = _FakeLoadStub()
    coordinator = _bare_coordinator()
    ok = await coordinator._load_model_on_worker(_fake_worker(stub), "transport-test-gguf")
    assert ok is True
    assert stub.request is not None
    assert stub.request.model_name == "transport-test-gguf"
    assert stub.request.config.metadata["engine"] == "llamacpp"
    assert stub.request.config.metadata["gguf_file"] == "qwen2.5-0.5b-instruct-q4_k_m.gguf"


@pytest.mark.asyncio
async def test_load_model_on_worker_sends_empty_metadata_for_burn() -> None:
    stub = _FakeLoadStub()
    coordinator = _bare_coordinator()
    ok = await coordinator._load_model_on_worker(_fake_worker(stub), "llama3-8b")
    assert ok is True
    assert stub.request is not None
    assert dict(stub.request.config.metadata) == {}


# ---------------------------------------------------------------------------
# Local multi-GPU split transport
# ---------------------------------------------------------------------------

_LOCAL_MULTI_GPU_REGISTRY = {
    "models": {
        "local-multi-gpu-gguf": {
            "family": "qwen",
            "parameters": "32B",
            "min_memory_gb": 20,
            "recommended_gpus": 1,
            "max_gpus": 4,
            "engine": "llamacpp",
            "gguf": {
                "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
                "file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
            },
            "local_gpu_ids": [2, 3],
            "local_tensor_split": [0.6, 0.4],
        }
    }
}


def _fake_worker_with_gpus(stub: _FakeLoadStub, gpu_ids: List[int]) -> WorkerInfo:
    """Duck-typed WorkerInfo with an arbitrary local GPU set."""
    return cast(
        WorkerInfo,
        SimpleNamespace(
            id="worker-multi-gpu",
            gpus=[SimpleNamespace(id=i) for i in gpu_ids],
            stub=stub,
            loaded_models={},
        ),
    )


@pytest.mark.asyncio
async def test_load_model_on_worker_sends_local_gpu_ids_and_tensor_split() -> None:
    ModelRegistry.load_from_dict(_LOCAL_MULTI_GPU_REGISTRY)
    stub = _FakeLoadStub()
    coordinator = _bare_coordinator()
    worker = _fake_worker_with_gpus(stub, [0, 1, 2, 3])
    ok = await coordinator._load_model_on_worker(worker, "local-multi-gpu-gguf")
    assert ok is True
    assert stub.request is not None
    # local_gpu_ids overrides the default worker.gpus[:recommended_gpus] slice
    # (recommended_gpus=1 would otherwise pick just gpu 0).
    assert list(stub.request.gpu_ids) == [2, 3]
    assert stub.request.config.metadata["tensor_split"] == "0.6,0.4"


@pytest.mark.asyncio
async def test_load_model_on_worker_default_gpu_selection_unchanged_without_local_gpu_ids() -> (
    None
):
    ModelRegistry.load_from_dict(_GGUF_REGISTRY)
    stub = _FakeLoadStub()
    coordinator = _bare_coordinator()
    ok = await coordinator._load_model_on_worker(_fake_worker(stub), "transport-test-gguf")
    assert ok is True
    assert stub.request is not None
    assert list(stub.request.gpu_ids) == [0]
    assert "tensor_split" not in stub.request.config.metadata
