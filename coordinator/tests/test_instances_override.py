"""Runtime `instances` override for engine="llamaserver" models: the
POST /v1/models/load field flows through ClusterCoordinator._load_model_on_worker
into ModelConfig.grpc_metadata()'s `llamaserver.parallel` gRPC metadata key,
overriding (not replacing) the registry value. See coordinator/models.py
ModelConfig.grpc_metadata/_llamaserver_metadata and worker/src/llamaserver_process.rs
llamaserver_spec_from_metadata for the worker side of this contract.
"""

from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest
from fastapi.testclient import TestClient

from coordinator.coordinator import ClusterCoordinator, WorkerInfo
from coordinator.main import app
from coordinator.models import ModelRegistry
from coordinator.tests.conftest import make_settings

client = TestClient(app)

_LLAMASERVER_REGISTRY = {
    "models": {
        "instances-override-test": {
            "family": "qwen",
            "parameters": "7B",
            "min_memory_gb": 6,
            "engine": "llamaserver",
            "gguf": {
                "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "file": "qwen2.5-7b-instruct-q4_k_m.gguf",
            },
            "llamaserver": {"port": 8199, "instances": 4},
        }
    }
}


def _bare_coordinator() -> ClusterCoordinator:
    coordinator = ClusterCoordinator.__new__(ClusterCoordinator)
    coordinator.settings = make_settings()
    return coordinator


class _FakeLoadStub:
    """Captures the LoadModelRequest the coordinator sends."""

    def __init__(self) -> None:
        self.request: Optional[Any] = None

    async def LoadModel(self, request: Any, timeout: Optional[int] = None) -> Any:
        self.request = request
        return SimpleNamespace(success=True, memory_used=0, message="ok")


def _fake_worker(stub: _FakeLoadStub) -> WorkerInfo:
    return cast(
        WorkerInfo,
        SimpleNamespace(
            id="worker-test", gpus=[SimpleNamespace(id=0)], stub=stub, loaded_models={}
        ),
    )


@pytest.mark.asyncio
async def test_instances_override_replaces_registry_value_in_metadata() -> None:
    ModelRegistry.load_from_dict(_LLAMASERVER_REGISTRY)
    stub = _FakeLoadStub()
    coordinator = _bare_coordinator()
    ok = await coordinator._load_model_on_worker(
        _fake_worker(stub), "instances-override-test", instances=9
    )
    assert ok is True
    assert stub.request is not None
    assert stub.request.config.metadata["llamaserver.parallel"] == "9"


@pytest.mark.asyncio
async def test_no_instances_override_falls_back_to_registry_value() -> None:
    ModelRegistry.load_from_dict(_LLAMASERVER_REGISTRY)
    stub = _FakeLoadStub()
    coordinator = _bare_coordinator()
    ok = await coordinator._load_model_on_worker(_fake_worker(stub), "instances-override-test")
    assert ok is True
    assert stub.request is not None
    assert stub.request.config.metadata["llamaserver.parallel"] == "4"


# ---------------------------------------------------------------------------
# Route-level: POST /v1/models/load
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_app_coordinator_state() -> Any:
    yield
    if hasattr(app.state, "coordinator"):
        del app.state.coordinator


def test_load_model_route_rejects_instances_for_non_llamaserver_model() -> None:
    settings = make_settings()
    fake_worker = SimpleNamespace(id="w1")
    fake_coordinator = SimpleNamespace(settings=settings, workers={"w1": fake_worker})
    app.state.coordinator = fake_coordinator

    # "deepseek-7b" is a built-in default-registry model with engine="burn".
    response = client.post(
        "/v1/models/load",
        json={"model_name": "deepseek-7b", "instances": 3},
    )
    assert response.status_code == 422
    assert "llamaserver" in response.json()["detail"]


def test_load_model_route_rejects_instances_below_one() -> None:
    response = client.post(
        "/v1/models/load",
        json={"model_name": "deepseek-7b", "instances": 0},
    )
    assert response.status_code == 422


def test_load_model_route_accepts_instances_for_llamaserver_model() -> None:
    ModelRegistry.load_from_dict(_LLAMASERVER_REGISTRY)
    settings = make_settings()
    fake_worker = SimpleNamespace(id="w1")
    captured = {}

    async def fake_load(worker: Any, model_name: str, **kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    fake_coordinator = SimpleNamespace(
        settings=settings,
        workers={"w1": fake_worker},
        _load_model_on_worker=fake_load,
    )
    app.state.coordinator = fake_coordinator

    response = client.post(
        "/v1/models/load",
        json={"model_name": "instances-override-test", "instances": 12},
    )
    assert response.status_code == 200
    assert captured["instances"] == 12
