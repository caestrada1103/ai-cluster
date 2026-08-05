"""Route-level test: POST /v1/models/load must 404 an unregistered
model_name by default instead of falling through to the HuggingFace-pull
path. Same fake-coordinator pattern as test_workers_manual.py.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from coordinator.main import app
from coordinator.models import ModelRegistry
from coordinator.tests.conftest import make_settings

client = TestClient(app)

_DISTRIBUTED_ROUTE_REGISTRY = {
    "models": {
        "distributed-route-test": {
            "family": "qwen",
            "parameters": "32B",
            "min_memory_gb": 20,
            "engine": "llamacpp",
            "gguf": {
                "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
                "file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
            },
            "distributed": {
                "enabled": True,
                "lead": "lead-1",
                "peers": ["peer-1"],
            },
        }
    }
}


@pytest.fixture(autouse=True)
def _reset_app_coordinator_state() -> Any:
    yield
    if hasattr(app.state, "coordinator"):
        del app.state.coordinator


def test_load_model_rejects_unregistered_model_by_default() -> None:
    settings = make_settings()
    fake_worker = SimpleNamespace(id="w1")
    fake_coordinator = SimpleNamespace(
        settings=settings,
        workers={"w1": fake_worker},
        _load_model_on_worker=AsyncMock(
            side_effect=ValueError(
                "Model 'unregistered-xyz' is not in the registry (config/models.toml) "
                "and unregistered-model pull-through is disabled."
            )
        ),
    )
    app.state.coordinator = fake_coordinator

    response = client.post(
        "/v1/models/load",
        json={"model_name": "unregistered-xyz"},
    )
    assert response.status_code == 404
    assert "not in the registry" in response.json()["detail"]
    fake_coordinator._load_model_on_worker.assert_awaited_once()


def test_load_model_unexpected_error_does_not_leak_exception_text() -> None:
    """A genuine internal error must never echo str(exc) to the client —
    only "Internal server error", with the real detail logged server-side."""
    settings = make_settings()
    fake_worker = SimpleNamespace(id="w1")
    secret_detail = "connection refused to internal-db-host:5432 (leaky!)"
    fake_coordinator = SimpleNamespace(
        settings=settings,
        workers={"w1": fake_worker},
        _load_model_on_worker=AsyncMock(side_effect=RuntimeError(secret_detail)),
    )
    app.state.coordinator = fake_coordinator

    response = client.post("/v1/models/load", json={"model_name": "deepseek-7b"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert secret_detail not in response.text


def test_load_model_succeeds_for_registered_model() -> None:
    settings = make_settings()
    fake_worker = SimpleNamespace(id="w1")
    fake_coordinator = SimpleNamespace(
        settings=settings,
        workers={"w1": fake_worker},
        _load_model_on_worker=AsyncMock(return_value=True),
    )
    app.state.coordinator = fake_coordinator

    response = client.post(
        "/v1/models/load",
        json={"model_name": "deepseek-7b"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "loaded"


# ---------------------------------------------------------------------------
# Distributed models: dispatch to _load_distributed_model, worker_id must be
# the configured lead (or omitted)
# ---------------------------------------------------------------------------


def test_load_model_distributed_dispatches_to_load_distributed_model() -> None:
    ModelRegistry.load_from_dict(_DISTRIBUTED_ROUTE_REGISTRY)
    settings = make_settings()
    fake_coordinator = SimpleNamespace(
        settings=settings,
        workers={},
        _load_distributed_model=AsyncMock(return_value=True),
    )
    app.state.coordinator = fake_coordinator

    response = client.post("/v1/models/load", json={"model_name": "distributed-route-test"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "loaded"
    assert body["worker_id"] == "lead-1"
    fake_coordinator._load_distributed_model.assert_awaited_once()


def test_load_model_distributed_rejects_worker_id_other_than_lead() -> None:
    ModelRegistry.load_from_dict(_DISTRIBUTED_ROUTE_REGISTRY)
    settings = make_settings()
    fake_coordinator = SimpleNamespace(
        settings=settings,
        workers={},
        _load_distributed_model=AsyncMock(return_value=True),
    )
    app.state.coordinator = fake_coordinator

    response = client.post(
        "/v1/models/load",
        json={"model_name": "distributed-route-test", "worker_id": "peer-1"},
    )

    assert response.status_code == 422
    assert "distributed" in response.json()["detail"]
    fake_coordinator._load_distributed_model.assert_not_awaited()


def test_load_model_distributed_allows_worker_id_matching_lead() -> None:
    ModelRegistry.load_from_dict(_DISTRIBUTED_ROUTE_REGISTRY)
    settings = make_settings()
    fake_coordinator = SimpleNamespace(
        settings=settings,
        workers={},
        _load_distributed_model=AsyncMock(return_value=True),
    )
    app.state.coordinator = fake_coordinator

    response = client.post(
        "/v1/models/load",
        json={"model_name": "distributed-route-test", "worker_id": "lead-1"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "loaded"


def test_load_model_distributed_reports_failure() -> None:
    ModelRegistry.load_from_dict(_DISTRIBUTED_ROUTE_REGISTRY)
    settings = make_settings()
    fake_coordinator = SimpleNamespace(
        settings=settings,
        workers={},
        _load_distributed_model=AsyncMock(return_value=False),
    )
    app.state.coordinator = fake_coordinator

    response = client.post("/v1/models/load", json={"model_name": "distributed-route-test"})

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
