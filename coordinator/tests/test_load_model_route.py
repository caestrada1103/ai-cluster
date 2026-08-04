"""Route-level test for H4: POST /v1/models/load must 404 an unregistered
model_name by default instead of letting it fall through to
ClusterCoordinator._load_model_on_worker's HuggingFace-pull path.

Same fake-coordinator-injected-into-app.state pattern as
test_workers_manual.py (see that file's module docstring for why entering
the real lifespan isn't needed/wanted here).
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from coordinator.main import app
from coordinator.tests.conftest import make_settings

client = TestClient(app)


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
    """M12: a genuine internal error must never echo str(exc) to the client
    — only "Internal server error", with the real detail logged server-side."""
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
