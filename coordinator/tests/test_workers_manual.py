"""Tests for POST /v1/workers/manual (C3) — disabled-by-default, independently
authenticated, address-validated worker registration.

Route-level tests inject a fake coordinator directly into `app.state`
(mirroring how `main.py`'s real `lifespan` does it) rather than entering the
app's lifespan (which would need a live/fake gRPC worker stack) — the same
pattern `test_auth.py`/`test_main.py` document and rely on for their own
`TestClient`.
"""

from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from coordinator import api
from coordinator.main import app
from coordinator.tests.conftest import make_settings

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COORDINATOR_API_KEYS", raising=False)


@pytest.fixture(autouse=True)
def _reset_app_coordinator_state() -> Any:
    """`app.state` is a shared, module-level object — restore it to the
    "no coordinator wired up" default other test modules
    (test_auth.py/test_main.py) assume, so this file's fake-coordinator
    injection can never leak into a test run after it."""
    yield
    if hasattr(app.state, "coordinator"):
        del app.state.coordinator


def _install_fake_coordinator(settings: Any, connect_worker: Optional[AsyncMock] = None) -> None:
    fake = SimpleNamespace(
        settings=settings,
        _connect_worker=connect_worker or AsyncMock(return_value=None),
    )
    app.state.coordinator = fake


# ---------------------------------------------------------------------------
# Pure address-validation helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address,host,port",
    [
        ("127.0.0.1:50051", "127.0.0.1", 50051),
        ("worker-1.internal:50051", "worker-1.internal", 50051),
        ("[::1]:50051", "::1", 50051),
    ],
)
def test_split_host_port_accepts_well_formed(address: str, host: str, port: int) -> None:
    assert api._split_host_port(address) == (host, port)


@pytest.mark.parametrize(
    "address",
    [
        "no-port-here",
        "host:not-a-port",
        "host:0",
        "host:70000",
        ":50051",
        "a:b:c:50051",
    ],
)
def test_split_host_port_rejects_malformed(address: str) -> None:
    with pytest.raises(ValueError):
        api._split_host_port(address)


def test_is_well_formed_host_accepts_ip_and_hostname() -> None:
    assert api._is_well_formed_host("127.0.0.1")
    assert api._is_well_formed_host("::1")
    assert api._is_well_formed_host("worker-1.internal")


def test_is_well_formed_host_rejects_garbage() -> None:
    assert not api._is_well_formed_host("http://evil")
    assert not api._is_well_formed_host("has spaces")
    assert not api._is_well_formed_host("-leading-dash")


def test_host_allowed_empty_allowlist_accepts_anything() -> None:
    assert api._host_allowed("10.9.8.7", [])


def test_host_allowed_exact_match() -> None:
    assert api._host_allowed("worker-1.internal", ["worker-1.internal"])
    assert not api._host_allowed("worker-2.internal", ["worker-1.internal"])


def test_host_allowed_cidr_match() -> None:
    assert api._host_allowed("192.168.1.42", ["192.168.1.0/24"])
    assert not api._host_allowed("10.0.0.5", ["192.168.1.0/24"])


def test_validate_manual_worker_address_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        api._validate_manual_worker_address("not-an-address", [])


def test_validate_manual_worker_address_rejects_disallowed_host() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        api._validate_manual_worker_address("10.0.0.5:50051", ["192.168.1.0/24"])


def test_validate_manual_worker_address_accepts_allowed_host() -> None:
    api._validate_manual_worker_address("192.168.1.5:50051", ["192.168.1.0/24"])  # no raise


# ---------------------------------------------------------------------------
# Route: disabled by default (C3)
# ---------------------------------------------------------------------------


def test_disabled_by_default_returns_403() -> None:
    settings = make_settings()
    assert settings.allow_manual_worker_registration is False
    _install_fake_coordinator(settings)
    response = client.post("/v1/workers/manual", json=["127.0.0.1:50051"])
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Route: enabled, but requires COORDINATOR_API_KEYS regardless of global auth
# ---------------------------------------------------------------------------


def test_enabled_without_any_keys_configured_returns_403(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COORDINATOR_API_KEYS", raising=False)
    settings = make_settings(allow_manual_worker_registration=True)
    _install_fake_coordinator(settings)
    response = client.post("/v1/workers/manual", json=["127.0.0.1:50051"])
    assert response.status_code == 403


def test_enabled_with_keys_but_no_credential_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    settings = make_settings(allow_manual_worker_registration=True)
    _install_fake_coordinator(settings)
    response = client.post("/v1/workers/manual", json=["127.0.0.1:50051"])
    assert response.status_code == 401


def test_enabled_with_wrong_credential_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    settings = make_settings(allow_manual_worker_registration=True)
    _install_fake_coordinator(settings)
    response = client.post(
        "/v1/workers/manual",
        json=["127.0.0.1:50051"],
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401


def test_enabled_with_valid_credential_connects_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    settings = make_settings(allow_manual_worker_registration=True)
    fake_worker = SimpleNamespace(id="w1")
    connect = AsyncMock(return_value=fake_worker)
    _install_fake_coordinator(settings, connect_worker=connect)
    response = client.post(
        "/v1/workers/manual",
        json=["127.0.0.1:50051"],
        headers={"Authorization": "Bearer secret-key"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "results": [{"address": "127.0.0.1:50051", "status": "connected", "id": "w1"}]
    }
    connect.assert_awaited_once_with("127.0.0.1:50051")


# ---------------------------------------------------------------------------
# Route: validation applied even when authenticated
# ---------------------------------------------------------------------------


def test_authenticated_but_malformed_address_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    settings = make_settings(allow_manual_worker_registration=True)
    connect = AsyncMock()
    _install_fake_coordinator(settings, connect_worker=connect)
    response = client.post(
        "/v1/workers/manual",
        json=["not-an-address"],
        headers={"Authorization": "Bearer secret-key"},
    )
    assert response.status_code == 422
    connect.assert_not_awaited()


def test_authenticated_but_too_many_addresses_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    settings = make_settings(allow_manual_worker_registration=True)
    connect = AsyncMock()
    _install_fake_coordinator(settings, connect_worker=connect)
    addresses: List[str] = [f"10.0.0.{i}:50051" for i in range(20)]
    response = client.post(
        "/v1/workers/manual",
        json=addresses,
        headers={"Authorization": "Bearer secret-key"},
    )
    assert response.status_code == 422
    connect.assert_not_awaited()


def test_authenticated_but_disallowed_host_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    settings = make_settings(
        allow_manual_worker_registration=True,
        manual_worker_allowed_hosts=["192.168.1.0/24"],
    )
    connect = AsyncMock()
    _install_fake_coordinator(settings, connect_worker=connect)
    response = client.post(
        "/v1/workers/manual",
        json=["10.0.0.5:50051"],
        headers={"Authorization": "Bearer secret-key"},
    )
    assert response.status_code == 422
    connect.assert_not_awaited()
