"""Tests for coordinator.main — CORS defaults and the insecure-bind refusal.

`_refuse_insecure_bind`/`_is_loopback_host` are exercised as pure functions
rather than through the app's real `lifespan` (see test_auth.py for why
`TestClient` is never entered as a context manager here).
"""

import pytest
from fastapi.testclient import TestClient

from coordinator.main import _is_loopback_host, _refuse_insecure_bind, app
from coordinator.tests.conftest import make_settings

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_api_keys_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COORDINATOR_API_KEYS", raising=False)


# ---------------------------------------------------------------------------
# _is_loopback_host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.5", "::1", "localhost"],
)
def test_is_loopback_host_accepts_loopback_forms(host: str) -> None:
    assert _is_loopback_host(host) is True


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.168.1.10", "10.0.0.5", "coordinator.internal", ""],
)
def test_is_loopback_host_rejects_non_loopback_forms(host: str) -> None:
    assert _is_loopback_host(host) is False


# ---------------------------------------------------------------------------
# _refuse_insecure_bind
# ---------------------------------------------------------------------------


def test_refuses_non_loopback_host_with_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COORDINATOR_API_KEYS", raising=False)
    settings = make_settings(host="0.0.0.0")
    with pytest.raises(RuntimeError, match="Refusing to start"):
        _refuse_insecure_bind(settings)


def test_allows_non_loopback_host_when_keys_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    settings = make_settings(host="0.0.0.0")
    _refuse_insecure_bind(settings)  # must not raise


def test_allows_loopback_host_with_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COORDINATOR_API_KEYS", raising=False)
    settings = make_settings(host="127.0.0.1")
    _refuse_insecure_bind(settings)  # must not raise


# ---------------------------------------------------------------------------
# CORS defaults to loopback, not "*"
# ---------------------------------------------------------------------------


def test_cors_allows_loopback_origin_by_default() -> None:
    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:3000"},
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_arbitrary_origin_by_default() -> None:
    response = client.get(
        "/health",
        headers={"Origin": "https://evil.example.com"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_allows_loopback_origin_by_default() -> None:
    response = client.options(
        "/v1/models",
        headers={
            "Origin": "http://127.0.0.1:8080",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:8080"


# ---------------------------------------------------------------------------
# Request-body size cap is wired into the real app
# ---------------------------------------------------------------------------


def test_oversized_body_rejected_end_to_end() -> None:
    response = client.post(
        "/v1/completions",
        content=b"x" * 100,
        headers={"Content-Length": "999999999"},
    )
    assert response.status_code == 413
