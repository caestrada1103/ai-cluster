"""Tests for coordinator.auth — opt-in API-key auth middleware.

COORDINATOR_API_KEYS is read live from the environment on every request
(coordinator.auth.load_api_keys), so tests just monkeypatch.setenv/delenv —
there is no cache to reset between tests.

The TestClient below is never entered as a context manager, so the app's
lifespan (and therefore ClusterCoordinator.start()) never runs. That's
deliberate: it keeps these tests independent of workers/gRPC, and every
assertion here only cares about the auth middleware's own behavior — either
a stable coordinator-free route ("/") for positive-auth checks, or asserting
a v1 route is reachable (not 401'd) rather than asserting its exact
(coordinator-dependent) status code.
"""

import inspect

import pytest
from fastapi.testclient import TestClient

from coordinator import auth
from coordinator.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_api_keys_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no stray COORDINATOR_API_KEYS leaks in from the real environment/.env."""
    monkeypatch.delenv("COORDINATOR_API_KEYS", raising=False)


# ---------------------------------------------------------------------------
# Env var unset/empty -> open access (no behavior change)
# ---------------------------------------------------------------------------


def test_open_access_when_env_unset() -> None:
    response = client.get("/")
    assert response.status_code == 200


def test_open_access_when_env_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "")
    response = client.get("/")
    assert response.status_code == 200


def test_open_access_v1_route_when_env_unset() -> None:
    response = client.get("/v1/models")
    # No coordinator wired up in this TestClient (lifespan not entered), so
    # the route itself 503s -- the point here is auth doesn't 401 it.
    assert response.status_code != 401


# ---------------------------------------------------------------------------
# Env var set -> gated access
# ---------------------------------------------------------------------------


def test_missing_key_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.get("/")
    assert response.status_code == 401
    assert response.json() == {
        "error": {"message": "invalid or missing API key", "type": "authentication_error"}
    }
    assert response.headers["www-authenticate"] == "Bearer"


def test_wrong_bearer_key_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.get("/", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401


def test_wrong_x_api_key_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.get("/", headers={"x-api-key": "wrong-key"})
    assert response.status_code == 401


def test_malformed_authorization_header_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    # Right scheme keyword, wrong casing/format is fine (case-insensitive
    # scheme is handled) but a non-Bearer scheme must still be rejected.
    response = client.get("/", headers={"Authorization": "Basic secret-key"})
    assert response.status_code == 401


def test_valid_bearer_key_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.get("/", headers={"Authorization": "Bearer secret-key"})
    assert response.status_code == 200


def test_valid_x_api_key_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.get("/", headers={"x-api-key": "secret-key"})
    assert response.status_code == 200


def test_any_key_in_comma_separated_list_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "key-one, key-two ,key-three")
    response = client.get("/", headers={"x-api-key": "key-two"})
    assert response.status_code == 200


def test_v1_route_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.get("/v1/models")
    assert response.status_code == 401


def test_v1_route_accepts_valid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.get("/v1/models", headers={"Authorization": "Bearer secret-key"})
    # Reaches the route handler past auth (no coordinator wired up in this
    # TestClient) -- the point here is it's NOT a 401.
    assert response.status_code != 401


# ---------------------------------------------------------------------------
# Exempt paths
# ---------------------------------------------------------------------------


def test_health_exempt_when_keys_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.get("/health")
    assert response.status_code == 200


def test_metrics_exempt_when_keys_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.get("/metrics")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# load_api_keys parsing
# ---------------------------------------------------------------------------


def test_load_api_keys_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COORDINATOR_API_KEYS", raising=False)
    assert auth.load_api_keys() == frozenset()


def test_load_api_keys_parses_trims_and_drops_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", " key-one ,, key-two ,")
    assert auth.load_api_keys() == {"key-one", "key-two"}


# ---------------------------------------------------------------------------
# Constant-time comparison
# ---------------------------------------------------------------------------


def test_uses_constant_time_comparison() -> None:
    """Guard against a naive `==`/`in` regression: must use secrets.compare_digest."""
    source = inspect.getsource(auth)
    assert "compare_digest" in source


# ---------------------------------------------------------------------------
# Non-ASCII candidate key must 401, never an unhandled 500
# ---------------------------------------------------------------------------


def test_non_ascii_x_api_key_header_returns_401_not_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-ASCII candidate used to make compare_digest raise TypeError,
    surfacing as a 500 instead of a 401. Sends the header as latin-1-encoded
    bytes to bypass httpx's own str-header ASCII validation.
    """
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.get(
        "/",
        headers=[(b"x-api-key", "café-not-the-key".encode("latin-1"))],
    )
    assert response.status_code == 401


def test_matches_any_handles_non_ascii_without_raising() -> None:
    """Direct unit check of the comparison helper for the same case."""
    assert auth._matches_any("café", frozenset({"secret-key"})) is False
    # A configured key itself containing non-ASCII must still be matchable.
    assert auth._matches_any("café", frozenset({"café"})) is True


# ---------------------------------------------------------------------------
# CORS preflight (OPTIONS) must never be gated behind the API key
# ---------------------------------------------------------------------------


def test_options_preflight_bypasses_auth_when_keys_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CORS preflight carries no Authorization/x-api-key by design — gating
    it behind auth 401s it before CORSMiddleware can attach headers, which
    the browser then misreports as a CORS failure on the real request."""
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    response = client.options(
        "/v1/models",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code != 401
