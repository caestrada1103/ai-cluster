"""Tests for per-key scope/admin enforcement in coordinator/api.py.

Same fake-coordinator pattern as test_load_model_route.py/test_workers_manual.py:
a `SimpleNamespace` + `AsyncMock` coordinator injected directly into `app.state`,
driven through the real `TestClient` so real auth middleware + identity
resolution run.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from coordinator import identity, proxy
from coordinator.main import app
from coordinator.models import ModelRegistry
from coordinator.tests.conftest import make_settings

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COORDINATOR_API_KEYS", raising=False)
    monkeypatch.delenv("COORDINATOR_API_KEY_FILE", raising=False)
    identity._file_cache.clear()


@pytest.fixture(autouse=True)
def _reset_app_coordinator_state() -> Any:
    yield
    if hasattr(app.state, "coordinator"):
        del app.state.coordinator


@pytest.fixture(autouse=True)
def _reset_model_registry() -> Any:
    """Restore the (global, class-level) registry so a test-registered model
    never leaks into a later test."""
    snapshot = dict(ModelRegistry.MODELS)
    yield
    ModelRegistry.MODELS.clear()
    ModelRegistry.MODELS.update(snapshot)


def _write_key_file(tmp_path: Path, text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    toml_path = tmp_path / "keys.toml"
    toml_path.write_text(text)
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", str(toml_path))


def _install_coordinator(**overrides: Any) -> SimpleNamespace:
    fake_worker = SimpleNamespace(id="w1")
    defaults: Dict[str, Any] = dict(
        settings=make_settings(),
        workers={"w1": fake_worker},
        _load_model_on_worker=AsyncMock(return_value=True),
        unload_model=AsyncMock(return_value=["w1"]),
        _connect_worker=AsyncMock(return_value=SimpleNamespace(id="w1")),
        list_models=AsyncMock(
            return_value=[
                {"name": "model-a", "family": "qwen", "parameters": "7B", "min_memory_gb": 1.0},
                {"name": "model-b", "family": "qwen", "parameters": "7B", "min_memory_gb": 1.0},
            ]
        ),
    )
    defaults.update(overrides)
    fake = SimpleNamespace(**defaults)
    app.state.coordinator = fake
    return fake


# ---------------------------------------------------------------------------
# 1. Flat COORDINATOR_API_KEYS only -- no regression
# ---------------------------------------------------------------------------


def test_flat_keys_admin_route_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "secret-key")
    _install_coordinator()

    headers = {"Authorization": "Bearer secret-key"}
    load_resp = client.post("/v1/models/load", json={"model_name": "model-a"}, headers=headers)
    assert load_resp.status_code == 200

    unload_resp = client.delete("/v1/models/model-a", headers=headers)
    assert unload_resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. No keys at all (auth off) -- no regression
# ---------------------------------------------------------------------------


def test_auth_off_admin_route_still_works() -> None:
    _install_coordinator()

    load_resp = client.post("/v1/models/load", json={"model_name": "model-a"})
    assert load_resp.status_code == 200

    unload_resp = client.delete("/v1/models/model-a")
    assert unload_resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. Key file, role="user" -- admin routes denied
# ---------------------------------------------------------------------------


def test_user_role_denied_load_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_key_file(tmp_path, '[keys.ci]\nkey = "user-secret"\nrole = "user"\n', monkeypatch)
    fake = _install_coordinator()

    resp = client.post(
        "/v1/models/load",
        json={"model_name": "model-a"},
        headers={"Authorization": "Bearer user-secret"},
    )
    assert resp.status_code == 403
    fake._load_model_on_worker.assert_not_awaited()


def test_user_role_denied_unload_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_key_file(tmp_path, '[keys.ci]\nkey = "user-secret"\nrole = "user"\n', monkeypatch)
    fake = _install_coordinator()

    resp = client.delete("/v1/models/model-a", headers={"Authorization": "Bearer user-secret"})
    assert resp.status_code == 403
    fake.unload_model.assert_not_awaited()


def test_user_role_denied_manual_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_key_file(tmp_path, '[keys.ci]\nkey = "user-secret"\nrole = "user"\n', monkeypatch)
    fake = _install_coordinator(settings=make_settings(allow_manual_worker_registration=True))

    resp = client.post(
        "/v1/workers/manual",
        json=["127.0.0.1:50051"],
        headers={"Authorization": "Bearer user-secret"},
    )
    assert resp.status_code == 403
    fake._connect_worker.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Key file, role="admin" -- admin routes work
# ---------------------------------------------------------------------------


def test_admin_role_can_load_and_unload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_key_file(tmp_path, '[keys.ops]\nkey = "admin-secret"\nrole = "admin"\n', monkeypatch)
    _install_coordinator()
    headers = {"Authorization": "Bearer admin-secret"}

    load_resp = client.post("/v1/models/load", json={"model_name": "model-a"}, headers=headers)
    assert load_resp.status_code == 200

    unload_resp = client.delete("/v1/models/model-a", headers=headers)
    assert unload_resp.status_code == 200


# ---------------------------------------------------------------------------
# 5. Model scoping -- 403, never 404, identical message for real/unknown model
# ---------------------------------------------------------------------------


def test_scoped_key_denied_out_of_scope_model_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_key_file(
        tmp_path,
        '[keys.ci]\nkey = "scoped-secret"\nrole = "user"\nmodels = ["a"]\n',
        monkeypatch,
    )
    _install_coordinator()

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "b", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer scoped-secret"},
    )
    assert resp.status_code == 403


def test_scoped_key_existence_leak_identical_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same out-of-scope model name, before vs after it's registered in the
    real ModelRegistry -- the 403 must be byte-identical either way, proving
    the scope check never leaks whether the model actually exists."""
    _write_key_file(
        tmp_path,
        '[keys.ci]\nkey = "scoped-secret"\nrole = "user"\nmodels = ["a"]\n',
        monkeypatch,
    )
    _install_coordinator()
    headers = {"Authorization": "Bearer scoped-secret"}
    body = {"model": "existence-leak-test-model", "messages": [{"role": "user", "content": "hi"}]}

    unknown_resp = client.post("/v1/chat/completions", json=body, headers=headers)

    ModelRegistry.load_from_dict(
        {
            "models": {
                "existence-leak-test-model": {
                    "family": "qwen",
                    "parameters": "7B",
                    "min_memory_gb": 1,
                    "engine": "llamaserver",
                    "gguf": {"repo_id": "org/repo", "file": "model.gguf"},
                    "llamaserver": {"port": 8999},
                }
            }
        }
    )
    real_resp = client.post("/v1/chat/completions", json=body, headers=headers)

    assert unknown_resp.status_code == 403
    assert real_resp.status_code == 403
    assert unknown_resp.json() == real_resp.json()


# ---------------------------------------------------------------------------
# 6. Same scoping check on /v1/completions and /v1/embeddings
# ---------------------------------------------------------------------------


def test_scoped_key_denied_out_of_scope_model_completions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_key_file(
        tmp_path,
        '[keys.ci]\nkey = "scoped-secret"\nrole = "user"\nmodels = ["a"]\n',
        monkeypatch,
    )
    _install_coordinator()

    resp = client.post(
        "/v1/completions",
        json={"model": "b", "prompt": "hi"},
        headers={"Authorization": "Bearer scoped-secret"},
    )
    assert resp.status_code == 403


def test_scoped_key_denied_out_of_scope_model_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_key_file(
        tmp_path,
        '[keys.ci]\nkey = "scoped-secret"\nrole = "user"\nmodels = ["a"]\n',
        monkeypatch,
    )
    _install_coordinator()

    resp = client.post(
        "/v1/embeddings",
        json={"model": "b", "input": "hi"},
        headers={"Authorization": "Bearer scoped-secret"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 7. Unrestricted key reaches any model
# ---------------------------------------------------------------------------


def test_unrestricted_key_reaches_any_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_key_file(tmp_path, '[keys.ci]\nkey = "plain-secret"\n', monkeypatch)
    _install_coordinator()

    resp = client.post(
        "/v1/completions",
        json={"model": "model-a", "prompt": "hi"},
        headers={"Authorization": "Bearer plain-secret"},
    )
    # Not 403: the (fake) in-process engine dispatch runs past the scope gate.
    assert resp.status_code != 403


# ---------------------------------------------------------------------------
# 8. GET /v1/models is filtered for a scoped key, complete for an unscoped one
# ---------------------------------------------------------------------------


def test_list_models_filtered_for_scoped_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_key_file(
        tmp_path,
        '[keys.ci]\nkey = "scoped-secret"\nrole = "user"\nmodels = ["model-a"]\n',
        monkeypatch,
    )
    _install_coordinator()

    resp = client.get("/v1/models", headers={"Authorization": "Bearer scoped-secret"})
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert ids == ["model-a"]


def test_list_models_complete_for_unscoped_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_key_file(tmp_path, '[keys.ci]\nkey = "plain-secret"\n', monkeypatch)
    _install_coordinator()

    resp = client.get("/v1/models", headers={"Authorization": "Bearer plain-secret"})
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert ids == {"model-a", "model-b"}


def test_list_models_complete_when_auth_off() -> None:
    _install_coordinator()
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert ids == {"model-a", "model-b"}


# ---------------------------------------------------------------------------
# 9. Audit records
# ---------------------------------------------------------------------------


def test_load_emits_one_audit_record_with_caller_and_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _write_key_file(tmp_path, '[keys.ops]\nkey = "admin-secret"\nrole = "admin"\n', monkeypatch)
    _install_coordinator()

    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        resp = client.post(
            "/v1/models/load",
            json={"model_name": "model-a"},
            headers={"Authorization": "Bearer admin-secret"},
        )
    assert resp.status_code == 200

    records = [r for r in caplog.records if r.name == "coordinator.audit"]
    assert len(records) == 1
    payload = json.loads(records[0].message)
    assert payload["action"] == "model.load"
    assert payload["caller"] == "ops"
    assert payload["outcome"] == "success"


def test_denied_admin_route_emits_denied_audit_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _write_key_file(tmp_path, '[keys.ci]\nkey = "user-secret"\nrole = "user"\n', monkeypatch)
    _install_coordinator()

    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        resp = client.post(
            "/v1/models/load",
            json={"model_name": "model-a"},
            headers={"Authorization": "Bearer user-secret"},
        )
    assert resp.status_code == 403

    records = [r for r in caplog.records if r.name == "coordinator.audit"]
    assert len(records) == 1
    payload = json.loads(records[0].message)
    assert payload["action"] == "model.load"
    assert payload["caller"] == "ci"
    assert payload["outcome"] == "denied"


# ---------------------------------------------------------------------------
# 10. /infill's no-"model" fallback only ever sees the caller's own models
# ---------------------------------------------------------------------------

_INFILL_SCOPE_REGISTRY = {
    "models": {
        "infill-scope-a": {
            "family": "qwen",
            "parameters": "7B",
            "min_memory_gb": 1,
            "engine": "llamaserver",
            "gguf": {"repo_id": "org/repo-a", "file": "a.gguf"},
            "llamaserver": {"port": 9001},
        },
        "infill-scope-b": {
            "family": "qwen",
            "parameters": "7B",
            "min_memory_gb": 1,
            "engine": "llamaserver",
            "gguf": {"repo_id": "org/repo-b", "file": "b.gguf"},
            "llamaserver": {"port": 9002},
        },
        "infill-scope-c": {
            "family": "qwen",
            "parameters": "7B",
            "min_memory_gb": 1,
            "engine": "llamaserver",
            "gguf": {"repo_id": "org/repo-c", "file": "c.gguf"},
            "llamaserver": {"port": 9003},
        },
    }
}


def test_infill_no_model_resolves_within_scope_ignoring_out_of_scope_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two llamaserver models are loaded, but the key may only use one of
    them -- /infill (no "model" field) must resolve to that one, not 400."""
    ModelRegistry.load_from_dict(_INFILL_SCOPE_REGISTRY)
    _write_key_file(
        tmp_path,
        '[keys.ci]\nkey = "scoped-secret"\nrole = "user"\nmodels = ["infill-scope-a"]\n',
        monkeypatch,
    )
    _install_coordinator(
        loaded_llamaserver_models=AsyncMock(return_value=["infill-scope-a", "infill-scope-b"]),
        find_worker_for_model=AsyncMock(
            return_value=SimpleNamespace(address="10.0.0.9:50051", id="w1")
        ),
    )
    captured: Dict[str, Any] = {}

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        captured["url"] = url
        return proxy.BufferedProxyResponse(200, {"content-type": "application/json"}, b"{}")

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    resp = client.post(
        "/v1/infill",
        json={"input_prefix": "a", "input_suffix": "b"},
        headers={"Authorization": "Bearer scoped-secret"},
    )
    assert resp.status_code == 200
    assert captured["url"] == "http://10.0.0.9:9001/infill"  # infill-scope-a's port


def test_infill_no_model_multiple_in_scope_hint_excludes_out_of_scope_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """400's hint must list only models the key may use, never a model
    outside its scope, even when that model is also loaded."""
    ModelRegistry.load_from_dict(_INFILL_SCOPE_REGISTRY)
    _write_key_file(
        tmp_path,
        '[keys.ci]\nkey = "scoped-secret"\nrole = "user"\n'
        'models = ["infill-scope-a", "infill-scope-c"]\n',
        monkeypatch,
    )
    _install_coordinator(
        loaded_llamaserver_models=AsyncMock(
            return_value=["infill-scope-a", "infill-scope-b", "infill-scope-c"]
        ),
    )

    resp = client.post(
        "/v1/infill",
        json={"input_prefix": "a"},
        headers={"Authorization": "Bearer scoped-secret"},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "infill-scope-a" in detail
    assert "infill-scope-c" in detail
    assert "infill-scope-b" not in detail  # out of scope for this key


def test_infill_no_model_unrestricted_key_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrestricted key keeps today's behavior: the hint lists every
    loaded model."""
    ModelRegistry.load_from_dict(_INFILL_SCOPE_REGISTRY)
    _write_key_file(tmp_path, '[keys.ci]\nkey = "plain-secret"\n', monkeypatch)
    _install_coordinator(
        loaded_llamaserver_models=AsyncMock(return_value=["infill-scope-a", "infill-scope-b"]),
    )

    resp = client.post(
        "/v1/infill",
        json={"input_prefix": "a"},
        headers={"Authorization": "Bearer plain-secret"},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "infill-scope-a" in detail
    assert "infill-scope-b" in detail
