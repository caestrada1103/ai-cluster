"""Tests for coordinator-side best-effort llama-server slot affinity.

`_slot_for_caller` and `_apply_slot_affinity` are exercised directly (see
coordinator/api.py); one test drives `_proxy_to_llamaserver` end-to-end-ish to
confirm the forwarded bytes match what `_apply_slot_affinity` would produce.
"""

import json
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from coordinator import identity, proxy
from coordinator.api import (
    _SLOT_AFFINITY_PATHS,
    _apply_slot_affinity,
    _proxy_to_llamaserver,
    _slot_for_caller,
)
from coordinator.identity import Caller
from coordinator.models import ModelConfig, ModelFamily


@pytest.fixture(autouse=True)
def _clear_identity_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COORDINATOR_API_KEYS", raising=False)
    monkeypatch.delenv("COORDINATOR_API_KEY_FILE", raising=False)
    identity._file_cache.clear()


def _make_llamaserver_model(
    name: str = "agentic-slots", port: int = 8290, parallel: Optional[int] = 4
) -> ModelConfig:
    """Build a standalone ModelConfig -- never touches ModelRegistry.MODELS.

    `_apply_slot_affinity`/`_proxy_to_llamaserver` take `model_cfg` as a plain
    argument and never consult the registry, so tests don't need it either.
    """
    return ModelConfig(
        name=name,
        family=ModelFamily.QWEN,
        parameters="7B",
        min_memory_gb=6,
        recommended_gpus=1,
        max_gpus=1,
        num_layers=0,
        hidden_size=0,
        num_attention_heads=0,
        vocab_size=0,
        max_seq_len=8192,
        intermediate_size=0,
        engine="llamaserver",
        gguf_repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        gguf_file="qwen2.5-7b-instruct-q4_k_m.gguf",
        llamaserver_port=port,
        llamaserver_parallel=parallel,
    )


def _fake_request(
    coordinator: Any,
    body: Dict[str, Any],
    *,
    path: str = "/v1/chat/completions",
    caller: Optional[Caller] = None,
) -> Any:
    raw = json.dumps(body).encode()

    async def _body() -> bytes:
        return raw

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(coordinator=coordinator)),
        state=SimpleNamespace(caller=caller) if caller is not None else SimpleNamespace(),
        url=SimpleNamespace(path=path),
        method="POST",
        headers={"content-type": "application/json"},
        body=_body,
    )


def _coordinator(*, slot_affinity: bool = True) -> Any:
    return SimpleNamespace(settings=SimpleNamespace(llamaserver_slot_affinity=slot_affinity))


_CALLER_A = Caller(id="caller-a", role="user", models=frozenset())
_CALLER_B = Caller(id="ci-runner", role="user", models=frozenset())


# ---------------------------------------------------------------------------
# _slot_for_caller
# ---------------------------------------------------------------------------


def test_slot_for_caller_known_value_regression() -> None:
    # Pins the hash choice: a future change to the algorithm must update this.
    assert _slot_for_caller("caller-a", 4) == 1
    assert _slot_for_caller("ci-runner", 4) == 2


def test_slot_for_caller_is_deterministic() -> None:
    assert _slot_for_caller("caller-a", 4) == _slot_for_caller("caller-a", 4)


def test_slot_for_caller_in_range() -> None:
    for cid in ("caller-a", "ci-runner", "ops", "anonymous", ""):
        slot = _slot_for_caller(cid, 8)
        assert 0 <= slot < 8


def test_slot_for_caller_different_ids_can_differ() -> None:
    assert _slot_for_caller("caller-a", 4) != _slot_for_caller("ci-runner", 4)


# ---------------------------------------------------------------------------
# _apply_slot_affinity -- injection happens
# ---------------------------------------------------------------------------


def test_injects_id_slot_when_all_conditions_met(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", "/some/file.toml")
    model_cfg = _make_llamaserver_model(name="agentic-inject", port=8330, parallel=4)
    data = {"model": "agentic-inject", "messages": [{"role": "user", "content": "hi"}]}
    raw = json.dumps(data).encode()
    request = _fake_request(None, data, caller=_CALLER_A)

    result = _apply_slot_affinity(
        request, _coordinator(), model_cfg, raw, data, "/v1/chat/completions"
    )

    parsed = json.loads(result)
    assert parsed["id_slot"] == _slot_for_caller("caller-a", 4)
    for key, value in data.items():
        assert parsed[key] == value


# ---------------------------------------------------------------------------
# _apply_slot_affinity -- body returned unchanged
# ---------------------------------------------------------------------------


def test_unchanged_when_no_identity_file(monkeypatch: pytest.MonkeyPatch) -> None:
    model_cfg = _make_llamaserver_model(name="agentic-noid", port=8331, parallel=4)
    data = {"model": "agentic-noid", "messages": []}
    raw = json.dumps(data).encode()
    request = _fake_request(None, data, caller=_CALLER_A)

    result = _apply_slot_affinity(
        request, _coordinator(), model_cfg, raw, data, "/v1/chat/completions"
    )
    assert result is raw


def test_unchanged_when_caller_is_unrestricted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", "/some/file.toml")
    model_cfg = _make_llamaserver_model(name="agentic-unres", port=8332, parallel=4)
    data = {"model": "agentic-unres", "messages": []}
    raw = json.dumps(data).encode()
    request = _fake_request(None, data, caller=None)  # no request.state.caller -> UNRESTRICTED

    result = _apply_slot_affinity(
        request, _coordinator(), model_cfg, raw, data, "/v1/chat/completions"
    )
    assert result is raw


def test_unchanged_when_path_not_affinity_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", "/some/file.toml")
    model_cfg = _make_llamaserver_model(name="agentic-path", port=8333, parallel=4)
    data = {"model": "agentic-path", "messages": []}
    raw = json.dumps(data).encode()
    request = _fake_request(None, data, caller=_CALLER_A, path="/v1/messages")
    assert "/v1/messages" not in _SLOT_AFFINITY_PATHS

    result = _apply_slot_affinity(request, _coordinator(), model_cfg, raw, data, "/v1/messages")
    assert result is raw


def test_unchanged_when_parallel_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", "/some/file.toml")
    model_cfg = _make_llamaserver_model(name="agentic-noparallel", port=8334, parallel=None)
    data = {"model": "agentic-noparallel", "messages": []}
    raw = json.dumps(data).encode()
    request = _fake_request(None, data, caller=_CALLER_A)

    result = _apply_slot_affinity(
        request, _coordinator(), model_cfg, raw, data, "/v1/chat/completions"
    )
    assert result is raw


def test_unchanged_when_parallel_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", "/some/file.toml")
    model_cfg = _make_llamaserver_model(name="agentic-single", port=8335, parallel=1)
    data = {"model": "agentic-single", "messages": []}
    raw = json.dumps(data).encode()
    request = _fake_request(None, data, caller=_CALLER_A)

    result = _apply_slot_affinity(
        request, _coordinator(), model_cfg, raw, data, "/v1/chat/completions"
    )
    assert result is raw


def test_unchanged_when_id_slot_already_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", "/some/file.toml")
    model_cfg = _make_llamaserver_model(name="agentic-explicit", parallel=4)
    data = {"model": "agentic-explicit", "messages": [], "id_slot": 3}
    raw = json.dumps(data).encode()
    request = _fake_request(None, data, caller=_CALLER_A)

    result = _apply_slot_affinity(
        request, _coordinator(), model_cfg, raw, data, "/v1/chat/completions"
    )
    assert result is raw


def test_unchanged_when_setting_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", "/some/file.toml")
    model_cfg = _make_llamaserver_model(name="agentic-disabled", parallel=4)
    data = {"model": "agentic-disabled", "messages": []}
    raw = json.dumps(data).encode()
    request = _fake_request(None, data, caller=_CALLER_A)

    result = _apply_slot_affinity(
        request, _coordinator(slot_affinity=False), model_cfg, raw, data, "/v1/chat/completions"
    )
    assert result is raw


def test_unchanged_when_data_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", "/some/file.toml")
    model_cfg = _make_llamaserver_model(name="agentic-nodata", parallel=4)
    raw = b"not-actually-json-but-doesnt-matter-here"
    request = _fake_request(None, {}, caller=_CALLER_A)

    result = _apply_slot_affinity(
        request, _coordinator(), model_cfg, raw, None, "/v1/chat/completions"
    )
    assert result is raw


# ---------------------------------------------------------------------------
# Two callers, same model -> consistent, independent slots
# ---------------------------------------------------------------------------


def test_two_callers_get_consistent_independent_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", "/some/file.toml")
    model_cfg = _make_llamaserver_model(name="agentic-two-callers", parallel=4)
    data = {"model": "agentic-two-callers", "messages": []}
    raw = json.dumps(data).encode()

    slots_a = set()
    slots_b = set()
    for _ in range(3):
        req_a = _fake_request(None, data, caller=_CALLER_A)
        req_b = _fake_request(None, data, caller=_CALLER_B)
        result_a = _apply_slot_affinity(
            req_a, _coordinator(), model_cfg, raw, data, "/v1/chat/completions"
        )
        result_b = _apply_slot_affinity(
            req_b, _coordinator(), model_cfg, raw, data, "/v1/chat/completions"
        )
        slots_a.add(json.loads(result_a)["id_slot"])
        slots_b.add(json.loads(result_b)["id_slot"])

    assert len(slots_a) == 1
    assert len(slots_b) == 1
    assert slots_a != slots_b


# ---------------------------------------------------------------------------
# End-to-end-ish: through _proxy_to_llamaserver
# ---------------------------------------------------------------------------


class _OneWorkerCoordinator:
    def __init__(self, address: str, slot_affinity: bool = True) -> None:
        self._address = address
        self.settings = SimpleNamespace(llamaserver_slot_affinity=slot_affinity)

    async def find_worker_for_model(self, name: str, session_id: Any = None) -> Any:
        return SimpleNamespace(address=self._address, id="w1")


@pytest.mark.asyncio
async def test_proxy_forwards_id_slot_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", "/some/file.toml")
    model_cfg = _make_llamaserver_model(name="agentic-e2e", port=8291, parallel=4)
    captured: Dict[str, Any] = {}

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        captured["body"] = body
        return proxy.BufferedProxyResponse(200, {"content-type": "application/json"}, b"{}")

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    data = {"model": "agentic-e2e", "messages": [{"role": "user", "content": "hi"}]}
    raw = json.dumps(data).encode()
    request = _fake_request(_OneWorkerCoordinator("10.0.0.9:50051"), data, caller=_CALLER_A)

    await _proxy_to_llamaserver(
        request, request.app.state.coordinator, model_cfg, raw, False, data=data
    )

    forwarded = json.loads(captured["body"])
    assert forwarded["id_slot"] == _slot_for_caller("caller-a", 4)
    assert forwarded["messages"] == data["messages"]


@pytest.mark.asyncio
async def test_proxy_forwards_raw_bytes_unchanged_when_no_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_cfg = _make_llamaserver_model(name="agentic-e2e-plain", port=8292, parallel=4)
    captured: Dict[str, Any] = {}

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        captured["body"] = body
        return proxy.BufferedProxyResponse(200, {"content-type": "application/json"}, b"{}")

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    data = {"model": "agentic-e2e-plain", "messages": [{"role": "user", "content": "hi"}]}
    raw = json.dumps(data).encode()
    request = _fake_request(_OneWorkerCoordinator("10.0.0.9:50051"), data, caller=_CALLER_A)

    await _proxy_to_llamaserver(
        request, request.app.state.coordinator, model_cfg, raw, False, data=data
    )

    assert captured["body"] == raw
