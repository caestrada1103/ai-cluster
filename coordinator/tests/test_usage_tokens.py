"""Regression tests for `usage.prompt_tokens` on the in-process chat completion
path (coordinator/api.py::_build_flat_response) and its plumbing through
coordinator.py.

Before the fix, `usage.prompt_tokens` was hardcoded to 0 because the gRPC
`InferenceResponse` (proto/cluster.proto) carried no prompt-token count. A
later, since-rejected attempt papered over this by estimating prompt tokens
coordinator-side from a character heuristic — an approximate
`usage.prompt_tokens` is worse than an obvious absence, since clients bill
against it. `InferenceResponse.prompt_tokens` is now a real, explicit-presence
proto3 `optional uint32` field the worker sets when it can report a real
tokenized prompt length; the coordinator forwards that value verbatim and,
when the worker didn't set it, omits `prompt_tokens`/`total_tokens` from
`usage` rather than guessing.
"""
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pytest

import coordinator.proto.cluster_pb2 as pb
from coordinator.api import ChatCompletionRequest, _build_flat_response, create_chat_completion
from coordinator.coordinator import ClusterCoordinator, RequestContext, WorkerInfo, WorkerState
from coordinator.tests.conftest import make_settings

# A number no character-based heuristic would ever produce for the short
# prompt used below, so a regression to estimating fails these tests loudly.
_REAL_PROMPT_TOKENS = 4177


class _FakeCoordinator:
    """Mimics ClusterCoordinator.infer() just enough for the API layer."""

    def __init__(self, settings: Any, completion_tokens: int = 200) -> None:
        self.settings = settings
        self.active_requests: Dict[str, Any] = {}
        self.last_prompt: str = ""
        self._completion_tokens = completion_tokens
        self._prompt_tokens: Optional[int] = _REAL_PROMPT_TOKENS

    async def infer(self, model_name: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.last_prompt = prompt
        return {
            "request_id": "r1",
            "text": "ok",
            "tokens_generated": self._completion_tokens,
            "prompt_tokens": self._prompt_tokens,
            "processing_time_ms": 1.0,
            "worker_id": "w1",
        }


def _fake_request(coordinator: _FakeCoordinator, body: Any) -> Any:
    """Build a duck-typed Request whose .body() returns the model's raw JSON.

    Mirrors the helper in test_context_compression_api.py. Model "m" is
    unknown to the registry, so this always takes the in-process path.
    """
    raw: bytes = body.model_dump_json().encode()

    async def _body() -> bytes:
        return raw

    app_state = SimpleNamespace(coordinator=coordinator)
    app = SimpleNamespace(state=app_state)
    return SimpleNamespace(app=app, state=SimpleNamespace(), body=_body)


def test_build_flat_response_reports_real_worker_prompt_tokens() -> None:
    """The helper must forward the worker-reported count verbatim, not estimate."""
    result = {
        "request_id": "r1",
        "text": "hi",
        "tokens_generated": 200,
        "prompt_tokens": _REAL_PROMPT_TOKENS,
    }
    resp = _build_flat_response(result, "some-model")

    assert resp["usage"]["prompt_tokens"] == _REAL_PROMPT_TOKENS
    assert resp["usage"]["completion_tokens"] == 200
    assert resp["usage"]["total_tokens"] == _REAL_PROMPT_TOKENS + 200


def test_build_flat_response_omits_prompt_and_total_when_worker_reports_nothing() -> None:
    """When the worker couldn't report a count, never guess: omit both keys."""
    result = {"request_id": "r1", "text": "hi", "tokens_generated": 5, "prompt_tokens": None}
    resp = _build_flat_response(result, "some-model")

    assert resp["usage"] == {"completion_tokens": 5}
    assert "prompt_tokens" not in resp["usage"]
    assert "total_tokens" not in resp["usage"]


def test_build_flat_response_missing_key_is_treated_as_absent() -> None:
    """Older fakes/dicts that never set `prompt_tokens` must not KeyError."""
    result = {"request_id": "r1", "text": "hi", "tokens_generated": 5}
    resp = _build_flat_response(result, "some-model")

    assert resp["usage"] == {"completion_tokens": 5}


@pytest.mark.asyncio
async def test_chat_completion_endpoint_reports_real_prompt_tokens() -> None:
    """End-to-end through create_chat_completion: mirrors the real-hardware repro
    (`usage: {prompt_tokens: 0, completion_tokens: 200, total_tokens: 200}`) and
    asserts the real worker-reported count is surfaced instead."""
    settings = make_settings(context_compression_enabled=False)
    coordinator = _FakeCoordinator(settings, completion_tokens=200)
    body = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "Explain the theory of relativity in detail."}],
    )
    response = await create_chat_completion(_fake_request(coordinator, body))

    assert isinstance(response, dict)
    usage = response["usage"]
    assert usage["prompt_tokens"] == _REAL_PROMPT_TOKENS
    assert usage["completion_tokens"] == 200
    assert usage["total_tokens"] == _REAL_PROMPT_TOKENS + 200


@pytest.mark.asyncio
async def test_chat_completion_endpoint_omits_usage_fields_when_worker_silent() -> None:
    """The end-to-end path must also omit, never estimate, when the worker
    sends no prompt-token count."""
    settings = make_settings(context_compression_enabled=False)
    coordinator = _FakeCoordinator(settings, completion_tokens=200)
    coordinator._prompt_tokens = None
    body = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "Explain the theory of relativity in detail."}],
    )
    response = await create_chat_completion(_fake_request(coordinator, body))

    assert isinstance(response, dict)
    usage = response["usage"]
    assert usage == {"completion_tokens": 200}


@pytest.mark.asyncio
async def test_execute_request_captures_prompt_tokens_only_when_hasfield_true() -> None:
    """coordinator._execute_request must distinguish HasField-absent from
    present-zero: an explicit 0 is a real count, absence stays None."""
    settings = make_settings(request_timeout=5)
    coord = ClusterCoordinator(settings)
    worker = WorkerInfo(id="w1", address="a:1", channel=AsyncMock(), stub=AsyncMock())
    worker.state = WorkerState.HEALTHY
    worker.loaded_models = {"deepseek-7b": object()}

    response_no_field = pb.InferenceResponse(
        request_id="r1", text="hi", tokens_generated=3, finished=True
    )
    assert response_no_field.HasField("prompt_tokens") is False

    async def _stream_no_field(*args: Any, **kwargs: Any) -> Any:
        yield response_no_field

    worker.stub.Infer = lambda *a, **k: _stream_no_field()

    ctx = RequestContext(
        id="r1", model_name="deepseek-7b", prompt="hi", params={}, created_at=time.time()
    )
    await coord._execute_request(ctx, worker)
    assert ctx.prompt_tokens is None

    # Now the worker explicitly reports a real, legitimate zero.
    worker.active_requests = 0
    response_zero = pb.InferenceResponse(
        request_id="r2", text="hi", tokens_generated=3, finished=True, prompt_tokens=0
    )
    assert response_zero.HasField("prompt_tokens") is True

    async def _stream_zero(*args: Any, **kwargs: Any) -> Any:
        yield response_zero

    worker.stub.Infer = lambda *a, **k: _stream_zero()

    ctx2 = RequestContext(
        id="r2", model_name="deepseek-7b", prompt="hi", params={}, created_at=time.time()
    )
    await coord._execute_request(ctx2, worker)
    assert ctx2.prompt_tokens == 0
