"""Integration tests: prove the middleware is wired into both endpoints at
the right point (before the prompt is built/forwarded) and is a true no-op
end-to-end when under budget or disabled."""
from typing import Any, Dict

import pytest

from coordinator.api import (
    ChatCompletionRequest,
    CompletionRequest,
    create_chat_completion,
    create_completion,
)
from coordinator.tests.conftest import make_settings


class _FakeCoordinator:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.active_requests: Dict[str, Any] = {}
        self.last_prompt: str = ""

    async def infer(self, model_name: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.last_prompt = prompt
        return {
            "request_id": "r1",
            "text": "ok",
            "tokens_generated": 1,
            "processing_time_ms": 1.0,
            "worker_id": "w1",
        }


def _fake_request(coordinator: _FakeCoordinator) -> Any:
    from types import SimpleNamespace

    app_state = SimpleNamespace(coordinator=coordinator)
    app = SimpleNamespace(state=app_state)
    return SimpleNamespace(app=app)


@pytest.mark.asyncio
async def test_chat_completions_noop_when_disabled() -> None:
    settings = make_settings(context_compression_enabled=False)
    coordinator = _FakeCoordinator(settings)
    body = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "```python\n" + "x=1\n" * 500 + "```\n"}]
    )
    await create_chat_completion(body, _fake_request(coordinator))
    assert "x=1" in coordinator.last_prompt  # untouched — not skeletonized


@pytest.mark.asyncio
async def test_chat_completions_compresses_when_enabled_and_over_budget() -> None:
    settings = make_settings(
        context_compression_enabled=True,
        context_compression_token_budget=10,
        context_compression_active_segments=0,
    )
    coordinator = _FakeCoordinator(settings)
    long_code = "```python\ndef old(x, y):\n    z = x + y\n    return z\n```\n"
    body = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": long_code * 5}])
    await create_chat_completion(body, _fake_request(coordinator))
    assert "z = x + y" not in coordinator.last_prompt
    assert "def old(x, y):" in coordinator.last_prompt


@pytest.mark.asyncio
async def test_chat_completions_per_request_override() -> None:
    settings = make_settings(
        context_compression_enabled=False,
        context_compression_token_budget=10,
        context_compression_active_segments=0,
    )
    coordinator = _FakeCoordinator(settings)
    long_code = "```python\ndef old(x, y):\n    z = x + y\n    return z\n```\n"
    body = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": long_code * 5}], compress_context=True
    )
    await create_chat_completion(body, _fake_request(coordinator))
    assert "z = x + y" not in coordinator.last_prompt  # server default is off, request forced it on


@pytest.mark.asyncio
async def test_completions_endpoint_also_wired() -> None:
    settings = make_settings(
        context_compression_enabled=True,
        context_compression_token_budget=10,
        context_compression_active_segments=0,
    )
    coordinator = _FakeCoordinator(settings)
    long_code = "```python\ndef old(x, y):\n    z = x + y\n    return z\n```\n"
    body = CompletionRequest(model="m", prompt=long_code * 5)
    await create_completion(body, _fake_request(coordinator))
    assert "z = x + y" not in coordinator.last_prompt
