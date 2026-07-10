"""Tests for the Plan 13 llamaserver HTTP proxy + coordinator engine dispatch.

The proxy unit tests never touch a real socket: they inject an httpx transport
(a ``MockTransport``) into the ``AsyncClient`` that ``coordinator.proxy`` builds
internally, so the raw request bytes and the streamed response are exercised
end-to-end in-process. The api dispatch tests drive the ``api.py`` route
handlers directly with a duck-typed Request + coordinator.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Dict, List

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from coordinator import proxy
from coordinator.api import (
    count_message_tokens,
    create_chat_completion,
    create_embeddings,
    create_infill,
    create_message,
)
from coordinator.models import ModelConfig, ModelFamily, ModelRegistry
from coordinator.tests.conftest import make_settings

_Handler = Callable[[httpx.Request], httpx.Response]


def _patch_transport(monkeypatch: Any, handler: _Handler) -> None:
    """Force ``coordinator.proxy``'s ``AsyncClient()`` onto a MockTransport.

    ``proxy`` constructs ``httpx.AsyncClient(...)`` itself, so we swap the class
    for a factory that pins a MockTransport (no real sockets). The real class is
    captured before the patch to build the client.
    """
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return real_client(transport=transport, **kwargs)

    # proxy.py does ``import httpx`` and calls ``httpx.AsyncClient`` at request
    # time, so patching the httpx module attribute reaches it.
    monkeypatch.setattr(httpx, "AsyncClient", factory)


# ---------------------------------------------------------------------------
# proxy.py unit tests (raw passthrough, SSE streaming, 502)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_passes_tools_body_through_byte_for_byte(monkeypatch: Any) -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content"] = request.content
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers.get("content-type")
        return httpx.Response(200, json={"ok": True})

    _patch_transport(monkeypatch, handler)

    body = json.dumps(
        {
            "model": "agentic",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_choice": "auto",
        }
    ).encode()

    result = await proxy.proxy_request(
        "POST",
        "http://10.0.0.5:8090/v1/chat/completions",
        body,
        {"content-type": "application/json"},
        stream=False,
    )

    assert isinstance(result, proxy.BufferedProxyResponse)
    assert result.status_code == 200
    # Byte-for-byte: `tools`/`tool_choice` survive because we never re-serialize.
    assert captured["content"] == body
    assert captured["method"] == "POST"
    assert captured["url"] == "http://10.0.0.5:8090/v1/chat/completions"
    assert captured["content_type"] == "application/json"


@pytest.mark.asyncio
async def test_proxy_preserves_upstream_status_code(monkeypatch: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    _patch_transport(monkeypatch, handler)

    result = await proxy.proxy_request(
        "POST", "http://h:8090/v1/completions", b"{}", {}, stream=False
    )
    assert isinstance(result, proxy.BufferedProxyResponse)
    assert result.status_code == 429


@pytest.mark.asyncio
async def test_proxy_streams_sse_chunks_incrementally(monkeypatch: Any) -> None:
    chunks = [b'data: {"a":1}\n\n', b'data: {"b":2}\n\n', b"data: [DONE]\n\n"]

    async def _stream() -> AsyncIterator[bytes]:
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_stream())

    _patch_transport(monkeypatch, handler)

    result = await proxy.proxy_request(
        "POST",
        "http://10.0.0.5:8090/v1/chat/completions",
        b"{}",
        {"content-type": "application/json"},
        stream=True,
    )

    assert isinstance(result, proxy.StreamingProxyResponse)
    assert result.status_code == 200
    assert result.headers.get("content-type") == "text/event-stream"

    received: List[bytes] = []
    async for chunk in result.body:
        received.append(chunk)
    # Arrived as discrete SSE events (proves incremental streaming, not one
    # buffered blob), and the terminator is intact.
    assert received == chunks
    assert b"".join(received).endswith(b"[DONE]\n\n")


@pytest.mark.asyncio
async def test_proxy_returns_502_on_connect_error(monkeypatch: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_transport(monkeypatch, handler)

    result = await proxy.proxy_request(
        "POST", "http://10.0.0.5:8090/v1/chat/completions", b"{}", {}, stream=False
    )
    assert isinstance(result, proxy.BufferedProxyResponse)
    assert result.status_code == 502
    payload = json.loads(result.content)
    assert payload["error"]["type"] == "upstream_unavailable"
    assert result.headers["content-type"] == "application/json"


@pytest.mark.asyncio
async def test_proxy_stream_connect_error_is_502(monkeypatch: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _patch_transport(monkeypatch, handler)

    result = await proxy.proxy_request(
        "POST", "http://10.0.0.5:8090/v1/chat/completions", b"{}", {}, stream=True
    )
    # A streaming request that can't even connect degrades to a buffered 502.
    assert isinstance(result, proxy.BufferedProxyResponse)
    assert result.status_code == 502


def test_filter_request_headers_strips_hop_by_hop_keeps_content_type() -> None:
    filtered = proxy.filter_request_headers(
        {
            "content-type": "application/json",
            "host": "coordinator:8000",
            "content-length": "42",
            "connection": "keep-alive",
            "authorization": "Bearer secret",
        }
    )
    assert filtered["content-type"] == "application/json"
    assert filtered["authorization"] == "Bearer secret"
    assert "host" not in filtered
    assert "content-length" not in filtered
    assert "connection" not in filtered


# ---------------------------------------------------------------------------
# api.py engine-dispatch tests (404 unloaded, 501 wrong engine, URL building)
# ---------------------------------------------------------------------------


def _register_llamaserver_model(name: str = "agentic-dispatch", port: int = 8190) -> ModelConfig:
    cfg = ModelConfig(
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
    )
    ModelRegistry.MODELS[name] = cfg
    return cfg


def _fake_request(
    coordinator: Any,
    body: Dict[str, Any],
    *,
    path: str,
    method: str = "POST",
    headers: Dict[str, str] | None = None,
) -> Any:
    raw = json.dumps(body).encode()

    async def _body() -> bytes:
        return raw

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(coordinator=coordinator)),
        url=SimpleNamespace(path=path),
        method=method,
        headers=headers or {"content-type": "application/json"},
        body=_body,
    )


class _NoWorkerCoordinator:
    """No worker holds the model AND auto-load is off — the 404 path."""

    def __init__(self) -> None:
        self.settings = SimpleNamespace(llamaserver_autoload=False)

    async def find_worker_for_model(self, name: str, session_id: Any = None) -> Any:
        return None


class _AutoloadCoordinator:
    """No worker holds the model; auto-load is on and succeeds (Task 5).

    ``ensure_llamaserver_model_loaded`` records its call count and returns a
    fixed worker so the api-level auto-load dispatch can be asserted without a
    real ClusterCoordinator.
    """

    def __init__(self, address: str, autoload: bool = True) -> None:
        self._address = address
        self.settings = SimpleNamespace(llamaserver_autoload=autoload)
        self.ensure_calls = 0

    async def find_worker_for_model(self, name: str, session_id: Any = None) -> Any:
        return None

    async def ensure_llamaserver_model_loaded(self, name: str, session_id: Any = None) -> Any:
        self.ensure_calls += 1
        return SimpleNamespace(address=self._address, id="w-auto")


class _AutoloadFailCoordinator:
    """Auto-load is on but the load fails/ times out (Task 5 → 503)."""

    def __init__(self, autoload: bool = True) -> None:
        self.settings = SimpleNamespace(llamaserver_autoload=autoload)

    async def find_worker_for_model(self, name: str, session_id: Any = None) -> Any:
        return None

    async def ensure_llamaserver_model_loaded(self, name: str, session_id: Any = None) -> Any:
        raise RuntimeError("llama-server health check timed out")


class _OneWorkerCoordinator:
    def __init__(self, address: str) -> None:
        self._address = address

    async def find_worker_for_model(self, name: str, session_id: Any = None) -> Any:
        return SimpleNamespace(address=self._address, id="w1")


@pytest.mark.asyncio
async def test_chat_completions_llamaserver_404_when_unloaded_and_autoload_off() -> None:
    # Auto-load disabled → Phase-1 behavior: 404 pointing at the load endpoint.
    _register_llamaserver_model()
    request = _fake_request(
        _NoWorkerCoordinator(),
        {"model": "agentic-dispatch", "messages": []},
        path="/v1/chat/completions",
    )
    with pytest.raises(HTTPException) as exc:
        await create_chat_completion(request)
    assert exc.value.status_code == 404
    assert "POST /v1/models/load" in exc.value.detail


@pytest.mark.asyncio
async def test_messages_501_for_non_llamaserver_engine() -> None:
    # llama3-8b is a Burn-engine model in the default registry.
    request = _fake_request(
        SimpleNamespace(),
        {"model": "llama3-8b", "messages": []},
        path="/v1/messages",
    )
    with pytest.raises(HTTPException) as exc:
        await create_message(request)
    assert exc.value.status_code == 501
    assert "llamaserver" in exc.value.detail


@pytest.mark.asyncio
async def test_messages_404_for_unknown_model() -> None:
    request = _fake_request(
        SimpleNamespace(),
        {"model": "does-not-exist", "messages": []},
        path="/v1/messages",
    )
    with pytest.raises(HTTPException) as exc:
        await create_message(request)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_messages_proxies_raw_body_to_worker_llamaserver(monkeypatch: Any) -> None:
    _register_llamaserver_model(name="agentic-msg", port=8195)
    captured: Dict[str, Any] = {}

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        captured.update(method=method, url=url, body=body, stream=stream)
        return proxy.BufferedProxyResponse(
            200, {"content-type": "application/json"}, b'{"id":"msg_1"}'
        )

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    anth_body = {
        "model": "agentic-msg",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "tools": [{"name": "x"}],
    }
    request = _fake_request(
        _OneWorkerCoordinator("192.168.1.50:50051"), anth_body, path="/v1/messages"
    )
    response = await create_message(request)

    assert response.status_code == 200
    # host portion of the worker's gRPC address + the model's llamaserver_port.
    assert captured["url"] == "http://192.168.1.50:8195/v1/messages"
    assert captured["method"] == "POST"
    assert json.loads(captured["body"]) == anth_body  # raw body incl. tools
    assert captured["stream"] is True  # sniffed from the Anthropic body


@pytest.mark.asyncio
async def test_chat_completions_llamaserver_returns_streaming_response(monkeypatch: Any) -> None:
    _register_llamaserver_model(name="agentic-stream", port=8196)

    async def _chunks() -> AsyncIterator[bytes]:
        yield b"data: hi\n\n"

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        assert stream is True
        return proxy.StreamingProxyResponse(200, {"content-type": "text/event-stream"}, _chunks())

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    body = {
        "model": "agentic-stream",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }
    request = _fake_request(
        _OneWorkerCoordinator("10.0.0.9:50051"), body, path="/v1/chat/completions"
    )
    response = await create_chat_completion(request)

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"


@pytest.mark.asyncio
async def test_count_tokens_never_streams(monkeypatch: Any) -> None:
    _register_llamaserver_model(name="agentic-count", port=8197)
    captured: Dict[str, Any] = {}

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        captured.update(stream=stream, url=url)
        return proxy.BufferedProxyResponse(200, {"content-type": "application/json"}, b"{}")

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    # Even with stream:true in the body, count_tokens must not stream.
    body = {"model": "agentic-count", "messages": [], "stream": True}
    request = _fake_request(
        _OneWorkerCoordinator("10.0.0.9:50051"), body, path="/v1/messages/count_tokens"
    )
    await count_message_tokens(request)

    assert captured["stream"] is False
    assert captured["url"].endswith("/v1/messages/count_tokens")


# ---------------------------------------------------------------------------
# Task 5 — auto-load-on-demand (coordinator single-flight + api-level gate)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_autoload_single_flight_loads_once(monkeypatch: Any) -> None:
    """Concurrent requests for the same cold model trigger EXACTLY ONE load."""
    from unittest.mock import AsyncMock

    from coordinator.coordinator import ClusterCoordinator, WorkerInfo, WorkerState

    _register_llamaserver_model(name="agentic-sf", port=8300)
    coord = ClusterCoordinator(make_settings())
    worker = WorkerInfo(id="w1", address="10.0.0.1:50051", channel=AsyncMock(), stub=AsyncMock())
    worker.state = WorkerState.HEALTHY
    coord.workers["w1"] = worker

    load_calls = 0

    async def fake_load(w: Any, name: str, quantization: Any = None) -> bool:
        nonlocal load_calls
        load_calls += 1
        # Hold the single-flight lock while the other callers pile up behind it.
        await asyncio.sleep(0.05)
        return True

    monkeypatch.setattr(coord, "_load_model_on_worker", fake_load)

    results = await asyncio.gather(
        *[coord.ensure_llamaserver_model_loaded("agentic-sf") for _ in range(12)]
    )

    assert load_calls == 1  # single-flight: one load despite 12 concurrent callers
    assert all(r is worker for r in results)
    assert "agentic-sf" in worker.loaded_models  # load reflected for later requests


@pytest.mark.asyncio
async def test_autoload_no_healthy_worker_raises() -> None:
    from coordinator.coordinator import ClusterCoordinator

    _register_llamaserver_model(name="agentic-noworker", port=8301)
    coord = ClusterCoordinator(make_settings())
    with pytest.raises(RuntimeError, match="No healthy worker"):
        await coord.ensure_llamaserver_model_loaded("agentic-noworker")


@pytest.mark.asyncio
async def test_autoload_load_failure_raises(monkeypatch: Any) -> None:
    from unittest.mock import AsyncMock

    from coordinator.coordinator import ClusterCoordinator, WorkerInfo, WorkerState

    _register_llamaserver_model(name="agentic-loadfail", port=8302)
    coord = ClusterCoordinator(make_settings())
    worker = WorkerInfo(id="w1", address="10.0.0.1:50051", channel=AsyncMock(), stub=AsyncMock())
    worker.state = WorkerState.HEALTHY
    coord.workers["w1"] = worker

    async def fake_load(w: Any, name: str, quantization: Any = None) -> bool:
        return False

    monkeypatch.setattr(coord, "_load_model_on_worker", fake_load)

    with pytest.raises(RuntimeError, match="Failed to load"):
        await coord.ensure_llamaserver_model_loaded("agentic-loadfail")


@pytest.mark.asyncio
async def test_chat_completions_autoload_triggers_and_proxies(monkeypatch: Any) -> None:
    """Auto-load on + model unloaded → load once, then proxy to that worker."""
    _register_llamaserver_model(name="agentic-auto", port=8305)
    captured: Dict[str, Any] = {}

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        captured.update(url=url, method=method)
        return proxy.BufferedProxyResponse(200, {"content-type": "application/json"}, b"{}")

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    coord = _AutoloadCoordinator("192.168.1.77:50051", autoload=True)
    request = _fake_request(
        coord, {"model": "agentic-auto", "messages": []}, path="/v1/chat/completions"
    )
    response = await create_chat_completion(request)

    assert isinstance(response, Response)
    assert response.status_code == 200
    assert coord.ensure_calls == 1
    assert captured["url"] == "http://192.168.1.77:8305/v1/chat/completions"


@pytest.mark.asyncio
async def test_chat_completions_autoload_failure_returns_503() -> None:
    _register_llamaserver_model(name="agentic-503", port=8306)
    request = _fake_request(
        _AutoloadFailCoordinator(autoload=True),
        {"model": "agentic-503", "messages": []},
        path="/v1/chat/completions",
    )
    with pytest.raises(HTTPException) as exc:
        await create_chat_completion(request)
    assert exc.value.status_code == 503
    assert "health check timed out" in exc.value.detail


@pytest.mark.asyncio
async def test_loaded_llamaserver_models_filters_by_engine() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from coordinator.coordinator import ClusterCoordinator, WorkerInfo, WorkerState

    _register_llamaserver_model(name="ls-loaded", port=8307)
    coord = ClusterCoordinator(make_settings())
    worker = WorkerInfo(id="w1", address="a:1", channel=AsyncMock(), stub=AsyncMock())
    worker.state = WorkerState.HEALTHY
    # ls-loaded is engine=llamaserver; deepseek-7b is a Burn model — filtered out.
    worker.loaded_models = {"ls-loaded": MagicMock(), "deepseek-7b": MagicMock()}
    coord.workers["w1"] = worker

    assert await coord.loaded_llamaserver_models() == ["ls-loaded"]


# ---------------------------------------------------------------------------
# Task 6 — /v1/embeddings + /infill proxy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embeddings_proxies_buffered(monkeypatch: Any) -> None:
    _register_llamaserver_model(name="embed-model", port=8310)
    captured: Dict[str, Any] = {}

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        captured.update(url=url, stream=stream, body=body)
        return proxy.BufferedProxyResponse(
            200, {"content-type": "application/json"}, b'{"data":[]}'
        )

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    body = {"model": "embed-model", "input": "hello"}
    request = _fake_request(_OneWorkerCoordinator("10.0.0.2:50051"), body, path="/v1/embeddings")
    response = await create_embeddings(request)

    assert response.status_code == 200
    assert captured["url"] == "http://10.0.0.2:8310/v1/embeddings"
    assert captured["stream"] is False  # embeddings are never SSE
    assert json.loads(captured["body"]) == body


@pytest.mark.asyncio
async def test_embeddings_501_for_non_llamaserver_engine() -> None:
    # llama3-8b is a Burn-engine model in the default registry.
    request = _fake_request(
        SimpleNamespace(), {"model": "llama3-8b", "input": "x"}, path="/v1/embeddings"
    )
    with pytest.raises(HTTPException) as exc:
        await create_embeddings(request)
    assert exc.value.status_code == 501
    assert "llamaserver" in exc.value.detail


@pytest.mark.asyncio
async def test_embeddings_404_for_unknown_model() -> None:
    request = _fake_request(
        SimpleNamespace(), {"model": "does-not-exist", "input": "x"}, path="/v1/embeddings"
    )
    with pytest.raises(HTTPException) as exc:
        await create_embeddings(request)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_infill_explicit_model_proxies_to_root_path(monkeypatch: Any) -> None:
    _register_llamaserver_model(name="infill-a", port=8320)
    captured: Dict[str, Any] = {}

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        captured.update(url=url, stream=stream, body=body)
        return proxy.BufferedProxyResponse(200, {"content-type": "application/json"}, b"{}")

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    body = {"model": "infill-a", "input_prefix": "def f(", "input_suffix": "):"}
    request = _fake_request(_OneWorkerCoordinator("10.0.0.3:50051"), body, path="/v1/infill")
    await create_infill(request)

    # Upstream path is pinned to the llama-server root /infill, NOT /v1/infill.
    assert captured["url"] == "http://10.0.0.3:8320/infill"
    assert captured["stream"] is False
    # Raw body forwarded unmodified — the `model` key is left in place.
    assert json.loads(captured["body"]) == body


@pytest.mark.asyncio
async def test_infill_single_loaded_fallback(monkeypatch: Any) -> None:
    _register_llamaserver_model(name="infill-solo", port=8321)
    captured: Dict[str, Any] = {}

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        captured.update(url=url)
        return proxy.BufferedProxyResponse(200, {"content-type": "application/json"}, b"{}")

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    class _SingleLoadedCoord:
        async def loaded_llamaserver_models(self) -> List[str]:
            return ["infill-solo"]

        async def find_worker_for_model(self, name: str, session_id: Any = None) -> Any:
            return SimpleNamespace(address="10.0.0.4:50051", id="w1")

    body = {"input_prefix": "a", "input_suffix": "b"}  # no "model" field
    request = _fake_request(_SingleLoadedCoord(), body, path="/v1/infill")
    await create_infill(request)

    assert captured["url"] == "http://10.0.0.4:8321/infill"


@pytest.mark.asyncio
async def test_infill_no_model_zero_loaded_returns_400() -> None:
    class _NoneLoadedCoord:
        async def loaded_llamaserver_models(self) -> List[str]:
            return []

    request = _fake_request(_NoneLoadedCoord(), {"input_prefix": "a"}, path="/v1/infill")
    with pytest.raises(HTTPException) as exc:
        await create_infill(request)
    assert exc.value.status_code == 400
    assert '"model"' in exc.value.detail


@pytest.mark.asyncio
async def test_infill_no_model_multiple_loaded_returns_400() -> None:
    class _MultiLoadedCoord:
        async def loaded_llamaserver_models(self) -> List[str]:
            return ["infill-m1", "infill-m2"]

    request = _fake_request(_MultiLoadedCoord(), {"input_prefix": "a"}, path="/v1/infill")
    with pytest.raises(HTTPException) as exc:
        await create_infill(request)
    assert exc.value.status_code == 400
    assert "multiple" in exc.value.detail


@pytest.mark.asyncio
async def test_infill_streams_when_stream_true(monkeypatch: Any) -> None:
    _register_llamaserver_model(name="infill-stream", port=8322)

    async def _chunks() -> AsyncIterator[bytes]:
        yield b"data: x\n\n"

    async def fake_proxy_request(
        method: str, url: str, body: bytes, headers: Dict[str, str], stream: bool
    ) -> proxy.ProxyResponse:
        assert stream is True  # sniffed from the body's "stream": true
        return proxy.StreamingProxyResponse(200, {"content-type": "text/event-stream"}, _chunks())

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    body = {"model": "infill-stream", "input_prefix": "a", "stream": True}
    request = _fake_request(_OneWorkerCoordinator("10.0.0.5:50051"), body, path="/v1/infill")
    response = await create_infill(request)

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
