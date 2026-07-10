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
from fastapi.responses import StreamingResponse

from coordinator import proxy
from coordinator.api import (
    count_message_tokens,
    create_chat_completion,
    create_message,
)
from coordinator.models import ModelConfig, ModelFamily, ModelRegistry

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
    async def find_worker_for_model(self, name: str, session_id: Any = None) -> Any:
        return None


class _OneWorkerCoordinator:
    def __init__(self, address: str) -> None:
        self._address = address

    async def find_worker_for_model(self, name: str, session_id: Any = None) -> Any:
        return SimpleNamespace(address=self._address, id="w1")


@pytest.mark.asyncio
async def test_chat_completions_llamaserver_404_when_unloaded() -> None:
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
