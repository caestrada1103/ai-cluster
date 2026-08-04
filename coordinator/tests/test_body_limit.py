"""Tests for coordinator.body_limit — request-body size cap."""

from typing import List

import pytest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from coordinator.body_limit import BodySizeLimitMiddleware


async def _echo_app(scope: Scope, receive: Receive, send: Send) -> None:
    """A minimal ASGI app that reads the full body then replies 200."""
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": b'{"ok": true}'})


def _http_scope(headers: List[tuple[bytes, bytes]]) -> Scope:
    return {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": headers,
    }


async def _run(app: ASGIApp, scope: Scope, body_chunks: List[bytes]) -> List[Message]:
    sent: List[Message] = []
    remaining = list(body_chunks)

    async def receive() -> Message:
        if not remaining:
            return {"type": "http.request", "body": b"", "more_body": False}
        chunk = remaining.pop(0)
        return {"type": "http.request", "body": chunk, "more_body": bool(remaining)}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent


def _status(sent: List[Message]) -> int:
    start = next(m for m in sent if m["type"] == "http.response.start")
    return int(start["status"])


@pytest.mark.asyncio
async def test_allows_body_under_the_limit() -> None:
    app = BodySizeLimitMiddleware(_echo_app, max_bytes=1024)
    scope = _http_scope([(b"content-length", b"12")])
    sent = await _run(app, scope, [b'{"ok": true}'])
    assert _status(sent) == 200


@pytest.mark.asyncio
async def test_rejects_via_content_length_header_before_reading_body() -> None:
    """The common case: a normal client declares an oversized Content-Length
    up front — rejected before the wrapped app ever runs."""
    app_called = False

    async def spy_app(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal app_called
        app_called = True  # pragma: no cover

    app = BodySizeLimitMiddleware(spy_app, max_bytes=1024)
    scope = _http_scope([(b"content-length", b"999999")])
    sent = await _run(app, scope, [b"x" * 10])
    assert _status(sent) == 413
    assert app_called is False


@pytest.mark.asyncio
async def test_rejects_streamed_body_exceeding_cap_with_no_content_length() -> None:
    """A chunked-transfer body with no Content-Length must still be capped."""
    app = BodySizeLimitMiddleware(_echo_app, max_bytes=16)
    scope = _http_scope([])  # no content-length header at all
    big_chunks = [b"a" * 10, b"b" * 10, b"c" * 10]  # 30 bytes total, cap is 16
    sent = await _run(app, scope, big_chunks)
    assert _status(sent) == 413


@pytest.mark.asyncio
async def test_malformed_content_length_falls_back_to_streaming_check() -> None:
    app = BodySizeLimitMiddleware(_echo_app, max_bytes=1024)
    scope = _http_scope([(b"content-length", b"not-a-number")])
    sent = await _run(app, scope, [b"small"])
    assert _status(sent) == 200


@pytest.mark.asyncio
async def test_non_http_scope_passes_through_untouched() -> None:
    calls = []

    async def lifespan_app(scope: Scope, receive: Receive, send: Send) -> None:
        calls.append(scope["type"])

    async def noop_receive() -> Message:
        return {"type": "lifespan.startup"}

    async def noop_send(message: Message) -> None:
        return None

    app = BodySizeLimitMiddleware(lifespan_app, max_bytes=1024)
    await app({"type": "lifespan"}, noop_receive, noop_send)
    assert calls == ["lifespan"]
