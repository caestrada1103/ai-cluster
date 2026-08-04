"""ASGI middleware enforcing a maximum HTTP request body size (H5).

Nothing in the coordinator's HTTP surface previously capped request body
size: ``api.py``'s ``_read_json_body`` calls ``await request.body()``, which
buffers the ENTIRE body into memory with no limit of its own. A client (or a
compromised/rogue registered worker feeding back through some future path)
sending an arbitrarily large body — buffered or chunked-transfer-encoded, no
``Content-Length`` required — can exhaust coordinator memory. This also
covers the ``engine="llamaserver"`` proxy path (``_proxy_to_llamaserver``),
which forwards the raw body straight through and has no ``max_tokens``/queue
admission control of its own to fall back on.

Two layers, both before any route handler (or ``request.body()``) ever runs:
  1. ``Content-Length`` header check — rejects the common case immediately,
     before a single body byte is read.
  2. A streaming byte counter wrapping ``receive()`` — catches a chunked
     body with no (or a lying) ``Content-Length`` too.
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _BodyTooLarge(Exception):
    """Internal sentinel — never escapes this module."""


def _too_large_response(max_bytes: int) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": f"request body exceeds the {max_bytes}-byte limit",
                "type": "request_too_large",
            }
        },
        status_code=413,
    )


class BodySizeLimitMiddleware:
    """Reject HTTP request bodies larger than ``max_bytes``."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = None  # malformed value — fall through to the streaming check
            if declared is not None and declared > self.max_bytes:
                await _too_large_response(self.max_bytes)(scope, receive, send)
                return

        seen = 0

        async def limited_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            # Reading the body is virtually always the first thing a route
            # does, so a response is not yet in flight in the overwhelming
            # common case — but guard against sending a second
            # `http.response.start` if some handler ever changes that.
            if not response_started:
                await _too_large_response(self.max_bytes)(scope, receive, send)
