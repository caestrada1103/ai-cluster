"""Transparent HTTP proxy to worker-local llama-server instances (Plan 13 Task 2).

Models whose registry ``engine == "llamaserver"`` are served by a ``llama-server``
child process running next to a worker (supervised over the existing gRPC load
path — see ``pending-work/13-agentic-serving-llama-server.md``). Rather than
model every OpenAI/Anthropic field in pydantic, the coordinator forwards the RAW
request bytes to that llama-server and streams the response back verbatim, so
``tools``, ``tool_choice``, streaming ``tool_calls``, Anthropic ``thinking``
blocks and any future field flow through unmodified and stay forward-compatible.

``proxy_request()`` is the single entry point: it returns a
``BufferedProxyResponse`` for ordinary JSON replies (and for synthesized 502
connect errors) and a ``StreamingProxyResponse`` — a live async byte iterator
plus the upstream status/headers the caller needs to build a
``fastapi.responses.StreamingResponse`` — for Server-Sent Events, which it never
buffers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Mapping, Union

import httpx

logger = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 7230 section 6.1) must never cross a proxy boundary;
# ``host`` and ``content-length`` are recomputed by httpx (request) / Starlette
# (response).
_HOP_BY_HOP: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


@dataclass
class BufferedProxyResponse:
    """A fully-read upstream reply (ordinary JSON, or a synthesized 502 error)."""

    status_code: int
    headers: Dict[str, str]
    content: bytes


@dataclass
class StreamingProxyResponse:
    """An upstream SSE reply: the upstream status/headers plus a live byte
    iterator that yields chunks as llama-server produces them (never buffered)."""

    status_code: int
    headers: Dict[str, str]
    body: AsyncIterator[bytes]


ProxyResponse = Union[BufferedProxyResponse, StreamingProxyResponse]


def _request_timeout(stream: bool) -> httpx.Timeout:
    """Reuse the coordinator's existing request-timeout setting for the proxy.

    Read at call time (per the Plan 13 contract) so a redeploy that changes
    ``COORDINATOR_REQUEST_TIMEOUT`` takes effect without touching this module.
    Streaming replies disable the read timeout: an agentic tool call may idle
    between SSE tokens far longer than a single unary request would.
    """
    from coordinator.config import Settings

    timeout = float(Settings().request_timeout)
    read = None if stream else timeout
    return httpx.Timeout(timeout, read=read)


def filter_request_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """Build the forward set of REQUEST headers for the upstream llama-server.

    Forwards Content-Type (so llama-server parses the JSON body) and every other
    client header except hop-by-hop ones, ``host`` and ``content-length`` (httpx
    re-derives those for the upstream connection).
    """
    dropped = _HOP_BY_HOP | {"host", "content-length"}
    return {k: v for k, v in headers.items() if k.lower() not in dropped}


def _response_headers(headers: httpx.Headers, *, drop_encoding: bool) -> Dict[str, str]:
    """Sanitize upstream RESPONSE headers before handing them back to the client.

    Always drops hop-by-hop headers and ``content-length`` (recomputed by
    Starlette, or re-framed for the stream). Buffered replies also drop
    ``content-encoding`` because httpx has already decoded ``.content``; the
    streaming path keeps it and forwards the raw (still-encoded) bytes verbatim.
    """
    dropped = set(_HOP_BY_HOP) | {"content-length"}
    if drop_encoding:
        dropped.add("content-encoding")
    return {k: v for k, v in headers.items() if k.lower() not in dropped}


def _connect_error(url: str, exc: Exception) -> BufferedProxyResponse:
    """Render an upstream connect/transport failure as a 502 JSON body."""
    logger.warning("llama-server proxy to %s failed: %s", url, exc)
    body = json.dumps(
        {
            "error": {
                "message": f"llama-server upstream unreachable: {exc}",
                "type": "upstream_unavailable",
                "code": 502,
            }
        }
    ).encode()
    return BufferedProxyResponse(
        status_code=502,
        headers={"content-type": "application/json"},
        content=body,
    )


async def _iter_stream(response: httpx.Response, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    """Yield raw upstream bytes as they arrive, then release response + client."""
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()
        await client.aclose()


async def _proxy_streaming(
    method: str, url: str, body: bytes, headers: Dict[str, str], timeout: httpx.Timeout
) -> ProxyResponse:
    """Open the upstream stream and hand back an iterator (no buffering)."""
    client = httpx.AsyncClient(timeout=timeout)
    try:
        request = client.build_request(method, url, content=body, headers=headers)
        response = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        return _connect_error(url, exc)
    return StreamingProxyResponse(
        status_code=response.status_code,
        headers=_response_headers(response.headers, drop_encoding=False),
        body=_iter_stream(response, client),
    )


async def _proxy_buffered(
    method: str, url: str, body: bytes, headers: Dict[str, str], timeout: httpx.Timeout
) -> ProxyResponse:
    """Read the whole upstream reply into memory (ordinary JSON responses)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, content=body, headers=headers)
            status = response.status_code
            content = response.content
            resp_headers = _response_headers(response.headers, drop_encoding=True)
    except httpx.HTTPError as exc:
        return _connect_error(url, exc)
    return BufferedProxyResponse(status_code=status, headers=resp_headers, content=content)


async def proxy_request(
    method: str,
    url: str,
    body: bytes,
    headers: Dict[str, str],
    stream: bool,
) -> ProxyResponse:
    """Forward one request to a worker-local llama-server and return its reply.

    Args:
        method: HTTP method of the incoming request (e.g. ``"POST"``).
        url: Fully-qualified upstream URL (``http://<worker_host>:<port><path>``).
        body: RAW request bytes, forwarded byte-for-byte (never re-serialized —
            this is what preserves ``tools`` and other unknown fields).
        headers: Pre-filtered request headers (see :func:`filter_request_headers`).
        stream: When True the reply is streamed as SSE (never buffered);
            otherwise it is read fully into a :class:`BufferedProxyResponse`.

    Returns a :class:`StreamingProxyResponse` when ``stream`` is True and the
    upstream connected, else a :class:`BufferedProxyResponse` (including a
    synthesized 502 on any connect/transport error, on either path). Upstream
    status codes are preserved.
    """
    timeout = _request_timeout(stream)
    if stream:
        return await _proxy_streaming(method, url, body, headers, timeout)
    return await _proxy_buffered(method, url, body, headers, timeout)


__all__ = [
    "BufferedProxyResponse",
    "StreamingProxyResponse",
    "ProxyResponse",
    "proxy_request",
    "filter_request_headers",
]
