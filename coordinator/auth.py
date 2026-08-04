"""Minimal opt-in API-key auth for the coordinator's HTTP surface.

Keys come from ``COORDINATOR_API_KEYS`` (comma-separated), read live from
the environment rather than ``Settings``. Empty/unset ⇒ no-op (every route
open). Set ⇒ every route needs ``Authorization: Bearer <key>`` or
``x-api-key: <key>``, except ``/health`` and ``/metrics``. See
docs/configuration.md.
"""

from __future__ import annotations

import os
import secrets
from typing import FrozenSet, Optional

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

#: Paths that stay reachable with no API key, even when auth is enabled.
_EXEMPT_PATHS: FrozenSet[str] = frozenset({"/health", "/metrics"})

_UNAUTHORIZED_BODY = {
    "error": {"message": "invalid or missing API key", "type": "authentication_error"}
}


def load_api_keys() -> FrozenSet[str]:
    """Parse ``COORDINATOR_API_KEYS`` from the environment.

    Re-read on every call (not cached) so a running process picks up a
    changed env without a restart.
    """
    raw = os.environ.get("COORDINATOR_API_KEYS", "")
    return frozenset(key.strip() for key in raw.split(",") if key.strip())


def _is_exempt(path: str) -> bool:
    """Whether ``path`` is reachable with no key (health/metrics probes)."""
    return path in _EXEMPT_PATHS or path.startswith("/metrics/")


def _extract_candidate_key(headers: Headers) -> Optional[str]:
    """Pull a caller-supplied key from ``Authorization: Bearer`` or ``x-api-key``."""
    authorization = headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token
    api_key = headers.get("x-api-key")
    if api_key:
        return api_key
    return None


def _matches_any(candidate: str, valid_keys: FrozenSet[str]) -> bool:
    """Constant-time membership check against the configured key set.

    Compares UTF-8 bytes, not ``str``: ``compare_digest`` requires ASCII-only
    ``str`` args and header values can be non-ASCII, which would otherwise
    raise ``TypeError`` (a 500) instead of yielding a 401.
    """
    candidate_bytes = candidate.encode("utf-8")
    matched = False
    for key in valid_keys:
        if secrets.compare_digest(candidate_bytes, key.encode("utf-8")):
            matched = True
    return matched


class APIKeyAuthMiddleware:
    """Pure-ASGI middleware gating HTTP routes behind ``COORDINATOR_API_KEYS``.

    Registered directly on the FastAPI app in ``coordinator/main.py`` (via
    ``app.add_middleware``) so it covers every router — including ones other
    modules mount on ``app`` — without each route needing its own dependency,
    and without buffering/altering streamed (SSE) responses.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        valid_keys = load_api_keys()
        if not valid_keys:
            await self.app(scope, receive, send)
            return

        if _is_exempt(scope["path"]):
            await self.app(scope, receive, send)
            return

        # A CORS preflight (OPTIONS) never carries real credentials, so
        # gating it would 401 before CORSMiddleware runs and the browser
        # would misreport it as a CORS failure. The real request that
        # follows is still authenticated below.
        if scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        candidate = _extract_candidate_key(headers)
        if candidate is None or not _matches_any(candidate, valid_keys):
            response = JSONResponse(
                _UNAUTHORIZED_BODY,
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
