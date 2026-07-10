"""Minimal opt-in API-key auth for the coordinator's HTTP surface.

Keys come from ``COORDINATOR_API_KEYS`` (comma-separated, whitespace-trimmed,
empty entries dropped), read directly from the process environment at
request time. This is deliberately NOT a ``coordinator.config.Settings``
field: ``coordinator/tests/test_config.py::test_dead_settings_fields_removed``
guards against unwired ``Settings`` fields (it already asserts ``api_keys``
and ``enable_auth`` are gone), and this env var is wired here instead, kept
env-only until Plan 15 Phase B formalizes config.

Behavior:
- Env var unset or empty ⇒ this middleware is a complete no-op — every route
  is open, matching the coordinator's current (pre-auth) default.
- Env var set ⇒ every HTTP route requires a valid key via
  ``Authorization: Bearer <key>`` or ``x-api-key: <key>``, except the
  liveness (``/health``) and Prometheus (``/metrics``) endpoints, which stay
  reachable for infra probes/scrapers that don't carry credentials.

LAN-trust note: this is a single shared-secret gate suitable for a trusted
LAN deployment behind a firewall/VPN. It does not provide TLS, per-client or
per-job keys, rotation, or replay protection, and the key travels in plain
HTTP headers. See ``pending-work/15-*.md`` ("Plan 15") Phase B for TLS
termination and per-job keys; this module implements Phase A only (folded
into Plan 13 Task 3).
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

    Re-reads the environment on every call instead of caching: parsing is a
    single cheap ``split(",")`` over a short string, and reading live means
    tests can flip the env var with ``monkeypatch.setenv``/``delenv`` with no
    cache to reset, and a running process picks up a changed env without a
    restart.
    """
    raw = os.environ.get("COORDINATOR_API_KEYS", "")
    return frozenset(key.strip() for key in raw.split(",") if key.strip())


def _is_exempt(path: str) -> bool:
    """Whether ``path`` is reachable with no key (health/metrics probes).

    ``/metrics`` is mounted as a sub-app (``app.mount("/metrics", ...)`` in
    main.py), so also exempt anything nested under it.
    """
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

    Uses ``secrets.compare_digest`` (not ``==``/``in``) for each comparison
    to avoid leaking key contents via a timing side channel, and checks
    every key rather than returning on the first match so the match's
    position in the set doesn't leak either.
    """
    matched = False
    for key in valid_keys:
        if secrets.compare_digest(candidate, key):
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
