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

A single shared secret over plain HTTP: no TLS, no per-client keys, no rotation
or replay protection. Deploy behind a firewall or VPN. See docs/configuration.md.
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

    L1: ``compare_digest`` requires its ``str`` arguments to be ASCII-only —
    it raises ``TypeError`` otherwise — but ``candidate`` comes straight from
    a caller-supplied HTTP header, which can legally carry non-ASCII bytes
    (ASGI header values are latin-1-decoded, so any byte sequence decodes
    without error but can land outside the ASCII range). An unhandled
    ``TypeError`` here previously surfaced as an unstyled 500 instead of the
    intended 401. Comparing UTF-8-encoded ``bytes`` instead sidesteps the
    ASCII restriction entirely (bytes-like objects have none) while still
    comparing every candidate byte-for-byte against every configured key.
    """
    # UTF-8 can encode any Unicode code point (including the 0x00-0xFF range
    # ASGI's latin-1 header decoding produces), so this never raises.
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

        # L2: a CORS preflight (`OPTIONS`) never carries the caller's real
        # credentials — browsers deliberately send it bare — so gating it
        # behind the API key just makes it fail with 401 before
        # ``CORSMiddleware`` ever runs, which the browser then reports as a
        # CORS failure on the REAL request (its console error points at CORS,
        # not auth, which is exactly why this was easy to miss). This bypass
        # is independent of middleware registration order — see
        # ``main.py``'s comment on why `CORSMiddleware` is also registered to
        # wrap this middleware, not the other way around, but this makes the
        # preflight bypass correct even if a future refactor gets the order
        # wrong again. The actual (non-OPTIONS) request that follows a
        # preflight is still fully authenticated below, same as any other
        # method.
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
