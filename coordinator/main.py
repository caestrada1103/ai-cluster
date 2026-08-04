"""FastAPI application entry point for AI cluster coordinator."""

import ipaddress
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from coordinator.api import router as api_router
from coordinator.auth import APIKeyAuthMiddleware, load_api_keys
from coordinator.body_limit import BodySizeLimitMiddleware
from coordinator.config import Settings
from coordinator.coordinator import ClusterCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global coordinator instance
coordinator: Optional[ClusterCoordinator] = None

#: Hostnames treated as loopback in addition to whatever `ipaddress` itself
#: recognizes (127.0.0.0/8, ::1). "localhost" is not an IP literal so
#: `ipaddress.ip_address` can't classify it.
_LOOPBACK_HOSTNAMES = frozenset({"localhost"})


def _is_loopback_host(host: str) -> bool:
    """Best-effort loopback check for `Settings.host` (C4).

    Conservative on purpose: anything that isn't recognizably loopback (an
    unparseable value, a bare LAN hostname, `0.0.0.0`/`::`) is treated as
    NOT loopback — i.e. this fails CLOSED, requiring `COORDINATOR_API_KEYS`,
    rather than accidentally waving through something that isn't actually
    loopback-only.
    """
    if host in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _refuse_insecure_bind(settings: Settings) -> None:
    """C4 — the core of the prod-by-default posture.

    Refuse to start when `COORDINATOR_HOST` is not loopback-only AND no
    `COORDINATOR_API_KEYS` are configured: that combination means every
    route (`/v1/models/load`, `/v1/workers/manual`, `/v1/chat/completions`,
    ...) is reachable, unauthenticated, to anyone who can route to this
    host. Raises rather than logging-and-continuing so a misconfiguration
    can never silently ship — the process exits immediately (uvicorn/Docker
    surfaces this as a crash-loop with the message below, not a
    slow-to-notice open API).

    This is intentionally a BREAKING change for anyone previously running
    `--host 0.0.0.0` with no keys (the AGENTS.md-documented dev command is
    one such case) — see .env.example / docs/deployment.md for the two
    opt-ins: bind loopback, or set COORDINATOR_API_KEYS.
    """
    if _is_loopback_host(settings.host):
        return
    if load_api_keys():
        return
    raise RuntimeError(
        f"Refusing to start: COORDINATOR_HOST='{settings.host}' is not "
        "loopback-only and COORDINATOR_API_KEYS is unset. Binding a "
        "non-loopback address with no auth means every route (models/load, "
        "workers/manual, chat/completions, ...) is reachable to anyone who "
        "can route to this host with zero credentials. Fix one of: "
        "(1) set COORDINATOR_HOST=127.0.0.1 for a loopback-only/single-host "
        "setup, or (2) set COORDINATOR_API_KEYS to a comma-separated list of "
        "secrets before binding beyond loopback. See .env.example and "
        "docs/deployment.md."
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    global coordinator

    # Load settings
    settings = Settings()
    logger.info(f"Starting coordinator with settings: {settings}")
    _refuse_insecure_bind(settings)

    # Initialize coordinator
    coordinator = ClusterCoordinator(settings)
    await coordinator.start()

    # Store in app state
    app.state.coordinator = coordinator
    app.state.settings = settings

    logger.info("Coordinator started successfully")

    yield

    # Shutdown
    logger.info("Shutting down coordinator...")
    await coordinator.stop()
    logger.info("Coordinator stopped")


# Create FastAPI app
app = FastAPI(
    title="AI Cluster Coordinator",
    description="Distributed AI inference cluster API",
    version="0.1.0",
    lifespan=lifespan,
)

# Opt-in API-key auth (Plan 15 Phase A / Plan 13 Task 3). No-op unless
# COORDINATOR_API_KEYS is set; see coordinator/auth.py for the contract.
# Covers every router mounted below, including /metrics and routes other
# modules add to api_router, except the /health and /metrics paths it
# exempts itself.
#
# L2: registered BEFORE CORSMiddleware on purpose. Starlette's
# `add_middleware` prepends to the middleware stack, so the LAST-added
# middleware ends up OUTERMOST (it sees the request first). Adding auth
# first here means CORSMiddleware ends up outermost — a cross-origin
# preflight (`OPTIONS`) reaches CORSMiddleware before this one, so it gets a
# proper CORS-headers response instead of being 401'd by auth before CORS
# ever runs (which the browser then misreports as a CORS failure on the real
# request, not an auth failure on the preflight). `APIKeyAuthMiddleware`
# also bypasses `OPTIONS` itself (see auth.py) — belt-and-suspenders so this
# stays correct even if the order above is ever accidentally swapped.
app.add_middleware(APIKeyAuthMiddleware)

# Add CORS middleware.
# M3: secure by default. `COORDINATOR_CORS_ORIGINS` now defaults to `[]`
# (see config.py), and a loopback `allow_origin_regex` is ALWAYS applied
# (unless the operator opts fully into "*") so a browser page served from
# the coordinator's own host (http(s)://localhost or 127.0.0.1, any port)
# keeps working out of the box for local development — the permissive case
# stays opt-in, not silently the default, for any other origin.
# When allow_origins contains "*", allow_credentials must be False —
# browsers reject the combination and it's a security anti-pattern.
_cors_origins = Settings().cors_origins
_cors_wildcard = _cors_origins == ["*"]
_LOOPBACK_CORS_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=None if _cors_wildcard else _LOOPBACK_CORS_REGEX,
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

# H5: request-body size cap, registered LAST so it ends up OUTERMOST (same
# "last add_middleware wins the outer position" mechanics documented above)
# — reject an oversized body before spending any cycles on CORS/auth. Covers
# every route, including the engine="llamaserver" raw-body proxy path, which
# has no max_tokens/queue admission control of its own.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=Settings().max_request_body_bytes)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include API routes
app.include_router(api_router, prefix="/v1")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    if coordinator and coordinator.is_running:
        return {"status": "healthy", "workers": len(coordinator.workers)}
    return {"status": "starting"}


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": "AI Cluster Coordinator",
        "version": "0.1.0",
        "endpoints": [
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/models",
            "/v1/models/load",
            "/v1/models/{name} (DELETE)",
            "/v1/workers",
            "/v1/workers/manual",
            "/health",
            "/metrics",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    _settings = Settings()
    uvicorn.run(app, host=_settings.host, port=_settings.port)
