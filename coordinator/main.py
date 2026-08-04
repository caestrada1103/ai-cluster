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
    """Best-effort loopback check for `Settings.host`.

    Fails closed: anything not recognizably loopback (unparseable, a LAN
    hostname, `0.0.0.0`/`::`) counts as NOT loopback.
    """
    if host in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _refuse_insecure_bind(settings: Settings) -> None:
    """Refuse to start on a non-loopback host with no API keys configured.

    Raises instead of logging-and-continuing so an insecure bind can never
    silently ship. See docs/deployment.md.
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

# Opt-in API-key auth; no-op unless COORDINATOR_API_KEYS is set (see auth.py).
# Registered before CORSMiddleware so CORS ends up outermost — a preflight
# OPTIONS gets proper CORS headers instead of a 401 from auth. See
# docs/deployment.md.
app.add_middleware(APIKeyAuthMiddleware)

# CORS: defaults to loopback-only (plus explicit COORDINATOR_CORS_ORIGINS).
# allow_credentials must be False when allow_origins is "*".
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

# Request-body size cap, registered last so it ends up outermost — rejects
# an oversized body before CORS/auth run.
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
