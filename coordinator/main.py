"""FastAPI application entry point for AI cluster coordinator."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from coordinator.api import router as api_router
from coordinator.config import Settings
from coordinator.coordinator import ClusterCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global coordinator instance
coordinator: ClusterCoordinator = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifecycle."""
    global coordinator
    
    # Load settings
    settings = Settings()
    logger.info(f"Starting coordinator with settings: {settings}")
    
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

# Add CORS middleware.
# When allow_origins contains "*", allow_credentials must be False —
# browsers reject the combination and it's a security anti-pattern.
_cors_origins = Settings().cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include API routes
app.include_router(api_router, prefix="/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if coordinator and coordinator.is_running:
        return {"status": "healthy", "workers": len(coordinator.workers)}
    return {"status": "starting"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Cluster Coordinator",
        "version": "0.1.0",
        "endpoints": [
            "/v1/completions",
            "/v1/models",
            "/v1/models/load",
            "/v1/workers",
            "/health",
            "/metrics",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)