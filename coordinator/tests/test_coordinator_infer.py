"""Tests for ClusterCoordinator.infer() error paths (no gRPC needed)."""

import asyncio

import pytest

from coordinator.coordinator import ClusterCoordinator
from coordinator.tests.conftest import make_settings


@pytest.mark.asyncio
async def test_target_worker_unavailable_fails_fast() -> None:
    """Targeting a nonexistent worker must raise promptly, not burn request_timeout."""
    settings = make_settings(request_timeout=30)
    coord = ClusterCoordinator(settings)
    coord.is_running = True
    processor = asyncio.create_task(coord._request_processor())
    try:
        with pytest.raises(RuntimeError, match="not available"):
            await asyncio.wait_for(
                coord.infer("deepseek-7b", "hi", worker_id="ghost-worker"),
                timeout=5,  # far below request_timeout=30 — proves fail-fast
            )
    finally:
        coord.is_running = False
        processor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await processor


@pytest.mark.asyncio
async def test_infer_pops_context_on_error() -> None:
    settings = make_settings(request_timeout=30)
    coord = ClusterCoordinator(settings)
    coord.is_running = True
    processor = asyncio.create_task(coord._request_processor())
    try:
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(
                coord.infer("deepseek-7b", "hi", worker_id="ghost-worker"), timeout=5
            )
        assert coord.active_requests == {}
    finally:
        coord.is_running = False
        processor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await processor
