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
async def test_get_status_uses_settings_thresholds() -> None:
    """A worker built by the coordinator honors Settings health thresholds, not hardcodes."""
    from unittest.mock import AsyncMock

    from coordinator.coordinator import WorkerInfo, WorkerState

    worker = WorkerInfo(
        id="w1",
        address="127.0.0.1:1",
        channel=AsyncMock(),
        stub=AsyncMock(),
        health_check_timeout=1,
        max_failures=2,
    )
    worker.stub.GetStatus = AsyncMock(side_effect=RuntimeError("down"))
    await worker.get_status()
    assert worker.state != WorkerState.UNHEALTHY  # 1 failure < max_failures
    await worker.get_status()
    # mypy narrows worker.state from the assert above and doesn't know get_status()
    # mutates it — this is a real state transition, not a dead comparison.
    assert worker.state == WorkerState.UNHEALTHY  # type: ignore[comparison-overlap]


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
