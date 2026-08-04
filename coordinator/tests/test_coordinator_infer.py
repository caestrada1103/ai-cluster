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
async def test_unload_model_calls_worker_stub() -> None:
    from unittest.mock import AsyncMock, MagicMock

    import coordinator.proto.cluster_pb2 as pb
    from coordinator.coordinator import WorkerInfo, WorkerState

    settings = make_settings()
    coord = ClusterCoordinator(settings)
    worker = WorkerInfo(id="w1", address="a:1", channel=AsyncMock(), stub=AsyncMock())
    worker.state = WorkerState.HEALTHY
    worker.loaded_models = {"deepseek-7b": MagicMock()}
    worker.stub.UnloadModel = AsyncMock(return_value=pb.Empty())
    coord.workers["w1"] = worker

    unloaded_from = await coord.unload_model("deepseek-7b")
    assert unloaded_from == ["w1"]
    worker.stub.UnloadModel.assert_awaited_once()
    assert "deepseek-7b" not in worker.loaded_models


@pytest.mark.asyncio
async def test_unload_model_not_loaded_raises() -> None:
    settings = make_settings()
    coord = ClusterCoordinator(settings)
    with pytest.raises(KeyError):
        await coord.unload_model("ghost-model")


@pytest.mark.asyncio
async def test_load_model_refreshes_worker_state_synchronously() -> None:
    """POST /models/load must not leave loaded_models/gpus stale until the next
    health-check tick (item 3): `_load_model_on_worker` calls GetStatus itself
    right after a successful LoadModel RPC, so the caller's very next
    GET /v1/workers | /v1/models read is already accurate."""
    from unittest.mock import AsyncMock

    import coordinator.proto.cluster_pb2 as pb
    from coordinator.coordinator import WorkerInfo, WorkerState

    settings = make_settings()
    coord = ClusterCoordinator(settings)
    worker = WorkerInfo(id="w1", address="a:1", channel=AsyncMock(), stub=AsyncMock())
    worker.state = WorkerState.HEALTHY
    worker.gpus = [pb.GPUInfo(id=0, name="gpu0", total_memory=10_000, available_memory=10_000)]

    worker.stub.LoadModel = AsyncMock(
        return_value=pb.LoadModelResponse(success=True, memory_used=1000)
    )
    post_load_status = pb.WorkerStatus(
        worker_id="w1",
        gpus=[pb.GPUInfo(id=0, name="gpu0", total_memory=10_000, available_memory=9_000)],
        loaded_models=[pb.LoadedModelInfo(model_name="deepseek-7b", memory_used=1000)],
    )
    worker.stub.GetStatus = AsyncMock(return_value=post_load_status)

    success = await coord._load_model_on_worker(worker, "deepseek-7b")

    assert success is True
    worker.stub.GetStatus.assert_awaited_once()
    assert "deepseek-7b" in worker.loaded_models  # reflected immediately
    assert worker.gpus[0].available_memory == 9_000  # reserved memory not stale


@pytest.mark.asyncio
async def test_unload_model_refreshes_worker_state_synchronously() -> None:
    """DELETE /models/{name} must not leave GPU memory figures stale (item 3):
    `unload_model` calls GetStatus itself right after a successful
    UnloadModel RPC."""
    from unittest.mock import AsyncMock, MagicMock

    import coordinator.proto.cluster_pb2 as pb
    from coordinator.coordinator import WorkerInfo, WorkerState

    settings = make_settings()
    coord = ClusterCoordinator(settings)
    worker = WorkerInfo(id="w1", address="a:1", channel=AsyncMock(), stub=AsyncMock())
    worker.state = WorkerState.HEALTHY
    worker.loaded_models = {"deepseek-7b": MagicMock()}
    worker.gpus = [pb.GPUInfo(id=0, name="gpu0", total_memory=10_000, available_memory=1_000)]
    coord.workers["w1"] = worker

    worker.stub.UnloadModel = AsyncMock(return_value=pb.Empty())
    post_unload_status = pb.WorkerStatus(
        worker_id="w1",
        gpus=[pb.GPUInfo(id=0, name="gpu0", total_memory=10_000, available_memory=10_000)],
        loaded_models=[],
    )
    worker.stub.GetStatus = AsyncMock(return_value=post_unload_status)

    unloaded_from = await coord.unload_model("deepseek-7b")

    assert unloaded_from == ["w1"]
    worker.stub.GetStatus.assert_awaited_once()
    assert worker.gpus[0].available_memory == 10_000  # freed immediately, no health-tick wait


@pytest.mark.asyncio
async def test_connect_worker_adopts_resident_loaded_models(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """A freshly-(re)started coordinator must adopt models already resident on a
    worker that kept running across the restart.

    Regression for a bug where ``_connect_worker`` copied ``status.gpus`` onto
    the new ``WorkerInfo`` but never ``status.loaded_models`` (unlike
    ``WorkerInfo.get_status()``, which does). That left a resident model
    unreachable via unload/list immediately after coordinator startup, until
    the periodic health-check loop happened to catch up.
    """
    from unittest.mock import AsyncMock, MagicMock

    import grpc

    import coordinator.proto.cluster_pb2 as pb
    import coordinator.proto.cluster_pb2_grpc as pb_grpc

    settings = make_settings()
    coord = ClusterCoordinator(settings)

    fake_status = pb.WorkerStatus(
        worker_id="w1",
        gpus=[],
        loaded_models=[pb.LoadedModelInfo(model_name="deepseek-7b", memory_used=123)],
    )
    fake_stub = MagicMock()
    fake_stub.GetStatus = AsyncMock(return_value=fake_status)
    fake_channel = MagicMock()
    fake_channel.close = AsyncMock()

    monkeypatch.setattr(grpc.aio, "insecure_channel", lambda *a, **k: fake_channel)
    monkeypatch.setattr(pb_grpc, "WorkerStub", lambda channel: fake_stub)

    worker = await coord._connect_worker("127.0.0.1:50051")
    assert worker is not None
    assert "deepseek-7b" in worker.loaded_models

    # The whole point of the fix: unload must find the model immediately,
    # without waiting on a health-check poll to populate loaded_models.
    fake_stub.UnloadModel = AsyncMock(return_value=pb.Empty())
    unloaded_from = await coord.unload_model("deepseek-7b")
    assert unloaded_from == ["w1"]


@pytest.mark.asyncio
async def test_load_model_on_worker_rejects_unregistered_model_by_default() -> None:
    """H4: an unregistered model_name must never reach the worker as an
    implicit HuggingFace-repo pull unless explicitly opted into."""
    from unittest.mock import AsyncMock, MagicMock

    from coordinator.coordinator import WorkerInfo, WorkerState

    settings = make_settings()
    assert settings.allow_unregistered_model_pull is False
    coord = ClusterCoordinator(settings)
    worker = WorkerInfo(id="w1", address="127.0.0.1:1", channel=AsyncMock(), stub=MagicMock())
    worker.state = WorkerState.HEALTHY
    worker.stub.LoadModel = AsyncMock()

    with pytest.raises(ValueError, match="not in the registry"):
        await coord._load_model_on_worker(worker, "totally-unregistered-model-xyz")

    # The critical assertion: the worker was never even asked to load it.
    worker.stub.LoadModel.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_model_on_worker_allows_unregistered_when_opted_in() -> None:
    """The opt-in (COORDINATOR_ALLOW_UNREGISTERED_MODEL_PULL) restores the
    original ad hoc HF-pull behavior."""
    from unittest.mock import AsyncMock, MagicMock

    import coordinator.proto.cluster_pb2 as pb
    from coordinator.coordinator import WorkerInfo, WorkerState

    settings = make_settings(allow_unregistered_model_pull=True)
    coord = ClusterCoordinator(settings)
    worker = WorkerInfo(id="w1", address="127.0.0.1:1", channel=AsyncMock(), stub=MagicMock())
    worker.state = WorkerState.HEALTHY
    worker.stub.LoadModel = AsyncMock(
        return_value=pb.LoadModelResponse(success=True, memory_used=1)
    )
    worker.stub.GetStatus = AsyncMock(
        return_value=pb.WorkerStatus(worker_id="w1", gpus=[], loaded_models=[])
    )

    ok = await coord._load_model_on_worker(worker, "some-org/some-arbitrary-repo")
    assert ok is True
    worker.stub.LoadModel.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_model_on_worker_allows_registered_model_by_default() -> None:
    """Registered models (config/models.toml) are unaffected by H4's gate."""
    from unittest.mock import AsyncMock, MagicMock

    import coordinator.proto.cluster_pb2 as pb
    from coordinator.coordinator import WorkerInfo, WorkerState

    settings = make_settings()
    coord = ClusterCoordinator(settings)
    worker = WorkerInfo(id="w1", address="127.0.0.1:1", channel=AsyncMock(), stub=MagicMock())
    worker.state = WorkerState.HEALTHY
    worker.stub.LoadModel = AsyncMock(
        return_value=pb.LoadModelResponse(success=True, memory_used=1)
    )
    worker.stub.GetStatus = AsyncMock(
        return_value=pb.WorkerStatus(worker_id="w1", gpus=[], loaded_models=[])
    )

    ok = await coord._load_model_on_worker(worker, "deepseek-7b")
    assert ok is True
    worker.stub.LoadModel.assert_awaited_once()


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
