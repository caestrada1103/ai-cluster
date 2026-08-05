"""Coordinator-side orchestration for distributed (cross-node ggml-RPC) model
loads: peers load before the lead, unload runs in reverse, a failed peer or
lead rolls back whatever already succeeded, and inference pins to the lead.

No real gRPC/network — a fake stub records call order and cooperates so
`WorkerInfo.loaded_models` reflects real load state, mirroring the fake-stub
pattern in test_llamacpp_transport.py.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Set
from unittest.mock import AsyncMock

import pytest

import coordinator.proto.cluster_pb2 as pb
from coordinator.coordinator import (
    ClusterCoordinator,
    RequestContext,
    WorkerInfo,
    WorkerState,
    _address_host,
    _largest_remainder_split,
)
from coordinator.models import ModelRegistry
from coordinator.tests.conftest import make_settings

_DISTRIBUTED_REGISTRY = {
    "models": {
        "distributed-orch-test": {
            "family": "qwen",
            "parameters": "32B",
            "min_memory_gb": 20,
            "recommended_gpus": 1,
            "max_gpus": 1,
            "engine": "llamacpp",
            "gguf": {
                "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
                "file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
            },
            "distributed": {
                "enabled": True,
                "lead": "lead-1",
                "peers": ["peer-1", "peer-2"],
                "rpc_port": 50151,
            },
        }
    }
}


class _FakeStub:
    """Records LoadModel/UnloadModel call order; cooperates with GetStatus so
    `WorkerInfo.loaded_models` genuinely tracks load state across calls."""

    def __init__(self, worker_id: str, log: List[Any], fail_load: bool = False) -> None:
        self.worker_id = worker_id
        self.log = log
        self.fail_load = fail_load
        self.last_metadata: Dict[str, str] = {}
        self._loaded: Set[str] = set()
        # Echoed back verbatim by GetStatus -- refreshing a worker's state
        # after a load/unload must not wipe out its GPU inventory.
        self.gpus: List[pb.GPUInfo] = []

    async def LoadModel(self, request: Any, timeout: Any = None) -> Any:
        self.log.append(("load", self.worker_id))
        self.last_metadata = dict(request.config.metadata)
        if self.fail_load:
            return SimpleNamespace(success=False, memory_used=0, message="boom")
        self._loaded.add(request.model_name)
        return SimpleNamespace(success=True, memory_used=1_000_000, message="ok")

    async def UnloadModel(self, request: Any, timeout: Any = None) -> Any:
        self.log.append(("unload", self.worker_id))
        self._loaded.discard(request.model_name)
        return pb.Empty()

    async def GetStatus(self, request: Any, timeout: Any = None) -> Any:
        return pb.WorkerStatus(
            worker_id=self.worker_id,
            gpus=self.gpus,
            loaded_models=[pb.LoadedModelInfo(model_name=n) for n in self._loaded],
        )


def _make_worker(
    worker_id: str,
    address: str,
    log: List[Any],
    total_memory: List[int],
    fail_load: bool = False,
) -> WorkerInfo:
    stub = _FakeStub(worker_id, log, fail_load=fail_load)
    worker = WorkerInfo(id=worker_id, address=address, channel=AsyncMock(), stub=stub)
    worker.state = WorkerState.HEALTHY
    worker.gpus = [
        pb.GPUInfo(id=i, name=f"gpu{i}", total_memory=mem, available_memory=mem)
        for i, mem in enumerate(total_memory)
    ]
    stub.gpus = worker.gpus
    return worker


def _cluster(log: List[Any], fail: str = "") -> ClusterCoordinator:
    """A coordinator with lead-1 (40 GB) + peer-1/peer-2 (20 GB each)
    registered; `fail` names a worker_id whose LoadModel should report
    failure."""
    ModelRegistry.load_from_dict(_DISTRIBUTED_REGISTRY)
    coord = ClusterCoordinator(make_settings())
    coord.workers["lead-1"] = _make_worker(
        "lead-1", "10.0.0.1:50051", log, [40_000_000_000], fail_load=(fail == "lead-1")
    )
    coord.workers["peer-1"] = _make_worker(
        "peer-1", "10.0.0.2:50051", log, [20_000_000_000], fail_load=(fail == "peer-1")
    )
    coord.workers["peer-2"] = _make_worker(
        "peer-2", "10.0.0.3:50051", log, [20_000_000_000], fail_load=(fail == "peer-2")
    )
    return coord


# ---------------------------------------------------------------------------
# _load_distributed_model: ordering, metadata, rollback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_distributed_model_loads_peers_before_lead() -> None:
    log: List[Any] = []
    coord = _cluster(log)
    model_config = ModelRegistry.get_model("distributed-orch-test")
    assert model_config is not None

    ok = await coord._load_distributed_model(model_config)

    assert ok is True
    assert log == [("load", "peer-1"), ("load", "peer-2"), ("load", "lead-1")]
    assert "distributed-orch-test" in coord.workers["lead-1"].loaded_models


@pytest.mark.asyncio
async def test_load_distributed_model_sends_role_specific_metadata() -> None:
    log: List[Any] = []
    coord = _cluster(log)
    model_config = ModelRegistry.get_model("distributed-orch-test")
    assert model_config is not None

    await coord._load_distributed_model(model_config)

    peer1_meta = coord.workers["peer-1"].stub.last_metadata
    lead_meta = coord.workers["lead-1"].stub.last_metadata
    assert peer1_meta["distributed_role"] == "rpc_server"
    assert peer1_meta["rpc_bind_port"] == "50151"
    assert lead_meta["distributed_role"] == "lead"
    assert lead_meta["rpc_peers"] == "10.0.0.2:50151,10.0.0.3:50151"
    # Equal VRAM (40/20/20 GB across lead+2 peers) -> 0.5/0.25/0.25 split.
    assert lead_meta["tensor_split"] == "0.5,0.25,0.25"


@pytest.mark.asyncio
async def test_load_distributed_model_peer_failure_rolls_back_earlier_peers() -> None:
    log: List[Any] = []
    coord = _cluster(log, fail="peer-2")
    model_config = ModelRegistry.get_model("distributed-orch-test")
    assert model_config is not None

    ok = await coord._load_distributed_model(model_config)

    assert ok is False
    # peer-1 loaded, peer-2 failed, peer-1 rolled back -- the lead is never touched.
    assert log == [("load", "peer-1"), ("load", "peer-2"), ("unload", "peer-1")]
    assert "distributed-orch-test" not in coord.workers["peer-1"].loaded_models


@pytest.mark.asyncio
async def test_load_distributed_model_lead_failure_rolls_back_all_peers() -> None:
    log: List[Any] = []
    coord = _cluster(log, fail="lead-1")
    model_config = ModelRegistry.get_model("distributed-orch-test")
    assert model_config is not None

    ok = await coord._load_distributed_model(model_config)

    assert ok is False
    assert log == [
        ("load", "peer-1"),
        ("load", "peer-2"),
        ("load", "lead-1"),
        ("unload", "peer-1"),
        ("unload", "peer-2"),
    ]
    assert "distributed-orch-test" not in coord.workers["peer-1"].loaded_models
    assert "distributed-orch-test" not in coord.workers["peer-2"].loaded_models


@pytest.mark.asyncio
async def test_load_distributed_model_raises_for_unregistered_worker() -> None:
    log: List[Any] = []
    ModelRegistry.load_from_dict(_DISTRIBUTED_REGISTRY)
    coord = ClusterCoordinator(make_settings())
    coord.workers["lead-1"] = _make_worker("lead-1", "10.0.0.1:50051", log, [1])
    # peer-1/peer-2 were never registered.
    model_config = ModelRegistry.get_model("distributed-orch-test")
    assert model_config is not None

    with pytest.raises(ValueError, match="unregistered"):
        await coord._load_distributed_model(model_config)
    assert log == []


@pytest.mark.asyncio
async def test_load_distributed_model_raises_for_unhealthy_worker() -> None:
    log: List[Any] = []
    coord = _cluster(log)
    coord.workers["peer-1"].state = WorkerState.UNHEALTHY
    model_config = ModelRegistry.get_model("distributed-orch-test")
    assert model_config is not None

    with pytest.raises(ValueError, match="not healthy"):
        await coord._load_distributed_model(model_config)
    assert log == []


# ---------------------------------------------------------------------------
# unload_model: reverse order (lead first)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unload_model_distributed_unloads_lead_before_peers() -> None:
    log: List[Any] = []
    coord = _cluster(log)
    model_config = ModelRegistry.get_model("distributed-orch-test")
    assert model_config is not None
    assert await coord._load_distributed_model(model_config) is True
    log.clear()

    unloaded_from = await coord.unload_model("distributed-orch-test")

    assert set(unloaded_from) == {"lead-1", "peer-1", "peer-2"}
    assert log == [("unload", "lead-1"), ("unload", "peer-1"), ("unload", "peer-2")]


# ---------------------------------------------------------------------------
# Inference pinning + load-on-demand dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_request_pins_distributed_model_to_lead() -> None:
    log: List[Any] = []
    coord = _cluster(log)

    ctx = await coord.submit_request("distributed-orch-test", "hello")

    assert ctx.target_worker_id == "lead-1"


@pytest.mark.asyncio
async def test_submit_request_explicit_worker_id_wins_over_lead_pin() -> None:
    log: List[Any] = []
    coord = _cluster(log)

    ctx = await coord.submit_request("distributed-orch-test", "hello", worker_id="peer-1")

    assert ctx.target_worker_id == "peer-1"


@pytest.mark.asyncio
async def test_execute_request_dispatches_distributed_load_on_demand() -> None:
    log: List[Any] = []
    coord = _cluster(log)
    lead = coord.workers["lead-1"]

    def _infer(request: Any) -> Any:
        async def _gen() -> Any:
            yield pb.InferenceResponse(text="hi", tokens_generated=1, finished=True)

        return _gen()

    lead.stub.Infer = _infer

    ctx = RequestContext(
        id="req-1", model_name="distributed-orch-test", prompt="hello", params={}, created_at=0.0
    )
    await coord._execute_request(ctx, lead)

    assert ctx.error is None
    assert log[:3] == [("load", "peer-1"), ("load", "peer-2"), ("load", "lead-1")]
    assert "distributed-orch-test" in lead.loaded_models


# ---------------------------------------------------------------------------
# Split derivation / endpoint helpers (unit-level)
# ---------------------------------------------------------------------------


def test_address_host_parses_host_port_and_ipv6() -> None:
    assert _address_host("10.0.0.2:50151") == "10.0.0.2"
    assert _address_host("[::1]:50151") == "::1"
    assert _address_host("just-a-host") == "just-a-host"


def test_largest_remainder_split_sums_to_exactly_one() -> None:
    weights = _largest_remainder_split([1.0, 1.0, 1.0])
    assert sum(weights) == pytest.approx(1.0)
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-4)


def test_derive_split_weights_proportional_to_total_memory() -> None:
    log: List[Any] = []
    coord = _cluster(log)
    lead = coord.workers["lead-1"]  # 40 GB
    peer1 = coord.workers["peer-1"]  # 20 GB
    peer2 = coord.workers["peer-2"]  # 20 GB

    weights = coord._derive_split_weights([(lead, [0]), (peer1, [0]), (peer2, [0])])

    assert sum(weights) == pytest.approx(1.0)
    assert weights == pytest.approx([0.5, 0.25, 0.25], abs=1e-4)


def test_peer_endpoints_increments_port_per_lent_gpu() -> None:
    log: List[Any] = []
    coord = _cluster(log)
    peer = coord.workers["peer-1"]
    peer.gpus = [
        pb.GPUInfo(id=0, name="g0", total_memory=1, available_memory=1),
        pb.GPUInfo(id=1, name="g1", total_memory=1, available_memory=1),
    ]

    endpoints = coord._peer_endpoints(peer, [0, 1], 50151)

    assert endpoints == ["10.0.0.2:50151", "10.0.0.2:50152"]


def test_resolve_node_gpu_ids_uses_explicit_override() -> None:
    log: List[Any] = []
    ModelRegistry.load_from_dict(
        {
            "models": {
                "distributed-gpu-override-test": {
                    "family": "qwen",
                    "parameters": "32B",
                    "min_memory_gb": 20,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
                        "file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
                    },
                    "distributed": {
                        "enabled": True,
                        "lead": "lead-1",
                        "peers": ["peer-1"],
                        "gpu_ids": {"peer-1": [3]},
                    },
                }
            }
        }
    )
    coord = ClusterCoordinator(make_settings())
    peer = _make_worker("peer-1", "10.0.0.2:50051", log, [1, 1, 1, 1])  # GPUs 0..3
    model_config = ModelRegistry.get_model("distributed-gpu-override-test")
    assert model_config is not None

    assert coord._resolve_node_gpu_ids(peer, model_config, "peer-1") == [3]


def test_resolve_node_gpu_ids_defaults_to_every_reported_gpu() -> None:
    log: List[Any] = []
    coord = _cluster(log)
    peer = coord.workers["peer-1"]
    peer.gpus = [
        pb.GPUInfo(id=0, name="g0", total_memory=1, available_memory=1),
        pb.GPUInfo(id=1, name="g1", total_memory=1, available_memory=1),
    ]
    model_config = ModelRegistry.get_model("distributed-orch-test")
    assert model_config is not None

    assert coord._resolve_node_gpu_ids(peer, model_config, "peer-1") == [0, 1]
