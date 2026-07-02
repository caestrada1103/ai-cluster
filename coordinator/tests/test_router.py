"""Tests for coordinator.router — CircuitBreaker, WorkerLoad, enums, and RequestRouter."""

from typing import Any, Dict
from unittest.mock import Mock

import pytest

from coordinator.coordinator import WorkerInfo
from coordinator.router import (
    CircuitBreaker,
    LoadBalancingStrategy,
    QueuePriority,
    RequestRouter,
    WorkerLoad,
)
from coordinator.tests.conftest import make_settings

# ---------------------------------------------------------------------------
# Enum sanity checks
# ---------------------------------------------------------------------------


def test_load_balancing_strategy_values() -> None:
    assert LoadBalancingStrategy.ROUND_ROBIN.value == "round_robin"
    assert LoadBalancingStrategy.LEAST_LOAD.value == "least_load"
    assert LoadBalancingStrategy.RANDOM.value == "random"
    assert LoadBalancingStrategy.AFFINITY.value == "affinity"
    assert LoadBalancingStrategy.POWER_OF_TWO.value == "power_of_two"


def test_queue_priority_ordering() -> None:
    assert QueuePriority.CRITICAL.value < QueuePriority.HIGH.value
    assert QueuePriority.HIGH.value < QueuePriority.NORMAL.value
    assert QueuePriority.NORMAL.value < QueuePriority.LOW.value
    assert QueuePriority.LOW.value < QueuePriority.BATCH.value


# ---------------------------------------------------------------------------
# WorkerLoad
# ---------------------------------------------------------------------------


def test_workload_score_all_zero() -> None:
    load = WorkerLoad(worker_id="w1")
    assert load.load_score == 0.0


def test_workload_score_active_requests() -> None:
    load = WorkerLoad(worker_id="w1", active_requests=5)
    # 5 active * 1.0 weight
    assert load.load_score == pytest.approx(5.0)


def test_workload_score_memory_pressure() -> None:
    # memory_used / memory_total * 2.0 weight
    load = WorkerLoad(worker_id="w1", memory_used_gb=8.0, memory_total_gb=8.0)
    score = load.load_score
    assert score == pytest.approx(2.0)


def test_workload_score_combined() -> None:
    load = WorkerLoad(
        worker_id="w1",
        active_requests=2,
        queued_requests=4,
        memory_used_gb=4.0,
        memory_total_gb=8.0,
        avg_latency_ms=100.0,
        error_rate=0.1,
    )
    expected = 2 * 1.0 + 4 * 0.5 + 0 * 0.1 + (4.0 / 8.0) * 2.0 + 100.0 * 0.01 + 0.1 * 10.0
    assert load.load_score == pytest.approx(expected)


def test_workload_score_memory_total_zero_doesnt_divide_by_zero() -> None:
    load = WorkerLoad(worker_id="w1", memory_used_gb=8.0, memory_total_gb=0.0)
    # max(0.0, 1) → 1, so score = 8.0 / 1 * 2.0 = 16.0
    assert load.load_score == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# CircuitBreaker — initial state
# ---------------------------------------------------------------------------


def test_cb_initial_state_closed() -> None:
    cb = CircuitBreaker()
    assert cb.state == CircuitBreaker.State.CLOSED


def test_cb_allows_request_when_closed() -> None:
    cb = CircuitBreaker()
    assert cb.allow_request() is True


def test_cb_total_requests_starts_zero() -> None:
    cb = CircuitBreaker()
    assert cb.total_requests == 0


# ---------------------------------------------------------------------------
# CircuitBreaker — state transitions
# ---------------------------------------------------------------------------


def test_cb_opens_after_threshold_failures() -> None:
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitBreaker.State.OPEN


def test_cb_does_not_open_before_threshold() -> None:
    cb = CircuitBreaker(failure_threshold=5)
    for _ in range(4):
        cb.record_failure()
    assert cb.state == CircuitBreaker.State.CLOSED


def test_cb_blocks_requests_when_open() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
    cb.record_failure()
    assert cb.state == CircuitBreaker.State.OPEN
    assert cb.allow_request() is False


def test_cb_transitions_to_half_open_after_timeout() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
    cb.record_failure()
    assert cb.state == CircuitBreaker.State.OPEN
    # recovery_timeout=0 means it should transition immediately
    assert cb.allow_request() is True
    # mypy narrows cb.state from the assert above and doesn't know allow_request()
    # mutates it — this is a real state transition, not a dead comparison.
    assert cb.state == CircuitBreaker.State.HALF_OPEN  # type: ignore[comparison-overlap]


def test_cb_closes_after_successes_in_half_open() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0, half_open_max_requests=2)
    cb.record_failure()
    cb.allow_request()  # transitions to HALF_OPEN
    cb.record_success()
    cb.record_success()
    assert cb.state == CircuitBreaker.State.CLOSED


def test_cb_reopens_on_failure_in_half_open() -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0, half_open_max_requests=3)
    cb.record_failure()
    cb.allow_request()  # → HALF_OPEN
    assert cb.state == CircuitBreaker.State.HALF_OPEN
    cb.record_failure()
    # mypy narrows cb.state from the assert above and doesn't know record_failure()
    # mutates it — this is a real state transition, not a dead comparison.
    assert cb.state == CircuitBreaker.State.OPEN  # type: ignore[comparison-overlap]


# ---------------------------------------------------------------------------
# CircuitBreaker — counters
# ---------------------------------------------------------------------------


def test_cb_counts_successes_and_failures() -> None:
    cb = CircuitBreaker()
    cb.record_success()
    cb.record_success()
    cb.record_failure()
    assert cb.total_successes == 2
    assert cb.total_failures == 1
    assert cb.total_requests == 3


def test_cb_stats_contains_expected_keys() -> None:
    cb = CircuitBreaker()
    stats = cb.stats
    assert "state" in stats
    assert "failure_count" in stats
    assert "total_failures" in stats
    assert "total_successes" in stats
    assert "last_failure" in stats


def test_cb_stats_state_is_string() -> None:
    cb = CircuitBreaker()
    assert isinstance(cb.stats["state"], str)
    assert cb.stats["state"] == "closed"


# ---------------------------------------------------------------------------
# RequestRouter — construction, selection, affinity, queues, circuit breakers
# ---------------------------------------------------------------------------


def _mock_worker(worker_id: str, active: int = 0) -> Any:
    worker = Mock(spec=WorkerInfo)
    worker.id = worker_id
    worker.active_requests = active
    worker.is_available = True
    worker.loaded_models = {"deepseek-7b": object()}
    worker.available_memory = 64 * 10**9
    worker.gpus = []
    worker.avg_latency_ms = 0.0
    return worker


def test_router_constructs_with_real_settings() -> None:
    settings = make_settings()
    router = RequestRouter(get_workers_callback=dict, settings=settings)
    assert router.strategy.value == "least_load"


@pytest.mark.asyncio
async def test_pick_worker_least_load() -> None:
    settings = make_settings()
    workers: Dict[str, Any] = {
        "w1": _mock_worker("w1", active=5),
        "w2": _mock_worker("w2", active=1),
    }
    router = RequestRouter(get_workers_callback=lambda: workers, settings=settings)
    picked = await router.pick_worker(workers, "deepseek-7b", session_id=None)
    assert picked is not None and picked.id == "w2"


@pytest.mark.asyncio
async def test_affinity_keyed_by_session_with_ttl() -> None:
    settings = make_settings(routing_strategy="affinity", affinity_ttl_seconds=600.0)
    workers: Dict[str, Any] = {"w1": _mock_worker("w1"), "w2": _mock_worker("w2")}
    router = RequestRouter(get_workers_callback=lambda: workers, settings=settings)
    first = await router.pick_worker(workers, "deepseek-7b", session_id="sess-1")
    second = await router.pick_worker(workers, "deepseek-7b", session_id="sess-1")
    assert first is not None and second is not None
    assert first.id == second.id  # sticky per session
    assert "sess-1" in router.affinity_map


def test_batch_priority_is_drained() -> None:
    """BATCH must appear in the queue drain order (starvation fix)."""
    settings = make_settings()
    router = RequestRouter(get_workers_callback=dict, settings=settings)
    assert QueuePriority.BATCH in router.drain_order


def test_circuit_breaker_settings_come_from_flat_fields() -> None:
    settings = make_settings(circuit_breaker_failure_threshold=2)
    router = RequestRouter(get_workers_callback=dict, settings=settings)
    router.record_failure("w1")
    router.record_failure("w1")
    assert router.circuit_breakers["w1"].state.value == "open"
