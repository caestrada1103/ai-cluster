"""Tests for coordinator.router — CircuitBreaker, WorkerLoad, and enums.

RequestRouter is not tested here because it depends on coordinator.monitoring.metrics
(Prometheus counters/gauges) and settings.routing (loaded from coordinator.yaml at
runtime). The pure dataclasses and state machines are fully testable in isolation.
"""

import time

import pytest

from coordinator.router import (
    CircuitBreaker,
    LoadBalancingStrategy,
    QueuePriority,
    WorkerLoad,
)


# ---------------------------------------------------------------------------
# Enum sanity checks
# ---------------------------------------------------------------------------

def test_load_balancing_strategy_values():
    assert LoadBalancingStrategy.ROUND_ROBIN.value == "round_robin"
    assert LoadBalancingStrategy.LEAST_LOAD.value == "least_load"
    assert LoadBalancingStrategy.RANDOM.value == "random"
    assert LoadBalancingStrategy.AFFINITY.value == "affinity"
    assert LoadBalancingStrategy.POWER_OF_TWO.value == "power_of_two"


def test_queue_priority_ordering():
    assert QueuePriority.CRITICAL.value < QueuePriority.HIGH.value
    assert QueuePriority.HIGH.value < QueuePriority.NORMAL.value
    assert QueuePriority.NORMAL.value < QueuePriority.LOW.value
    assert QueuePriority.LOW.value < QueuePriority.BATCH.value


# ---------------------------------------------------------------------------
# WorkerLoad
# ---------------------------------------------------------------------------

def test_workload_score_all_zero():
    load = WorkerLoad(worker_id="w1")
    assert load.load_score == 0.0


def test_workload_score_active_requests():
    load = WorkerLoad(worker_id="w1", active_requests=5)
    # 5 active * 1.0 weight
    assert load.load_score == pytest.approx(5.0)


def test_workload_score_memory_pressure():
    # memory_used / memory_total * 2.0 weight
    load = WorkerLoad(worker_id="w1", memory_used_gb=8.0, memory_total_gb=8.0)
    score = load.load_score
    assert score == pytest.approx(2.0)


def test_workload_score_combined():
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


def test_workload_score_memory_total_zero_doesnt_divide_by_zero():
    load = WorkerLoad(worker_id="w1", memory_used_gb=8.0, memory_total_gb=0.0)
    # max(0.0, 1) → 1, so score = 8.0 / 1 * 2.0 = 16.0
    assert load.load_score == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# CircuitBreaker — initial state
# ---------------------------------------------------------------------------

def test_cb_initial_state_closed():
    cb = CircuitBreaker()
    assert cb.state == CircuitBreaker.State.CLOSED


def test_cb_allows_request_when_closed():
    cb = CircuitBreaker()
    assert cb.allow_request() is True


def test_cb_total_requests_starts_zero():
    cb = CircuitBreaker()
    assert cb.total_requests == 0


# ---------------------------------------------------------------------------
# CircuitBreaker — state transitions
# ---------------------------------------------------------------------------

def test_cb_opens_after_threshold_failures():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitBreaker.State.OPEN


def test_cb_does_not_open_before_threshold():
    cb = CircuitBreaker(failure_threshold=5)
    for _ in range(4):
        cb.record_failure()
    assert cb.state == CircuitBreaker.State.CLOSED


def test_cb_blocks_requests_when_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
    cb.record_failure()
    assert cb.state == CircuitBreaker.State.OPEN
    assert cb.allow_request() is False


def test_cb_transitions_to_half_open_after_timeout():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
    cb.record_failure()
    assert cb.state == CircuitBreaker.State.OPEN
    # recovery_timeout=0 means it should transition immediately
    assert cb.allow_request() is True
    assert cb.state == CircuitBreaker.State.HALF_OPEN


def test_cb_closes_after_successes_in_half_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0, half_open_max_requests=2)
    cb.record_failure()
    cb.allow_request()  # transitions to HALF_OPEN
    cb.record_success()
    cb.record_success()
    assert cb.state == CircuitBreaker.State.CLOSED


def test_cb_reopens_on_failure_in_half_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0, half_open_max_requests=3)
    cb.record_failure()
    cb.allow_request()  # → HALF_OPEN
    assert cb.state == CircuitBreaker.State.HALF_OPEN
    cb.record_failure()
    assert cb.state == CircuitBreaker.State.OPEN


# ---------------------------------------------------------------------------
# CircuitBreaker — counters
# ---------------------------------------------------------------------------

def test_cb_counts_successes_and_failures():
    cb = CircuitBreaker()
    cb.record_success()
    cb.record_success()
    cb.record_failure()
    assert cb.total_successes == 2
    assert cb.total_failures == 1
    assert cb.total_requests == 3


def test_cb_stats_contains_expected_keys():
    cb = CircuitBreaker()
    stats = cb.stats
    assert "state" in stats
    assert "failure_count" in stats
    assert "total_failures" in stats
    assert "total_successes" in stats
    assert "last_failure" in stats


def test_cb_stats_state_is_string():
    cb = CircuitBreaker()
    assert isinstance(cb.stats["state"], str)
    assert cb.stats["state"] == "closed"
