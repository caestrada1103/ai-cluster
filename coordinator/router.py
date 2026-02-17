"""Request routing module for distributing inference requests to workers.

This module handles:
- Model-aware routing
- Load balancing strategies
- Circuit breaking
- Request queuing and prioritization
- Affinity routing
"""

import asyncio
import logging
import time
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import hashlib

from coordinator.coordinator import WorkerInfo
from coordinator.models import ModelConfig, ModelRegistry
from coordinator.monitoring import metrics

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOAD = "least_load"
    RANDOM = "random"
    AFFINITY = "affinity"
    POWER_OF_TWO = "power_of_two"


class RoutingStrategy(str, Enum):
    """Request routing strategies."""
    MODEL_AWARE = "model_aware"
    MEMORY_AWARE = "memory_aware"
    PERFORMANCE_AWARE = "performance_aware"
    COST_AWARE = "cost_aware"


class QueuePriority(Enum):
    """Request priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4


@dataclass
class QueuedRequest:
    """Request waiting in queue."""
    
    request_id: str
    model_name: str
    prompt: str
    params: Dict[str, Any]
    priority: QueuePriority
    enqueued_at: float
    timeout: float
    future: asyncio.Future
    
    @property
    def age(self) -> float:
        return time.time() - self.enqueued_at


@dataclass
class WorkerLoad:
    """Current load on a worker."""
    
    worker_id: str
    active_requests: int = 0
    queued_requests: int = 0
    gpu_utilization: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    last_update: float = field(default_factory=time.time)
    
    @property
    def load_score(self) -> float:
        """Calculate composite load score."""
        # Weighted combination of metrics
        return (
            self.active_requests * 1.0 +
            self.queued_requests * 0.5 +
            self.gpu_utilization * 0.1 +
            (self.memory_used_gb / max(self.memory_total_gb, 1)) * 2.0 +
            self.avg_latency_ms * 0.01 +
            self.error_rate * 10.0
        )


class CircuitBreaker:
    """Circuit breaker for failing workers."""
    
    class State(Enum):
        CLOSED = "closed"      # Normal operation
        OPEN = "open"          # Failing, don't send requests
        HALF_OPEN = "half_open"  # Testing if recovered
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_requests: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests
        
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_requests = 0
        self.total_failures = 0
        self.total_successes = 0
    
    def record_success(self):
        """Record a successful request."""
        self.total_successes += 1
        
        if self.state == self.State.HALF_OPEN:
            self.half_open_requests -= 1
            if self.half_open_requests <= 0:
                logger.info("Circuit breaker closing after successful test")
                self._close()
    
    def record_failure(self):
        """Record a failed request."""
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        if self.state == self.State.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                self._open()
        
        elif self.state == self.State.HALF_OPEN:
            logger.warning("Circuit breaker re-opening after test failure")
            self._open()
    
    def _open(self):
        """Open the circuit."""
        self.state = self.State.OPEN
        self.failure_count = 0
        self.half_open_requests = 0
    
    def _close(self):
        """Close the circuit."""
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.half_open_requests = 0
    
    def _half_open(self):
        """Move to half-open state."""
        self.state = self.State.HALF_OPEN
        self.half_open_requests = 0
    
    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == self.State.CLOSED:
            return True
        
        if self.state == self.State.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self._half_open()
                return True
            return False
        
        if self.state == self.State.HALF_OPEN:
            if self.half_open_requests < self.half_open_max_requests:
                self.half_open_requests += 1
                return True
            return False
        
        return False
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "last_failure": self.last_failure_time,
        }


class RequestRouter:
    """Routes inference requests to appropriate workers."""
    
    def __init__(
        self,
        get_workers_callback: Callable[[], Dict[str, WorkerInfo]],
        settings: Any,
    ):
        self.get_workers = get_workers_callback
        self.settings = settings
        
        # Load balancing
        self.strategy = LoadBalancingStrategy(
            settings.routing.get("strategy", "least_load")
        )
        self.rr_index: Dict[str, int] = defaultdict(int)  # Per-model round-robin index
        
        # Circuit breakers per worker
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Request queues by priority
        self.queues: Dict[QueuePriority, asyncio.Queue] = {
            QueuePriority.CRITICAL: asyncio.Queue(),
            QueuePriority.HIGH: asyncio.Queue(),
            QueuePriority.NORMAL: asyncio.Queue(),
            QueuePriority.LOW: asyncio.Queue(),
            QueuePriority.BATCH: asyncio.Queue(maxsize=settings.max_batch_size),
        }
        
        # Worker load tracking
        self.worker_loads: Dict[str, WorkerLoad] = {}
        
        # Affinity tracking (request_id -> worker_id)
        self.affinity_map: Dict[str, str] = {}
        
        # Background tasks
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._load_updater_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.routed_requests = metrics.counter(
            "router_routed_requests_total",
            "Total requests routed",
            ["strategy", "model"]
        )
        self.queue_size = metrics.gauge(
            "router_queue_size",
            "Current queue size",
            ["priority"]
        )
        self.circuit_breaker_open = metrics.gauge(
            "router_circuit_breaker_open",
            "Circuit breaker open count"
        )
    
    async def start(self):
        """Start background tasks."""
        self._queue_processor_task = asyncio.create_task(self._process_queues())
        self._load_updater_task = asyncio.create_task(self._update_loads())
        logger.info("Request router started")
    
    async def stop(self):
        """Stop background tasks."""
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
        
        if self._load_updater_task:
            self._load_updater_task.cancel()
            try:
                await self._load_updater_task
            except asyncio.CancelledError:
                pass
    
    async def route_request(
        self,
        request_id: str,
        model_name: str,
        prompt: str,
        params: Dict[str, Any],
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> Tuple[Optional[WorkerInfo], Optional[QueuedRequest]]:
        """
        Route a request to the best worker or queue it.
        
        Returns:
            Tuple of (selected_worker, queued_request)
            If worker is None, request was queued
        """
        # Check circuit breakers first
        workers = self.get_workers()
        available_workers = self._get_available_workers(workers, model_name)
        
        if not available_workers:
            # No workers available, queue the request
            return None, await self._queue_request(
                request_id, model_name, prompt, params, priority
            )
        
        # Select best worker
        worker = await self._select_worker(available_workers, model_name, request_id)
        
        if not worker:
            # No suitable worker, queue
            return None, await self._queue_request(
                request_id, model_name, prompt, params, priority
            )
        
        # Update metrics
        self.routed_requests.labels(
            strategy=self.strategy.value,
            model=model_name
        ).inc()
        
        return worker, None
    
    def _get_available_workers(
        self,
        workers: Dict[str, WorkerInfo],
        model_name: str,
    ) -> List[WorkerInfo]:
        """Get workers that can handle the request."""
        available = []
        
        for worker_id, worker in workers.items():
            # Check circuit breaker
            cb = self.circuit_breakers.get(worker_id)
            if cb and not cb.allow_request():
                continue
            
            # Check if worker is healthy
            if not worker.is_available:
                continue
            
            # Check if model is loaded or can be loaded
            model_config = ModelRegistry.get_model(model_name)
            if not model_config:
                continue
            
            if model_name in worker.loaded_models:
                available.append(worker)
            elif worker.available_memory >= model_config.min_memory_gb * 1e9:
                available.append(worker)
        
        return available
    
    async def _select_worker(
        self,
        workers: List[WorkerInfo],
        model_name: str,
        request_id: str,
    ) -> Optional[WorkerInfo]:
        """Select the best worker using the configured strategy."""
        if not workers:
            return None
        
        # Check affinity first
        if request_id in self.affinity_map:
            worker_id = self.affinity_map[request_id]
            for worker in workers:
                if worker.id == worker_id:
                    return worker
        
        # Apply strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(workers, model_name)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_LOAD:
            return await self._least_load_select(workers)
        
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(workers)
        
        elif self.strategy == LoadBalancingStrategy.POWER_OF_TWO:
            return await self._power_of_two_select(workers)
        
        elif self.strategy == LoadBalancingStrategy.AFFINITY:
            # Create affinity based on prompt hash
            affinity_key = hashlib.md5(f"{model_name}:{request_id}".encode()).hexdigest()
            index = int(affinity_key[:8], 16) % len(workers)
            worker = workers[index]
            self.affinity_map[request_id] = worker.id
            return worker
        
        # Default to least load
        return await self._least_load_select(workers)
    
    def _round_robin_select(
        self,
        workers: List[WorkerInfo],
        model_name: str,
    ) -> WorkerInfo:
        """Round-robin selection."""
        index = self.rr_index[model_name]
        worker = workers[index % len(workers)]
        self.rr_index[model_name] = index + 1
        return worker
    
    async def _least_load_select(
        self,
        workers: List[WorkerInfo],
    ) -> WorkerInfo:
        """Select worker with lowest load."""
        best_worker = None
        best_score = float('inf')
        
        for worker in workers:
            load = self.worker_loads.get(worker.id)
            if load:
                score = load.load_score
            else:
                score = worker.active_requests  # Fallback
            
            if score < best_score:
                best_score = score
                best_worker = worker
        
        return best_worker or workers[0]
    
    async def _power_of_two_select(
        self,
        workers: List[WorkerInfo],
    ) -> WorkerInfo:
        """Pick two random workers, select the one with lower load."""
        if len(workers) < 2:
            return workers[0]
        
        w1, w2 = random.sample(workers, 2)
        
        load1 = self.worker_loads.get(w1.id)
        load2 = self.worker_loads.get(w2.id)
        
        score1 = load1.load_score if load1 else w1.active_requests
        score2 = load2.load_score if load2 else w2.active_requests
        
        return w1 if score1 <= score2 else w2
    
    async def _queue_request(
        self,
        request_id: str,
        model_name: str,
        prompt: str,
        params: Dict[str, Any],
        priority: QueuePriority,
    ) -> QueuedRequest:
        """Queue a request for later processing."""
        # Create future for the result
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        request = QueuedRequest(
            request_id=request_id,
            model_name=model_name,
            prompt=prompt,
            params=params,
            priority=priority,
            enqueued_at=time.time(),
            timeout=params.get("timeout", self.settings.request_timeout),
            future=future,
        )
        
        # Add to appropriate queue
        await self.queues[priority].put(request)
        
        # Update metrics
        self.queue_size.labels(priority=priority.value).inc()
        
        logger.debug(f"Request {request_id} queued with priority {priority.value}")
        
        return request
    
    async def _process_queues(self):
        """Process queued requests in priority order."""
        while True:
            try:
                # Check queues in priority order
                request = None
                for priority in [
                    QueuePriority.CRITICAL,
                    QueuePriority.HIGH,
                    QueuePriority.NORMAL,
                    QueuePriority.LOW,
                ]:
                    try:
                        # Non-blocking check
                        request = self.queues[priority].get_nowait()
                        self.queue_size.labels(priority=priority.value).dec()
                        break
                    except asyncio.QueueEmpty:
                        continue
                
                if request is None:
                    # All queues empty, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if request timed out
                if time.time() - request.enqueued_at > request.timeout:
                    request.future.set_exception(
                        TimeoutError(f"Request {request.request_id} timed out in queue")
                    )
                    continue
                
                # Try to route again
                worker, _ = await self.route_request(
                    request.request_id,
                    request.model_name,
                    request.prompt,
                    request.params,
                    request.priority,
                )
                
                if worker:
                    # Success! Return the worker to the caller
                    request.future.set_result(worker)
                else:
                    # Still no worker, re-queue with backoff
                    await asyncio.sleep(0.5)
                    await self.queues[request.priority].put(request)
                    self.queue_size.labels(priority=request.priority.value).inc()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(1)
    
    async def _update_loads(self):
        """Update worker load information."""
        while True:
            try:
                workers = self.get_workers()
                
                for worker_id, worker in workers.items():
                    # Calculate load metrics
                    load = WorkerLoad(
                        worker_id=worker_id,
                        active_requests=worker.active_requests,
                        gpu_utilization=sum(g.utilization for g in worker.gpus) / len(worker.gpus) if worker.gpus else 0,
                        memory_used_gb=sum(g.total_memory - g.available_memory for g in worker.gpus) / 1e9,
                        memory_total_gb=sum(g.total_memory for g in worker.gpus) / 1e9,
                        avg_latency_ms=worker.avg_latency_ms,
                    )
                    
                    # Calculate error rate from circuit breaker
                    cb = self.circuit_breakers.get(worker_id)
                    if cb and cb.total_requests > 0:
                        load.error_rate = cb.total_failures / cb.total_requests
                    
                    self.worker_loads[worker_id] = load
                
                # Clean up stale loads
                current_ids = set(workers.keys())
                for worker_id in list(self.worker_loads.keys()):
                    if worker_id not in current_ids:
                        del self.worker_loads[worker_id]
                
                # Update circuit breaker metrics
                open_count = sum(
                    1 for cb in self.circuit_breakers.values()
                    if cb.state == CircuitBreaker.State.OPEN
                )
                self.circuit_breaker_open.set(open_count)
                
            except Exception as e:
                logger.error(f"Load updater error: {e}")
            
            await asyncio.sleep(5)  # Update every 5 seconds
    
    def record_success(self, worker_id: str):
        """Record a successful request."""
        if worker_id in self.circuit_breakers:
            self.circuit_breakers[worker_id].record_success()
    
    def record_failure(self, worker_id: str):
        """Record a failed request."""
        if worker_id not in self.circuit_breakers:
            self.circuit_breakers[worker_id] = CircuitBreaker(
                failure_threshold=self.settings.routing.get("circuit_breaker", {}).get("failure_threshold", 5),
                recovery_timeout=self.settings.routing.get("circuit_breaker", {}).get("recovery_timeout", 30),
            )
        
        self.circuit_breakers[worker_id].record_failure()
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "queues": {
                priority.value: {
                    "size": self.queues[priority].qsize(),
                }
                for priority in QueuePriority
            },
            "total_queued": sum(q.qsize() for q in self.queues.values()),
        }
    
    def get_circuit_breaker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker statistics."""
        return {
            worker_id: cb.stats
            for worker_id, cb in self.circuit_breakers.items()
        }
    
    def get_load_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get worker load statistics."""
        return {
            worker_id: {
                "load_score": load.load_score,
                "active_requests": load.active_requests,
                "gpu_utilization": load.gpu_utilization,
                "memory_used_gb": load.memory_used_gb,
                "memory_total_gb": load.memory_total_gb,
                "avg_latency_ms": load.avg_latency_ms,
                "error_rate": load.error_rate,
            }
            for worker_id, load in self.worker_loads.items()
        }