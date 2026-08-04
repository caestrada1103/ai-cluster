"""Core coordinator logic for managing workers and routing requests."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import grpc

import coordinator.proto.cluster_pb2 as pb
import coordinator.proto.cluster_pb2_grpc as pb_grpc
from coordinator import audit
from coordinator.config import Settings
from coordinator.discovery import WorkerDiscovery
from coordinator.models import ModelRegistry, Quantization
from coordinator.monitoring import metrics

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker connection state."""

    CONNECTING = "connecting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class WorkerInfo:
    """Information about a connected worker."""

    id: str
    address: str
    channel: grpc.aio.Channel
    stub: pb_grpc.WorkerStub
    state: WorkerState = WorkerState.CONNECTING
    gpus: List[pb.GPUInfo] = field(default_factory=list)
    loaded_models: Dict[str, pb.LoadedModelInfo] = field(default_factory=dict)

    # Health tracking
    last_health_check: float = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None

    # Performance metrics
    active_requests: int = 0
    total_requests: int = 0
    avg_latency_ms: float = 0
    max_requests: int = 10  # Configurable via coordinator settings
    health_check_timeout: int = 5
    max_failures: int = 3

    async def get_status(self) -> Optional[pb.WorkerStatus]:
        """Get current worker status."""
        try:
            status = await self.stub.GetStatus(pb.Empty(), timeout=self.health_check_timeout)
            self.state = WorkerState.HEALTHY
            self.consecutive_failures = 0
            self.gpus = list(status.gpus)
            self.loaded_models = {m.model_name: m for m in status.loaded_models}
            self.active_requests = status.active_requests
            return status
        except Exception as e:
            self.consecutive_failures += 1
            self.last_error = str(e)
            if self.consecutive_failures >= self.max_failures:
                self.state = WorkerState.UNHEALTHY
            logger.warning(f"Failed to get status from {self.id}: {e}")
            return None

    @property
    def total_memory(self) -> int:
        """Total GPU memory in bytes."""
        return sum(g.total_memory for g in self.gpus)

    @property
    def available_memory(self) -> int:
        """Available GPU memory in bytes."""
        return sum(g.available_memory for g in self.gpus)

    @property
    def is_available(self) -> bool:
        """Whether worker can accept new requests."""
        return self.state == WorkerState.HEALTHY and self.active_requests < self.max_requests


@dataclass
class RequestContext:
    """Context for an inference request."""

    id: str
    model_name: str
    prompt: str
    params: Dict[str, Any]
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None
    target_worker_id: Optional[str] = None
    error: Optional[str] = None
    tokens_generated: int = 0
    # Real worker-reported prompt token count; None if the worker never sent one.
    prompt_tokens: Optional[int] = None

    # Streaming
    token_queue: "asyncio.Queue[Any]" = field(default_factory=asyncio.Queue)
    accumulated_text: str = ""

    # Set exactly once when the request reaches a terminal state (success or error)
    done: asyncio.Event = field(default_factory=asyncio.Event)


class ClusterCoordinator:
    """Main coordinator that manages workers and routes requests."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.workers: Dict[str, WorkerInfo] = {}
        self.request_queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=settings.max_queue_size)
        self.active_requests: Dict[str, RequestContext] = {}

        # Locks for thread safety
        self._workers_lock = asyncio.Lock()
        # Per-model single-flight locks for llamaserver auto-load: concurrent
        # requests for the same unloaded model serialize here so exactly one
        # load runs. Lazily created, never evicted.
        self._autoload_locks: Dict[str, asyncio.Lock] = {}

        # Background tasks
        self._discovery_task: Optional["asyncio.Task[None]"] = None
        self._health_check_task: Optional["asyncio.Task[None]"] = None
        self._processor_task: Optional["asyncio.Task[None]"] = None
        self._inflight: "Set[asyncio.Task[None]]" = set()

        # Components
        self.discovery = WorkerDiscovery(settings)
        # Local import: router imports WorkerInfo from this module (cycle avoidance).
        from coordinator.router import RequestRouter

        self.router = RequestRouter(
            get_workers_callback=lambda: dict(self.workers),
            settings=settings,
        )

        # Load models from configuration
        config_dict = settings.load_models_config()
        if config_dict:
            ModelRegistry.load_from_dict(config_dict)

        # Metrics
        self.request_counter = metrics.counter(
            "coordinator_requests_total", "Total requests processed", ["model", "status"]
        )
        self.request_duration = metrics.histogram(
            "coordinator_request_duration_seconds", "Request duration in seconds", ["model"]
        )
        self.active_requests_gauge = metrics.gauge(
            "coordinator_active_requests", "Currently active requests"
        )

        self.is_running = False

    async def start(self) -> None:
        """Start the coordinator."""
        logger.info("Starting cluster coordinator...")
        self.is_running = True

        # Start components
        await self.discovery.start()
        await self.router.start()

        # Start background tasks
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._processor_task = asyncio.create_task(self._request_processor())

        logger.info("Coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        logger.info("Stopping coordinator...")
        self.is_running = False

        # Cancel background tasks
        for task in [self._discovery_task, self._health_check_task, self._processor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await self.router.stop()

        # Close worker connections
        async with self._workers_lock:
            for worker in self.workers.values():
                await worker.channel.close()

        logger.info("Coordinator stopped")

    async def _discovery_loop(self) -> None:
        """Background task for discovering workers."""
        while self.is_running:
            try:
                # Discover new workers
                addresses = await self.discovery.discover()

                for addr in addresses:
                    await self._connect_worker(addr.address)

                # Remove workers that are no longer discovered
                async with self._workers_lock:
                    discovered_set = {addr.address for addr in addresses}
                    for worker_id in list(self.workers.keys()):
                        worker = self.workers[worker_id]
                        if worker.address not in discovered_set:
                            logger.info(f"Worker {worker_id} no longer discovered")
                            worker.state = WorkerState.OFFLINE

            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")

            await asyncio.sleep(self.settings.discovery_interval)

    async def _connect_worker(self, address: str) -> Optional[WorkerInfo]:
        """Connect to a worker at the given address (network RPC held outside the lock)."""
        async with self._workers_lock:
            for worker in self.workers.values():
                if worker.address == address:
                    return worker

        try:
            logger.info(f"Connecting to worker at {address}")
            channel = grpc.aio.insecure_channel(
                address,
                options=[
                    ("grpc.keepalive_time_ms", 60000),
                    ("grpc.keepalive_timeout_ms", 40000),
                    ("grpc.http2.max_pings_without_data", 0),
                    ("grpc.keepalive_permit_without_calls", 1),
                ],
            )
            stub = pb_grpc.WorkerStub(channel)
            status = await stub.GetStatus(pb.Empty(), timeout=30)
        except Exception as e:
            logger.error(f"Failed to connect to worker at {address}: {e}")
            return None

        worker_id = status.worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        worker = WorkerInfo(
            id=worker_id,
            address=address,
            channel=channel,
            stub=stub,
            max_requests=self.settings.max_concurrent_requests_per_worker,
            health_check_timeout=self.settings.health_check_timeout,
            max_failures=self.settings.max_failures,
        )
        worker.state = WorkerState.HEALTHY
        worker.gpus = list(status.gpus)
        # Adopt models already resident on the worker (e.g. the coordinator
        # restarted while the worker kept a model loaded).
        worker.loaded_models = {m.model_name: m for m in status.loaded_models}

        async with self._workers_lock:
            # Re-check: another task may have connected the same address meanwhile.
            for existing in self.workers.values():
                if existing.address == address:
                    await channel.close()
                    return existing
            if worker_id in self.workers:
                logger.warning(
                    f"Worker id {worker_id} already registered from "
                    f"{self.workers[worker_id].address}; refusing duplicate from {address}"
                )
                await channel.close()
                return None
            self.workers[worker_id] = worker

        logger.info(f"Connected to worker {worker_id} with {len(worker.gpus)} GPUs")
        return worker

    async def _health_check_loop(self) -> None:
        """Background task for checking worker health."""
        while self.is_running:
            try:
                # 1. Get a snapshot of workers under lock
                async with self._workers_lock:
                    workers_to_check = list(self.workers.values())

                if not workers_to_check:
                    await asyncio.sleep(self.settings.health_check_interval)
                    continue

                # Health checks run concurrently with no lock held, so slow
                # network IO doesn't block the request processor. OFFLINE
                # workers are skipped, not resurrected, and evicted below.
                await asyncio.gather(
                    *[
                        worker.get_status()
                        for worker in workers_to_check
                        if worker.state != WorkerState.OFFLINE
                    ],
                    return_exceptions=True,
                )

                # 3. Handle cleanup of offline workers under lock
                async with self._workers_lock:
                    for worker in workers_to_check:
                        if worker.state == WorkerState.OFFLINE:
                            if worker.id in self.workers:
                                try:
                                    await worker.channel.close()
                                except Exception as e:
                                    logger.warning(f"Error closing channel for {worker.id}: {e}")
                                finally:
                                    self.workers.pop(worker.id, None)
                                    logger.info(f"Removed offline worker {worker.id}")

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

            await asyncio.sleep(self.settings.health_check_interval)

    async def _request_processor(self) -> None:
        """Background task for processing queued requests."""
        while self.is_running:
            ctx: Optional[RequestContext] = None
            try:
                # Get next request from queue
                request_id = await self.request_queue.get()

                if request_id not in self.active_requests:
                    continue

                ctx = self.active_requests[request_id]

                # Check for explicit worker targeting first
                worker = None
                if ctx.target_worker_id:
                    async with self._workers_lock:
                        worker = self.workers.get(ctx.target_worker_id)
                    if not worker or not worker.is_available:
                        ctx.error = f"Target worker '{ctx.target_worker_id}' is not available"
                        ctx.completed_at = time.time()
                        ctx.done.set()
                        self.request_counter.labels(model=ctx.model_name, status="error").inc()
                        continue
                else:
                    # Default load balancing logic if no explicit target
                    worker = await self._select_worker(ctx.model_name, ctx.params.get("session_id"))

                if not worker:
                    logger.warning(f"No worker available for model {ctx.model_name}")
                    ctx.error = "No available workers"
                    ctx.completed_at = time.time()
                    ctx.done.set()
                    self.request_counter.labels(model=ctx.model_name, status="error").inc()
                    continue

                logger.debug(f"Selected worker {worker.id} for request {request_id}")
                logger.info(f"Dispatching request {request_id} to worker {worker.id}")
                # Execute request concurrently (keep a strong ref so it isn't GC'd mid-flight)
                task = asyncio.create_task(self._execute_request(ctx, worker))
                self._inflight.add(task)
                task.add_done_callback(self._inflight.discard)

            except asyncio.CancelledError:
                # Propagate — stop() awaits this task expecting CancelledError.
                # Mark any request already picked up so infer()/SSE callers don't
                # hang until their own timeout during coordinator shutdown.
                if ctx is not None and not ctx.done.is_set():
                    ctx.error = "Coordinator is shutting down"
                    ctx.completed_at = time.time()
                    ctx.done.set()
                raise
            except Exception as e:
                logger.error(f"Error in request processor: {e}")

    async def _select_worker(
        self, model_name: str, session_id: Optional[str] = None
    ) -> Optional[WorkerInfo]:
        """Select the best worker via the RequestRouter (strategies + circuit breakers)."""
        async with self._workers_lock:
            workers = dict(self.workers)
        return await self.router.pick_worker(workers, model_name, session_id)

    async def find_worker_for_model(
        self, model_name: str, session_id: Optional[str] = None
    ) -> Optional[WorkerInfo]:
        """Return a healthy worker that ALREADY reports ``model_name`` loaded.

        Used by the llamaserver HTTP proxy path in ``coordinator/api.py``.
        Returns None when no healthy worker holds the model (caller then
        either auto-loads or 404s).
        """
        async with self._workers_lock:
            holders = {
                wid: w
                for wid, w in self.workers.items()
                if model_name in w.loaded_models and w.state == WorkerState.HEALTHY
            }
        if not holders:
            return None
        picked = await self.router.pick_worker(holders, model_name, session_id)
        if picked is not None:
            return picked
        return next(iter(holders.values()))

    async def _pick_autoload_worker(
        self, model_name: str, session_id: Optional[str] = None
    ) -> Optional[WorkerInfo]:
        """Pick a healthy worker to auto-load ``model_name`` onto.

        Prefers the router's strategy among HEALTHY workers, falling back to
        the first healthy one if the router declines them all. Returns None
        only when no healthy worker exists.
        """
        async with self._workers_lock:
            healthy = {wid: w for wid, w in self.workers.items() if w.state == WorkerState.HEALTHY}
        if not healthy:
            return None
        picked = await self.router.pick_worker(healthy, model_name, session_id)
        if picked is not None:
            return picked
        return next(iter(healthy.values()))

    async def ensure_llamaserver_model_loaded(
        self,
        model_name: str,
        session_id: Optional[str] = None,
        caller: Optional[str] = None,
    ) -> WorkerInfo:
        """Ensure ``model_name`` is loaded on a healthy worker, loading on demand.

        Single-flight per model: concurrent callers serialize on a per-model
        lock and re-check the loaded state after acquiring it, so exactly one
        load runs under a burst of requests for the same cold model.

        Returns the worker now serving the model. Raises ``RuntimeError`` when
        no healthy worker is available or the load fails (mapped to a 503).
        """
        # Fast path: already loaded somewhere (no lock — the common case).
        worker = await self.find_worker_for_model(model_name, session_id)
        if worker is not None:
            return worker

        lock = self._autoload_locks.setdefault(model_name, asyncio.Lock())
        async with lock:
            # Waiters re-check: a concurrent load may have finished while blocked.
            worker = await self.find_worker_for_model(model_name, session_id)
            if worker is not None:
                return worker

            target = await self._pick_autoload_worker(model_name, session_id)
            if target is None:
                raise RuntimeError(f"No healthy worker available to auto-load model '{model_name}'")

            loaded = await self._load_model_on_worker(target, model_name, caller=caller)
            if not loaded:
                audit.record(
                    audit.ACTION_MODEL_AUTOLOAD,
                    caller=caller or "unknown",
                    outcome=audit.OUTCOME_FAILURE,
                    model=model_name,
                    worker=target.id,
                )
                raise RuntimeError(f"Failed to load model '{model_name}' on worker {target.id}")

            audit.record(
                audit.ACTION_MODEL_AUTOLOAD,
                caller=caller or "unknown",
                outcome=audit.OUTCOME_SUCCESS,
                model=model_name,
                worker=target.id,
            )
            # Reflect the load immediately so waiters don't trigger a second
            # load; get_status() overwrites this on its next health tick.
            target.loaded_models.setdefault(model_name, pb.LoadedModelInfo(model_name=model_name))
            return target

    async def loaded_llamaserver_models(self) -> List[str]:
        """Distinct ``engine=="llamaserver"`` model names loaded on any healthy
        worker. Used for the ``/infill`` single-model fallback."""
        names: Set[str] = set()
        async with self._workers_lock:
            for worker in self.workers.values():
                if worker.state != WorkerState.HEALTHY:
                    continue
                for name in worker.loaded_models:
                    cfg = ModelRegistry.get_model(name)
                    if cfg is not None and cfg.engine == "llamaserver":
                        names.add(name)
        return sorted(names)

    async def _execute_request(self, ctx: RequestContext, worker: WorkerInfo) -> None:
        """Execute a request on a worker."""
        ctx.started_at = time.time()
        ctx.worker_id = worker.id
        worker.active_requests += 1
        worker.total_requests += 1

        try:
            # Check if model needs to be loaded
            if ctx.model_name not in worker.loaded_models:
                success = await self._load_model_on_worker(worker, ctx.model_name)
                if not success:
                    raise RuntimeError(f"Failed to load model {ctx.model_name}")

            # Prepare inference request
            request = pb.InferenceRequest(
                model_name=ctx.model_name,
                prompt=ctx.prompt,
                max_tokens=ctx.params.get("max_tokens", 512),
                temperature=ctx.params.get("temperature", 0.7),
                top_p=ctx.params.get("top_p", 0.95),
                top_k=ctx.params.get("top_k", 40),
                request_id=ctx.id,
                stream=ctx.params.get("stream", False),
            )

            # Execute inference
            response_stream = worker.stub.Infer(request)

            async for response in response_stream:
                ctx.tokens_generated = response.tokens_generated
                # Constant per request; only set when the worker actually reported one.
                if response.HasField("prompt_tokens"):
                    ctx.prompt_tokens = response.prompt_tokens

                # Accumulate text
                ctx.accumulated_text += response.text

                # Push to token queue for streaming
                await ctx.token_queue.put(response)

                if response.finished:
                    break

            ctx.completed_at = time.time()
            duration = ctx.completed_at - ctx.started_at

            # Update metrics
            self.request_counter.labels(model=ctx.model_name, status="success").inc()
            self.request_duration.labels(model=ctx.model_name).observe(duration)
            self.router.record_success(worker.id)

            logger.info(
                f"Request {ctx.id} completed: {ctx.tokens_generated} tokens " f"in {duration:.2f}s"
            )

        except Exception as e:
            ctx.error = str(e)
            ctx.completed_at = time.time()
            self.request_counter.labels(model=ctx.model_name, status="error").inc()
            self.router.record_failure(worker.id)
            logger.error(f"Request {ctx.id} failed: {e}")

        finally:
            ctx.done.set()
            worker.active_requests -= 1
            self.active_requests_gauge.set(len(self.active_requests))

    async def _refresh_worker_state(self, worker: WorkerInfo) -> None:
        """Synchronously refresh one worker's ``gpus``/``loaded_models``.

        Called right after a load/unload RPC completes so a client that just
        hit ``POST /models/load`` or ``DELETE /models/{name}`` sees consistent
        data on its very next ``GET /v1/workers`` or ``GET /v1/models`` call,
        instead of waiting up to ``health_check_interval`` for the periodic
        health-check loop to catch up (item 3). This also picks up any change
        the worker made on its own as a side effect of the load/unload (e.g.
        evicting another model to free VRAM), not just the model we touched.

        Bounded by ``worker.health_check_timeout`` — the exact timeout
        ``WorkerInfo.get_status()`` already uses for its gRPC call — so this
        can never hang the load/unload request path longer than a normal
        health check would. ``get_status()`` itself swallows/logs RPC errors
        and updates failure bookkeeping rather than raising, so a worker that
        is momentarily unreachable degrades to "stale until the next periodic
        poll" rather than failing the load/unload response.

        Deliberately never lets an exception escape: this is a best-effort
        freshness refresh, not part of the load/unload contract itself — a
        successful load/unload must still be reported as successful even if
        this immediate refresh fails for any reason (the periodic health-check
        loop remains the fallback source of truth).
        """
        try:
            await worker.get_status()
        except Exception as e:
            logger.warning(f"Failed to refresh worker {worker.id} state after load/unload: {e}")

    async def _load_model_on_worker(
        self,
        worker: WorkerInfo,
        model_name: str,
        quantization: Quantization = Quantization.NONE,
        instances: Optional[int] = None,
        caller: Optional[str] = None,
    ) -> bool:
        """Load a model on a worker.

        `instances` overrides the registry's llamaserver instance/slot count
        for this load only; ignored for non-llamaserver models.
        """
        model_config = ModelRegistry.get_model(model_name)

        # An unregistered model_name would otherwise fall through to an
        # arbitrary HuggingFace pull. Reject it here (before the try/except
        # below) unless the operator opts in, so callers get a 4xx instead
        # of a misleading 200 {"status": "failed"}.
        if model_config is None and not self.settings.allow_unregistered_model_pull:
            raise ValueError(
                f"Model '{model_name}' is not in the registry (config/models.toml) "
                "and unregistered-model pull-through is disabled. Add it to the "
                "registry, or set COORDINATOR_ALLOW_UNREGISTERED_MODEL_PULL=true "
                "to explicitly opt into loading arbitrary HuggingFace repos by name."
            )

        try:
            if model_config:
                # Use strict registry configuration if known
                config_pb = pb.ModelConfig(
                    architecture=model_config.family.value,
                    num_layers=model_config.num_layers,
                    hidden_size=model_config.hidden_size,
                    num_attention_heads=model_config.num_attention_heads,
                    num_kv_heads=model_config.num_kv_heads or 0,
                    vocab_size=model_config.vocab_size,
                    max_position_embeddings=model_config.max_seq_len,
                    intermediate_size=model_config.intermediate_size,
                    # Engine routing (empty for burn models): the worker reads
                    # these string keys to select the llama.cpp GGUF path.
                    metadata=model_config.grpc_metadata(instances=instances),
                )
                if model_config.local_gpu_ids is not None:
                    # Level 1 — local multi-GPU split: send the exact GPU ids
                    # the registry pins this model to (tensor_split metadata
                    # above tells the worker how to apportion layers).
                    gpu_ids = list(model_config.local_gpu_ids)
                else:
                    gpu_ids = [g.id for g in worker.gpus[: model_config.recommended_gpus]]
                    if not gpu_ids and worker.gpus:
                        gpu_ids = [worker.gpus[0].id]
            else:
                # If unknown, it's a HuggingFace pull.
                # The Rust worker model_loader.rs will download config.json from HF
                # and override these empty placeholder values anyway.
                config_pb = pb.ModelConfig(
                    architecture="llama",  # default fallback
                    num_layers=0,
                    hidden_size=0,
                    num_attention_heads=0,
                    num_kv_heads=0,
                    vocab_size=0,
                    max_position_embeddings=0,
                    intermediate_size=0,
                )
                gpu_ids = [g.id for g in worker.gpus]  # default to all available GPUs

            # model_path carries the resolved HF repo id; empty means
            # "model_name is already a repo id — download it directly".
            hf_repo_id = model_config.hf_repo_id if model_config else None
            request = pb.LoadModelRequest(
                model_name=model_name,
                model_path=hf_repo_id or "",
                config=config_pb,
                gpu_ids=gpu_ids,
                quantization=getattr(pb, quantization.value.upper()),
                parallelism=pb.ParallelismStrategy.AUTO,
            )

            # Deadline covers the worker's GGUF download, not just the load —
            # see Settings.model_load_timeout for the sizing rationale.
            response = await worker.stub.LoadModel(
                request, timeout=self.settings.model_load_timeout
            )

            if response.success:
                logger.info(
                    f"Loaded {model_name} on worker {worker.id}, "
                    f"using {response.memory_used / 1e9:.2f}GB VRAM"
                )
                # Refresh state synchronously instead of waiting on the
                # periodic health-check loop, so callers see it immediately.
                before = set(worker.loaded_models)
                await self._refresh_worker_state(worker)
                # The worker may have evicted another model on its own to fit
                # this one — the only place that eviction is observable.
                for evicted in before - set(worker.loaded_models):
                    audit.record(
                        audit.ACTION_MODEL_EVICTED,
                        caller=caller or "unknown",
                        outcome=audit.OUTCOME_SUCCESS,
                        model=evicted,
                        worker=worker.id,
                    )
                return True
            else:
                logger.error(f"Failed to load model: {response.message}")
                return False

        except Exception as e:
            logger.exception(f"Error loading model on worker {worker.id}: {e}")
            return False

    async def unload_model(self, model_name: str, worker_id: Optional[str] = None) -> List[str]:
        """Unload a model from one worker (worker_id given) or every worker holding it.

        Returns the worker ids it was unloaded from. Raises KeyError when the
        model is loaded nowhere (or the named worker doesn't hold/exist it).
        """
        async with self._workers_lock:
            if worker_id is not None:
                candidates = [w for w in [self.workers.get(worker_id)] if w is not None]
            else:
                candidates = list(self.workers.values())
            targets = [w for w in candidates if model_name in w.loaded_models]

        if not targets:
            raise KeyError(f"Model {model_name} is not loaded on any matching worker")

        unloaded_from: List[str] = []
        for worker in targets:
            try:
                await worker.stub.UnloadModel(
                    pb.UnloadModelRequest(model_name=model_name), timeout=60
                )
                worker.loaded_models.pop(model_name, None)
                unloaded_from.append(worker.id)
                logger.info(f"Unloaded {model_name} from worker {worker.id}")
                # Refresh synchronously so freed GPU memory isn't stale until
                # the next health tick.
                await self._refresh_worker_state(worker)
            except Exception as e:
                logger.error(f"Failed to unload {model_name} from {worker.id}: {e}")
        if not unloaded_from:
            raise RuntimeError(f"UnloadModel failed on all workers holding {model_name}")
        return unloaded_from

    async def submit_request(self, model_name: str, prompt: str, **kwargs: Any) -> RequestContext:
        """Create a request context and enqueue it. Returns immediately (streaming path)."""
        request_id = str(uuid.uuid4())
        ctx = RequestContext(
            id=request_id,
            model_name=model_name,
            prompt=prompt,
            params=kwargs,
            created_at=time.time(),
            target_worker_id=kwargs.get("worker_id"),
        )

        self.active_requests[request_id] = ctx
        self.active_requests_gauge.set(len(self.active_requests))

        try:
            await asyncio.wait_for(self.request_queue.put(request_id), timeout=5)
        except asyncio.TimeoutError as exc:
            self.active_requests.pop(request_id, None)
            self.active_requests_gauge.set(len(self.active_requests))
            raise RuntimeError("Request queue full, try again later") from exc
        return ctx

    async def infer(self, model_name: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Submit an inference request and wait for the full result."""
        ctx = await self.submit_request(model_name, prompt, **kwargs)
        timeout = kwargs.get("timeout", self.settings.request_timeout)

        try:
            await asyncio.wait_for(ctx.done.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Request {ctx.id} timed out") from exc
        finally:
            # Buffered path always cleans up here; the streaming path uses
            # submit_request() directly and cleans up in api.py's generator.
            self.active_requests.pop(ctx.id, None)
            self.active_requests_gauge.set(len(self.active_requests))

        if ctx.error:
            raise RuntimeError(ctx.error)

        return {
            "request_id": ctx.id,
            "text": ctx.accumulated_text,
            "tokens_generated": ctx.tokens_generated,
            "prompt_tokens": ctx.prompt_tokens,
            "processing_time_ms": (
                (ctx.completed_at - ctx.created_at) * 1000 if ctx.completed_at else 0.0
            ),
            "worker_id": ctx.worker_id,
        }

    async def list_workers(self) -> List[Dict[str, Any]]:
        """List all connected workers."""
        workers = []
        async with self._workers_lock:
            for worker in self.workers.values():
                workers.append(
                    {
                        "id": worker.id,
                        "address": worker.address,
                        "state": worker.state.value,
                        "gpus": [
                            {
                                "id": g.id,
                                "name": g.name,
                                "memory_gb": g.total_memory / 1e9,
                                "available_gb": g.available_memory / 1e9,
                            }
                            for g in worker.gpus
                        ],
                        "loaded_models": list(worker.loaded_models.keys()),
                        "active_requests": worker.active_requests,
                    }
                )
        return workers

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models and their load status."""
        models = []

        for model_name in ModelRegistry.list_models():
            config = ModelRegistry.get_model(model_name)
            if config is None:
                # Registry listed the name but couldn't resolve it — skip defensively.
                continue

            # Find which workers have this model loaded
            loaded_on = []
            async with self._workers_lock:
                for worker in self.workers.values():
                    if model_name in worker.loaded_models:
                        loaded_on.append(
                            {
                                "worker_id": worker.id,
                                "gpus": [g.id for g in worker.gpus],
                            }
                        )

            models.append(
                {
                    "name": model_name,
                    "family": config.family.value,
                    "parameters": config.parameters,
                    "min_memory_gb": config.min_memory_gb,
                    "loaded_on": loaded_on,
                    "supports_quantization": [q.value for q in config.supports_quantization],
                }
            )

        return models
