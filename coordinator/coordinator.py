"""Core coordinator logic for managing workers and routing requests."""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum

import grpc
from tenacity import retry, stop_after_attempt, wait_exponential

from coordinator.config import Settings
from coordinator.models import ModelRegistry, Quantization
from coordinator.discovery import WorkerDiscovery
from coordinator.monitoring import metrics
import coordinator.proto.cluster_pb2 as pb
import coordinator.proto.cluster_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker connection state."""
    CONNECTING = "connecting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
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
    
    async def get_status(self) -> Optional[pb.WorkerStatus]:
        """Get current worker status."""
        try:
            status = await self.stub.GetStatus(pb.Empty(), timeout=5)
            self.state = WorkerState.HEALTHY
            self.consecutive_failures = 0
            self.gpus = list(status.gpus)
            self.loaded_models = {
                m.model_name: m for m in status.loaded_models
            }
            self.active_requests = status.active_requests
            return status
        except Exception as e:
            self.consecutive_failures += 1
            self.last_error = str(e)
            if self.consecutive_failures >= 3:
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
        return (
            self.state == WorkerState.HEALTHY
            and self.active_requests < 10  # Configurable limit
        )


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
    error: Optional[str] = None
    tokens_generated: int = 0


class ClusterCoordinator:
    """Main coordinator that manages workers and routes requests."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.workers: Dict[str, WorkerInfo] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.max_queue_size)
        self.active_requests: Dict[str, RequestContext] = {}
        
        # Locks for thread safety
        self._workers_lock = asyncio.Lock()
        
        # Background tasks
        self._discovery_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._processor_task: Optional[asyncio.Task] = None
        
        # Components
        self.discovery = WorkerDiscovery(settings)
        
        # Metrics
        self.request_counter = metrics.counter(
            "coordinator_requests_total",
            "Total requests processed",
            ["model", "status"]
        )
        self.request_duration = metrics.histogram(
            "coordinator_request_duration_seconds",
            "Request duration in seconds",
            ["model"]
        )
        self.active_requests_gauge = metrics.gauge(
            "coordinator_active_requests",
            "Currently active requests"
        )
        
        self.is_running = False
    
    async def start(self):
        """Start the coordinator."""
        logger.info("Starting cluster coordinator...")
        self.is_running = True
        
        # Start background tasks
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._processor_task = asyncio.create_task(self._request_processor())
        
        logger.info("Coordinator started")
    
    async def stop(self):
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
        
        # Close worker connections
        async with self._workers_lock:
            for worker in self.workers.values():
                await worker.channel.close()
        
        logger.info("Coordinator stopped")
    
    async def _discovery_loop(self):
        """Background task for discovering workers."""
        while self.is_running:
            try:
                # Discover new workers
                addresses = await self.discovery.discover()
                
                for addr in addresses:
                    await self._connect_worker(addr)
                
                # Remove workers that are no longer discovered
                async with self._workers_lock:
                    discovered_set = set(addresses)
                    for worker_id in list(self.workers.keys()):
                        worker = self.workers[worker_id]
                        if worker.address not in discovered_set:
                            logger.info(f"Worker {worker_id} no longer discovered")
                            worker.state = WorkerState.OFFLINE
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
            
            await asyncio.sleep(self.settings.discovery_interval)
    
    async def _connect_worker(self, address: str) -> Optional[WorkerInfo]:
        """Connect to a worker at the given address."""
        async with self._workers_lock:
            # Check if already connected
            for worker in self.workers.values():
                if worker.address == address:
                    return worker
            
            try:
                logger.info(f"Connecting to worker at {address}")
                
                # Create gRPC channel
                channel = grpc.aio.insecure_channel(
                    address,
                    options=[
                        ('grpc.keepalive_time_ms', 10000),
                        ('grpc.keepalive_timeout_ms', 5000),
                        ('grpc.http2.max_pings_without_data', 0),
                        ('grpc.keepalive_permit_without_calls', 1),
                    ]
                )
                
                # Create stub
                stub = pb_grpc.WorkerStub(channel)
                
                # Test connection with status check
                status = await stub.GetStatus(pb.Empty(), timeout=5)
                
                worker_id = status.worker_id or f"worker-{len(self.workers)}"
                
                worker = WorkerInfo(
                    id=worker_id,
                    address=address,
                    channel=channel,
                    stub=stub,
                )
                worker.state = WorkerState.HEALTHY
                worker.gpus = list(status.gpus)
                
                self.workers[worker_id] = worker
                logger.info(f"Connected to worker {worker_id} with {len(worker.gpus)} GPUs")
                
                return worker
                
            except Exception as e:
                logger.error(f"Failed to connect to worker at {address}: {e}")
                return None
    
    async def _health_check_loop(self):
        """Background task for checking worker health."""
        while self.is_running:
            try:
                async with self._workers_lock:
                    for worker in list(self.workers.values()):
                        await worker.get_status()
                        
                        # Remove offline workers
                        if worker.state == WorkerState.OFFLINE:
                            del self.workers[worker.id]
                            await worker.channel.close()
                            logger.info(f"Removed offline worker {worker.id}")
                            
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
            
            await asyncio.sleep(self.settings.health_check_interval)
    
    async def _request_processor(self):
        """Background task for processing queued requests."""
        while self.is_running:
            try:
                # Get next request from queue
                request_id = await self.request_queue.get()
                
                if request_id not in self.active_requests:
                    continue
                
                ctx = self.active_requests[request_id]
                
                # Find suitable worker
                worker = await self._select_worker(ctx.model_name)
                
                if not worker:
                    ctx.error = "No available workers"
                    self.request_counter.labels(model=ctx.model_name, status="failed").inc()
                    continue
                
                # Execute request
                await self._execute_request(ctx, worker)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in request processor: {e}")
    
    async def _select_worker(self, model_name: str) -> Optional[WorkerInfo]:
        """Select the best worker for a request."""
        available_workers = []
        
        async with self._workers_lock:
            for worker in self.workers.values():
                if not worker.is_available:
                    continue
                
                # Check if model is loaded
                if model_name in worker.loaded_models:
                    available_workers.append(worker)
                else:
                    # Check if worker has enough memory to load model
                    model_config = ModelRegistry.get_model(model_name)
                    if model_config:
                        required_memory = model_config.min_memory_gb * 1e9
                        if worker.available_memory >= required_memory:
                            available_workers.append(worker)
        
        if not available_workers:
            return None
        
        # Simple load balancing: least active requests
        return min(available_workers, key=lambda w: w.active_requests)
    
    async def _execute_request(self, ctx: RequestContext, worker: WorkerInfo):
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
                stream=False,
            )
            
            # Execute inference
            response_stream = worker.stub.Infer(request)
            
            async for response in response_stream:
                ctx.tokens_generated = response.tokens_generated
                
                # For non-streaming, just accumulate
                if not hasattr(ctx, 'accumulated_text'):
                    ctx.accumulated_text = ""
                ctx.accumulated_text += response.text
                
                if response.finished:
                    break
            
            ctx.completed_at = time.time()
            duration = ctx.completed_at - ctx.started_at
            
            # Update metrics
            self.request_counter.labels(model=ctx.model_name, status="success").inc()
            self.request_duration.labels(model=ctx.model_name).observe(duration)
            
            logger.info(
                f"Request {ctx.id} completed: {ctx.tokens_generated} tokens "
                f"in {duration:.2f}s"
            )
            
        except Exception as e:
            ctx.error = str(e)
            ctx.completed_at = time.time()
            self.request_counter.labels(model=ctx.model_name, status="error").inc()
            logger.error(f"Request {ctx.id} failed: {e}")
            
        finally:
            worker.active_requests -= 1
            self.active_requests_gauge.set(len(self.active_requests))
    
    async def _load_model_on_worker(
        self,
        worker: WorkerInfo,
        model_name: str,
        quantization: Quantization = Quantization.FP16,
    ) -> bool:
        """Load a model on a worker."""
        try:
            model_config = ModelRegistry.get_model(model_name)
            if not model_config:
                logger.error(f"Unknown model: {model_name}")
                return False
            
            # Prepare load request
            request = pb.LoadModelRequest(
                model_name=model_name,
                model_path=str(self.settings.model_cache_dir / f"{model_name}.mpk"),
                config=pb.ModelConfig(
                    architecture=model_config.family.value,
                    num_layers=model_config.num_layers,
                    hidden_size=model_config.hidden_size,
                    num_attention_heads=model_config.num_attention_heads,
                    num_kv_heads=model_config.num_kv_heads or 0,
                    vocab_size=model_config.vocab_size,
                    max_position_embeddings=model_config.max_seq_len,
                    intermediate_size=model_config.intermediate_size,
                ),
                gpu_ids=[g.id for g in worker.gpus[:model_config.recommended_gpus]],
                quantization=getattr(pb, quantization.value.upper()),
                parallelism=pb.ParallelismStrategy.AUTO,
            )
            
            response = await worker.stub.LoadModel(request, timeout=300)  # 5 min timeout
            
            if response.success:
                logger.info(
                    f"Loaded {model_name} on worker {worker.id}, "
                    f"using {response.memory_used / 1e9:.2f}GB VRAM"
                )
                return True
            else:
                logger.error(f"Failed to load model: {response.message}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model on worker {worker.id}: {e}")
            return False
    
    async def infer(
        self,
        model_name: str,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit an inference request."""
        
        # Create request context
        request_id = str(uuid.uuid4())
        ctx = RequestContext(
            id=request_id,
            model_name=model_name,
            prompt=prompt,
            params=kwargs,
            created_at=time.time(),
        )
        
        self.active_requests[request_id] = ctx
        self.active_requests_gauge.set(len(self.active_requests))
        
        # Queue for processing
        try:
            await asyncio.wait_for(
                self.request_queue.put(request_id),
                timeout=5
            )
        except asyncio.TimeoutError:
            self.active_requests.pop(request_id, None)
            raise RuntimeError("Request queue full, try again later")
        
        # Wait for completion
        timeout = kwargs.get("timeout", self.settings.request_timeout)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if ctx.completed_at is not None:
                if ctx.error:
                    raise RuntimeError(ctx.error)
                
                return {
                    "request_id": ctx.id,
                    "text": getattr(ctx, 'accumulated_text', ''),
                    "tokens_generated": ctx.tokens_generated,
                    "processing_time_ms": (ctx.completed_at - ctx.created_at) * 1000,
                    "worker_id": ctx.worker_id,
                }
            
            await asyncio.sleep(0.1)
        
        # Timeout
        self.active_requests.pop(request_id, None)
        raise TimeoutError(f"Request {request_id} timed out")
    
    async def list_workers(self) -> List[Dict[str, Any]]:
        """List all connected workers."""
        workers = []
        async with self._workers_lock:
            for worker in self.workers.values():
                workers.append({
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
                })
        return workers
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models and their load status."""
        models = []
        
        for model_name in ModelRegistry.list_models():
            config = ModelRegistry.get_model(model_name)
            
            # Find which workers have this model loaded
            loaded_on = []
            async with self._workers_lock:
                for worker in self.workers.values():
                    if model_name in worker.loaded_models:
                        loaded_on.append({
                            "worker_id": worker.id,
                            "gpus": [g.id for g in worker.gpus],
                        })
            
            models.append({
                "name": model_name,
                "family": config.family.value,
                "parameters": config.parameters,
                "min_memory_gb": config.min_memory_gb,
                "loaded_on": loaded_on,
                "supports_quantization": [q.value for q in config.supports_quantization],
            })
        
        return models