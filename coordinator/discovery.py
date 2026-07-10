"""Worker discovery module for finding and connecting to workers.

Implemented: static configuration (settings.static_workers + config/workers.yaml).
Planned (enum reserved, not implemented): mDNS, UDP broadcast, Consul.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

import yaml

from coordinator.config import DiscoveryMethod, Settings

logger = logging.getLogger(__name__)


@dataclass
class WorkerEndpoint:
    """Represents a discovered worker endpoint."""

    address: str  # host:port
    worker_id: Optional[str] = None
    gpu_count: int = 0
    total_memory_gb: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)


class DiscoveryProvider(ABC):
    """Base class for discovery providers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.on_worker_found: Optional[Callable[[WorkerEndpoint], Coroutine[Any, Any, None]]] = None
        self.on_worker_lost: Optional[Callable[[WorkerEndpoint], Coroutine[Any, Any, None]]] = None

    @abstractmethod
    async def start(self) -> None:
        """Start the discovery provider."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the discovery provider."""
        pass

    @abstractmethod
    async def discover(self) -> List[WorkerEndpoint]:
        """Perform a single discovery cycle."""
        pass


class StaticDiscoveryProvider(DiscoveryProvider):
    """Static discovery from configuration file."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.workers: List[WorkerEndpoint] = []
        self.config_path = Path("config/workers.yaml")
        self._watch_task: Optional["asyncio.Task[None]"] = None
        self._notify_tasks: "set[asyncio.Task[None]]" = set()

    async def start(self) -> None:
        """Start watching for config changes."""
        self._load_config()
        self._watch_task = asyncio.create_task(self._watch_config())
        logger.info(f"Static discovery started with {len(self.workers)} workers")

    async def stop(self) -> None:
        """Stop watching config."""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

    def _load_config(self) -> None:
        """Load worker configuration from settings and file."""
        new_workers: List[WorkerEndpoint] = []

        # 1. Load from settings
        for addr in self.settings.static_workers:
            new_workers.append(WorkerEndpoint(address=addr))

        # 2. Load from file if it exists (merges with settings)
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    if self.config_path.suffix == ".yaml":
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)

                for worker_config in config.get("workers", []):
                    # Avoid duplicates if already in settings
                    addr = worker_config["address"]
                    if any(w.address == addr for w in new_workers):
                        continue

                    endpoint = WorkerEndpoint(
                        address=addr,
                        worker_id=worker_config.get("id"),
                        gpu_count=worker_config.get("gpu_count", 0),
                        total_memory_gb=worker_config.get("memory_gb", 0),
                        tags=worker_config.get("tags", {}),
                    )
                    new_workers.append(endpoint)
            except Exception as e:
                logger.error(f"Failed to load worker config from {self.config_path}: {e}")

        # 3. Check for changes and notify
        old_addresses = {w.address for w in self.workers}
        new_addresses = {w.address for w in new_workers}

        # Notify about new workers
        for addr in new_addresses - old_addresses:
            worker = next((w for w in new_workers if w.address == addr), None)
            if worker and self.on_worker_found:
                task = asyncio.create_task(self.on_worker_found(worker))
                self._notify_tasks.add(task)
                task.add_done_callback(self._notify_tasks.discard)
                logger.info(f"Discovered static worker: {addr}")

        # Notify about removed workers
        for addr in old_addresses - new_addresses:
            worker = next((w for w in self.workers if w.address == addr), None)
            if worker and self.on_worker_lost:
                task = asyncio.create_task(self.on_worker_lost(worker))
                self._notify_tasks.add(task)
                task.add_done_callback(self._notify_tasks.discard)
                logger.info(f"Lost static worker: {addr}")

        self.workers = new_workers
        logger.debug(f"Static discovery updated: {len(self.workers)} workers total")

    async def _watch_config(self) -> None:
        """Watch config file for changes."""
        last_mtime = self.config_path.stat().st_mtime if self.config_path.exists() else 0

        while True:
            await asyncio.sleep(5)  # Check every 5 seconds

            try:
                if self.config_path.exists():
                    mtime = self.config_path.stat().st_mtime
                    if mtime != last_mtime:
                        logger.info("Worker config changed, reloading...")
                        self._load_config()
                        last_mtime = mtime
            except Exception as e:
                logger.error(f"Error watching config: {e}")

    async def discover(self) -> List[WorkerEndpoint]:
        """Return current static worker list."""
        return self.workers.copy()


class WorkerDiscovery:
    """Main worker discovery manager."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers: Dict[DiscoveryMethod, DiscoveryProvider] = {}
        self.workers: Dict[str, WorkerEndpoint] = {}
        self._discovery_task: Optional["asyncio.Task[None]"] = None
        self._running = False

        # Callbacks
        self.on_worker_found: Optional[Callable[[WorkerEndpoint], Coroutine[Any, Any, None]]] = None
        self.on_worker_lost: Optional[Callable[[WorkerEndpoint], Coroutine[Any, Any, None]]] = None

        # Initialize providers
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize discovery providers based on settings."""
        method = self.settings.discovery_method

        if method == DiscoveryMethod.STATIC:
            self.providers[method] = StaticDiscoveryProvider(self.settings)
        else:
            raise ValueError(
                f"Discovery method '{method.value}' is not implemented yet — "
                "only 'static' is supported. mDNS/broadcast/Consul are planned."
            )

        # Set callbacks on providers
        for provider in self.providers.values():
            provider.on_worker_found = self._on_worker_found
            provider.on_worker_lost = self._on_worker_lost

    async def start(self) -> None:
        """Start all discovery providers."""
        self._running = True

        for provider in self.providers.values():
            await provider.start()

        # Start periodic discovery
        self._discovery_task = asyncio.create_task(self._discovery_loop())

        logger.info(f"Worker discovery started with {len(self.providers)} providers")

    async def stop(self) -> None:
        """Stop all discovery providers."""
        self._running = False

        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

        for provider in self.providers.values():
            await provider.stop()

        logger.info("Worker discovery stopped")

    async def _discovery_loop(self) -> None:
        """Periodic discovery loop."""
        while self._running:
            try:
                await self.discover()
            except Exception as e:
                logger.error(f"Discovery error: {e}")

            await asyncio.sleep(self.settings.discovery_interval)

    async def discover(self) -> List[WorkerEndpoint]:
        """Run discovery on all providers."""
        all_workers: List[WorkerEndpoint] = []

        for provider in self.providers.values():
            try:
                workers = await provider.discover()
                all_workers.extend(workers)
            except Exception as e:
                logger.error(f"Discovery provider {provider.__class__.__name__} error: {e}")

        return all_workers

    async def _on_worker_found(self, worker: WorkerEndpoint) -> None:
        """Handle worker found event."""
        self.workers[worker.address] = worker
        if self.on_worker_found:
            await self.on_worker_found(worker)

    async def _on_worker_lost(self, worker: WorkerEndpoint) -> None:
        """Handle worker lost event."""
        if worker.address in self.workers:
            del self.workers[worker.address]
        if self.on_worker_lost:
            await self.on_worker_lost(worker)

    def get_worker(self, address: str) -> Optional[WorkerEndpoint]:
        """Get worker by address."""
        return self.workers.get(address)

    def get_all_workers(self) -> List[WorkerEndpoint]:
        """Get all discovered workers."""
        return list(self.workers.values())
