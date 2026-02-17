"""Worker discovery module for finding and connecting to workers.

This module provides multiple discovery methods:
- Static configuration
- mDNS (Bonjour/Avahi)
- Broadcast discovery
- Consul service discovery
"""

import asyncio
import ipaddress
import json
import logging
import socket
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Awaitable
from enum import Enum
from pathlib import Path
import time

import yaml
import aiohttp

from coordinator.config import Settings, DiscoveryMethod

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
    
    @property
    def host(self) -> str:
        return self.address.split(":")[0]
    
    @property
    def port(self) -> int:
        parts = self.address.split(":")
        return int(parts[1]) if len(parts) > 1 else 50051


class DiscoveryProvider(ABC):
    """Base class for discovery providers."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.on_worker_found: Optional[Callable[[WorkerEndpoint], Awaitable[None]]] = None
        self.on_worker_lost: Optional[Callable[[WorkerEndpoint], Awaitable[None]]] = None
    
    @abstractmethod
    async def start(self):
        """Start the discovery provider."""
        pass
    
    @abstractmethod
    async def stop(self):
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
        self._watch_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start watching for config changes."""
        self._load_config()
        self._watch_task = asyncio.create_task(self._watch_config())
        logger.info(f"Static discovery started with {len(self.workers)} workers")
    
    async def stop(self):
        """Stop watching config."""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
    
    def _load_config(self):
        """Load worker configuration from file."""
        if not self.config_path.exists():
            # Use static workers from settings
            self.workers = [
                WorkerEndpoint(address=addr)
                for addr in self.settings.static_workers
            ]
            return
        
        try:
            with open(self.config_path) as f:
                if self.config_path.suffix == ".yaml":
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            new_workers = []
            for worker_config in config.get("workers", []):
                endpoint = WorkerEndpoint(
                    address=worker_config["address"],
                    worker_id=worker_config.get("id"),
                    gpu_count=worker_config.get("gpu_count", 0),
                    total_memory_gb=worker_config.get("memory_gb", 0),
                    tags=worker_config.get("tags", {}),
                )
                new_workers.append(endpoint)
            
            # Check for changes
            old_addresses = {w.address for w in self.workers}
            new_addresses = {w.address for w in new_workers}
            
            self.workers = new_workers
            
            # Notify about new workers
            for addr in new_addresses - old_addresses:
                worker = next(w for w in new_workers if w.address == addr)
                if self.on_worker_found:
                    asyncio.create_task(self.on_worker_found(worker))
            
            # Notify about lost workers
            for addr in old_addresses - new_addresses:
                worker = WorkerEndpoint(address=addr)  # Minimal info
                if self.on_worker_lost:
                    asyncio.create_task(self.on_worker_lost(worker))
            
            logger.debug(f"Loaded {len(self.workers)} workers from config")
            
        except Exception as e:
            logger.error(f"Failed to load worker config: {e}")
    
    async def _watch_config(self):
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


class MDNSDiscoveryProvider(DiscoveryProvider):
    """mDNS (Bonjour/Avahi) discovery provider."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.service_name = settings.discovery.mdns.service_name
        self.known_workers: Dict[str, WorkerEndpoint] = {}
        self._browser = None
        self._running = False
        
        # Try to import zeroconf
        try:
            from zeroconf import Zeroconf, ServiceBrowser, ServiceListener
            self.zeroconf = Zeroconf
            self.service_browser = ServiceBrowser
            self.service_listener = ServiceListener
            self._import_success = True
        except ImportError:
            logger.warning("zeroconf not installed, mDNS discovery disabled")
            self._import_success = False
    
    class Listener:
        """mDNS service listener."""
        
        def __init__(self, provider: 'MDNSDiscoveryProvider'):
            self.provider = provider
        
        def add_service(self, zc, type_, name):
            """Service added."""
            asyncio.create_task(self.provider._handle_service_added(zc, type_, name))
        
        def remove_service(self, zc, type_, name):
            """Service removed."""
            asyncio.create_task(self.provider._handle_service_removed(name))
        
        def update_service(self, zc, type_, name):
            """Service updated."""
            asyncio.create_task(self.provider._handle_service_updated(zc, type_, name))
    
    async def start(self):
        """Start mDNS browsing."""
        if not self._import_success:
            return
        
        try:
            from zeroconf import Zeroconf
            
            self.zc = Zeroconf()
            self.listener = self.Listener(self)
            self.browser = self.service_browser(
                self.zc, self.service_name, self.listener
            )
            self._running = True
            logger.info(f"mDNS discovery started for {self.service_name}")
        except Exception as e:
            logger.error(f"Failed to start mDNS discovery: {e}")
    
    async def stop(self):
        """Stop mDNS browsing."""
        self._running = False
        if hasattr(self, 'zc'):
            self.zc.close()
    
    async def _handle_service_added(self, zc, type_, name):
        """Handle new service discovery."""
        try:
            info = zc.get_service_info(type_, name)
            if info:
                address = socket.inet_ntoa(info.addresses[0])
                port = info.port
                
                # Parse TXT records
                tags = {}
                if info.properties:
                    for key, value in info.properties.items():
                        if isinstance(key, bytes):
                            key = key.decode('utf-8')
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        tags[key] = value
                
                endpoint = WorkerEndpoint(
                    address=f"{address}:{port}",
                    worker_id=tags.get("worker_id"),
                    gpu_count=int(tags.get("gpus", 0)),
                    total_memory_gb=float(tags.get("memory_gb", 0)),
                    tags=tags,
                )
                
                self.known_workers[endpoint.address] = endpoint
                
                if self.on_worker_found:
                    await self.on_worker_found(endpoint)
                    
                logger.debug(f"Discovered worker via mDNS: {endpoint.address}")
                
        except Exception as e:
            logger.error(f"Error handling mDNS service add: {e}")
    
    async def _handle_service_removed(self, name):
        """Handle service removal."""
        # Find endpoint by service name (difficult without mapping)
        # For now, we'll rely on periodic discovery to detect removals
        pass
    
    async def _handle_service_updated(self, zc, type_, name):
        """Handle service update."""
        await self._handle_service_added(zc, type_, name)
    
    async def discover(self) -> List[WorkerEndpoint]:
        """Return currently discovered workers."""
        return list(self.known_workers.values())


class BroadcastDiscoveryProvider(DiscoveryProvider):
    """UDP broadcast discovery provider."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.port = settings.discovery.broadcast.port
        self.interface = settings.discovery.broadcast.interface
        self.broadcast_address = settings.discovery.broadcast.broadcast_address
        self.known_workers: Dict[str, WorkerEndpoint] = {}
        self._running = False
        self._server_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start broadcast discovery."""
        self._running = True
        
        # Start UDP server to listen for responses
        self._server_task = asyncio.create_task(self._udp_server())
        
        # Start periodic broadcasting
        self._broadcast_task = asyncio.create_task(self._periodic_broadcast())
        
        logger.info(f"Broadcast discovery started on port {self.port}")
    
    async def stop(self):
        """Stop broadcast discovery."""
        self._running = False
        
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
    
    async def _udp_server(self):
        """UDP server to listen for worker responses."""
        loop = asyncio.get_running_loop()
        
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to interface if specified
        if self.interface:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
                           self.interface.encode())
        
        sock.bind(('', self.port))
        sock.setblocking(False)
        
        try:
            while self._running:
                try:
                    data, addr = await loop.sock_recvfrom(sock, 1024)
                    await self._handle_broadcast_response(data, addr)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"UDP server error: {e}")
        finally:
            sock.close()
    
    async def _handle_broadcast_response(self, data: bytes, addr: Tuple[str, int]):
        """Handle response from worker."""
        try:
            message = json.loads(data.decode())
            
            if message.get("type") == "worker_announce":
                endpoint = WorkerEndpoint(
                    address=f"{addr[0]}:{message.get('port', 50051)}",
                    worker_id=message.get("worker_id"),
                    gpu_count=message.get("gpus", 0),
                    total_memory_gb=message.get("memory_gb", 0),
                    tags=message.get("tags", {}),
                )
                
                self.known_workers[endpoint.address] = endpoint
                
                if self.on_worker_found:
                    await self.on_worker_found(endpoint)
                    
                logger.debug(f"Discovered worker via broadcast: {endpoint.address}")
                
        except Exception as e:
            logger.error(f"Error handling broadcast response: {e}")
    
    async def _periodic_broadcast(self):
        """Periodically broadcast discovery requests."""
        while self._running:
            try:
                await self._broadcast_discovery()
                await asyncio.sleep(self.settings.discovery_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
    
    async def _broadcast_discovery(self):
        """Send UDP broadcast discovery request."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Bind to interface if specified
        if self.interface:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
                           self.interface.encode())
        
        message = {
            "type": "discovery_request",
            "coordinator_id": socket.gethostname(),
            "timestamp": time.time(),
        }
        
        try:
            sock.sendto(
                json.dumps(message).encode(),
                (self.broadcast_address, self.port)
            )
            logger.debug(f"Broadcast discovery request sent to {self.broadcast_address}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to send broadcast: {e}")
        finally:
            sock.close()
    
    async def discover(self) -> List[WorkerEndpoint]:
        """Return currently discovered workers."""
        # Clean up stale workers (older than 3 intervals)
        stale_threshold = time.time() - (self.settings.discovery_interval * 3)
        self.known_workers = {
            addr: worker
            for addr, worker in self.known_workers.items()
            if worker.last_seen > stale_threshold
        }
        
        return list(self.known_workers.values())


class ConsulDiscoveryProvider(DiscoveryProvider):
    """Consul service discovery provider."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.host = settings.discovery.consul.host
        self.port = settings.discovery.consul.port
        self.service_name = settings.discovery.consul.service_name
        self.datacenter = settings.discovery.consul.datacenter
        self.session = None
        self.known_workers: Dict[str, WorkerEndpoint] = {}
        self._watch_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start Consul discovery."""
        self.session = aiohttp.ClientSession()
        self._watch_task = asyncio.create_task(self._watch_services())
        logger.info(f"Consul discovery started for {self.service_name}")
    
    async def stop(self):
        """Stop Consul discovery."""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
    
    async def _watch_services(self):
        """Watch for service changes in Consul."""
        last_index = 0
        
        while True:
            try:
                # Long-poll for changes
                url = f"http://{self.host}:{self.port}/v1/health/service/{self.service_name}"
                params = {
                    "index": last_index,
                    "wait": "60s",
                    "dc": self.datacenter,
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Update index for next request
                        last_index = int(response.headers.get("X-Consul-Index", last_index))
                        
                        # Process services
                        await self._process_services(data)
                    
                    elif response.status != 404:
                        logger.error(f"Consul API error: {response.status}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consul watch error: {e}")
                await asyncio.sleep(5)
    
    async def _process_services(self, services: List[Dict]):
        """Process service entries from Consul."""
        current_addresses = set()
        
        for service in services:
            # Check if service is healthy
            if not self._is_service_healthy(service):
                continue
            
            # Extract service info
            node = service.get("Node", {})
            svc = service.get("Service", {})
            
            address = node.get("Address")
            port = svc.get("Port", 50051)
            tags = svc.get("Tags", [])
            
            # Parse tags into dict
            tag_dict = {}
            for tag in tags:
                if "=" in tag:
                    key, value = tag.split("=", 1)
                    tag_dict[key] = value
            
            endpoint = WorkerEndpoint(
                address=f"{address}:{port}",
                worker_id=svc.get("ID"),
                gpu_count=int(tag_dict.get("gpus", 0)),
                total_memory_gb=float(tag_dict.get("memory_gb", 0)),
                tags=tag_dict,
            )
            
            current_addresses.add(endpoint.address)
            
            # Check if new worker
            if endpoint.address not in self.known_workers:
                self.known_workers[endpoint.address] = endpoint
                if self.on_worker_found:
                    await self.on_worker_found(endpoint)
                    logger.info(f"Discovered worker via Consul: {endpoint.address}")
        
        # Check for removed workers
        for addr in list(self.known_workers.keys()):
            if addr not in current_addresses:
                worker = self.known_workers.pop(addr)
                if self.on_worker_lost:
                    await self.on_worker_lost(worker)
                    logger.info(f"Worker removed from Consul: {addr}")
    
    def _is_service_healthy(self, service: Dict) -> bool:
        """Check if service is healthy."""
        # Check node health
        for check in service.get("Checks", []):
            if check.get("Status") != "passing":
                return False
        
        # Check service health
        for check in service.get("ServiceChecks", []):
            if check.get("Status") != "passing":
                return False
        
        return True
    
    async def discover(self) -> List[WorkerEndpoint]:
        """Return currently discovered workers."""
        return list(self.known_workers.values())


class WorkerDiscovery:
    """Main worker discovery manager."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers: Dict[DiscoveryMethod, DiscoveryProvider] = {}
        self.workers: Dict[str, WorkerEndpoint] = {}
        self._discovery_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Callbacks
        self.on_worker_found: Optional[Callable[[WorkerEndpoint], Awaitable[None]]] = None
        self.on_worker_lost: Optional[Callable[[WorkerEndpoint], Awaitable[None]]] = None
        
        # Initialize providers
        self._init_providers()
    
    def _init_providers(self):
        """Initialize discovery providers based on settings."""
        method = self.settings.discovery_method
        
        if method == DiscoveryMethod.STATIC or method == "static":
            self.providers[method] = StaticDiscoveryProvider(self.settings)
        
        elif method == DiscoveryMethod.MDNS or method == "mdns":
            self.providers[method] = MDNSDiscoveryProvider(self.settings)
        
        elif method == DiscoveryMethod.BROADCAST or method == "broadcast":
            self.providers[method] = BroadcastDiscoveryProvider(self.settings)
        
        elif method == DiscoveryMethod.CONSUL or method == "consul":
            self.providers[method] = ConsulDiscoveryProvider(self.settings)
        
        else:
            # Use multiple providers
            if DiscoveryMethod.STATIC in method or "static" in method:
                self.providers[DiscoveryMethod.STATIC] = StaticDiscoveryProvider(self.settings)
            if DiscoveryMethod.MDNS in method or "mdns" in method:
                self.providers[DiscoveryMethod.MDNS] = MDNSDiscoveryProvider(self.settings)
            if DiscoveryMethod.BROADCAST in method or "broadcast" in method:
                self.providers[DiscoveryMethod.BROADCAST] = BroadcastDiscoveryProvider(self.settings)
            if DiscoveryMethod.CONSUL in method or "consul" in method:
                self.providers[DiscoveryMethod.CONSUL] = ConsulDiscoveryProvider(self.settings)
        
        # Set callbacks on providers
        for provider in self.providers.values():
            provider.on_worker_found = self._on_worker_found
            provider.on_worker_lost = self._on_worker_lost
    
    async def start(self):
        """Start all discovery providers."""
        self._running = True
        
        for provider in self.providers.values():
            await provider.start()
        
        # Start periodic discovery
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        
        logger.info(f"Worker discovery started with {len(self.providers)} providers")
    
    async def stop(self):
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
    
    async def _discovery_loop(self):
        """Periodic discovery loop."""
        while self._running:
            try:
                await self.discover()
            except Exception as e:
                logger.error(f"Discovery error: {e}")
            
            await asyncio.sleep(self.settings.discovery_interval)
    
    async def discover(self) -> List[WorkerEndpoint]:
        """Run discovery on all providers."""
        all_workers = []
        
        for provider in self.providers.values():
            try:
                workers = await provider.discover()
                all_workers.extend(workers)
            except Exception as e:
                logger.error(f"Discovery provider {provider.__class__.__name__} error: {e}")
        
        return all_workers
    
    async def _on_worker_found(self, worker: WorkerEndpoint):
        """Handle worker found event."""
        self.workers[worker.address] = worker
        if self.on_worker_found:
            await self.on_worker_found(worker)
    
    async def _on_worker_lost(self, worker: WorkerEndpoint):
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
    
    def get_workers_by_tag(self, key: str, value: str) -> List[WorkerEndpoint]:
        """Get workers with specific tag."""
        return [
            w for w in self.workers.values()
            if w.tags.get(key) == value
        ]