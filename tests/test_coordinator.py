"""Integration and unit tests for the coordinator.

This module contains tests for:
- API endpoints
- Worker discovery
- Request routing
- Model management
- Error handling
- Performance under load
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator
from unittest.mock import Mock, patch, AsyncMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from httpx import AsyncClient
import grpc
import yaml

from coordinator.main import app, coordinator
from coordinator.coordinator import ClusterCoordinator, WorkerInfo, WorkerState
from coordinator.models import ModelRegistry, ModelConfig
from coordinator.discovery import (
    WorkerDiscovery, StaticDiscoveryProvider, 
    MDNSDiscoveryProvider, WorkerEndpoint
)
from coordinator.router import RequestRouter, LoadBalancingStrategy, QueuePriority
from coordinator.config import Settings


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(
        host="127.0.0.1",
        port=8000,
        discovery_method="static",
        static_workers=["127.0.0.1:50051"],
        health_check_interval=5,
        max_failures=2,
    )


@pytest.fixture
def mock_worker_info():
    """Create a mock worker info."""
    worker = Mock(spec=WorkerInfo)
    worker.id = "test-worker-1"
    worker.address = "127.0.0.1:50051"
    worker.state = WorkerState.HEALTHY
    worker.active_requests = 0
    worker.total_requests = 0
    worker.avg_latency_ms = 100
    worker.gpus = [
        Mock(id=0, name="Test GPU", total_memory=16*1e9, available_memory=15*1e9)
    ]
    worker.loaded_models = {}
    worker.is_available = True
    return worker


@pytest_asyncio.fixture
async def test_coordinator(test_settings):
    """Create a test coordinator instance."""
    coord = ClusterCoordinator(test_settings)
    await coord.start()
    yield coord
    await coord.stop()


@pytest_asyncio.fixture
async def async_client():
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# ============================================================================
# API Endpoint Tests
# ============================================================================

class TestAPIEndpoints:
    """Test FastAPI endpoints."""

    def test_health_check(self, client: TestClient):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "workers" in data

    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    @pytest.mark.asyncio
    async def test_list_workers(self, test_coordinator, async_client):
        """Test list workers endpoint."""
        # Add mock worker
        worker = WorkerInfo(
            id="test-1",
            address="127.0.0.1:50051",
            channel=AsyncMock(),
            stub=AsyncMock(),
            state=WorkerState.HEALTHY,
        )
        test_coordinator.workers["test-1"] = worker
        
        response = await async_client.get("/v1/workers")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "test-1"

    @pytest.mark.asyncio
    async def test_list_models(self, async_client):
        """Test list models endpoint."""
        response = await async_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "name" in data[0]
        assert "family" in data[0]

    @pytest.mark.asyncio
    async def test_load_model(self, test_coordinator, async_client, mock_worker_info):
        """Test load model endpoint."""
        # Add mock worker
        test_coordinator.workers["test-1"] = mock_worker_info
        
        # Mock successful model load
        mock_worker_info.stub.LoadModel = AsyncMock(return_value=Mock(
            success=True,
            message="Loaded",
            memory_used=8*1e9,
            loaded_on_gpus=[0],
        ))
        
        response = await async_client.post(
            "/v1/models/load",
            json={"model_name": "deepseek-7b"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "loaded"


# ============================================================================
# Coordinator Tests
# ============================================================================

class TestClusterCoordinator:
    """Test ClusterCoordinator class."""

    @pytest.mark.asyncio
    async def test_worker_discovery(self, test_coordinator):
        """Test worker discovery."""
        # Should discover static workers
        assert len(test_coordinator.workers) > 0

    @pytest.mark.asyncio
    async def test_worker_health_check(self, test_coordinator, mock_worker_info):
        """Test worker health monitoring."""
        # Add worker
        test_coordinator.workers["test-1"] = mock_worker_info
        
        # Mock successful health check
        mock_worker_info.get_status = AsyncMock(return_value=Mock(
            gpus=[Mock(id=0, available_memory=15*1e9)],
            loaded_models=[],
            active_requests=0,
        ))
        
        await test_coordinator._health_check_loop.__wrapped__(test_coordinator)
        assert mock_worker_info.state == WorkerState.HEALTHY
        assert mock_worker_info.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_worker_failure_detection(self, test_coordinator, mock_worker_info):
        """Test worker failure detection."""
        # Add worker
        test_coordinator.workers["test-1"] = mock_worker_info
        
        # Mock failing health check
        mock_worker_info.get_status = AsyncMock(side_effect=Exception("Connection failed"))
        
        # Run health check multiple times
        for _ in range(3):
            await test_coordinator._health_check_loop.__wrapped__(test_coordinator)
        
        assert mock_worker_info.state == WorkerState.UNHEALTHY
        assert mock_worker_info.consecutive_failures >= 3

    @pytest.mark.asyncio
    async def test_request_routing(self, test_coordinator, mock_worker_info):
        """Test request routing to workers."""
        # Add worker
        test_coordinator.workers["test-1"] = mock_worker_info
        
        # Mock model loaded
        mock_worker_info.loaded_models = {"deepseek-7b": Mock()}
        
        # Route request
        worker = await test_coordinator._select_worker("deepseek-7b")
        assert worker is not None
        assert worker.id == "test-1"

    @pytest.mark.asyncio
    async def test_model_loading_on_worker(self, test_coordinator, mock_worker_info):
        """Test loading model on worker."""
        # Add worker
        test_coordinator.workers["test-1"] = mock_worker_info
        
        # Mock successful load
        mock_worker_info.stub.LoadModel = AsyncMock(return_value=Mock(
            success=True,
            message="Loaded",
            memory_used=8*1e9,
            loaded_on_gpus=[0],
        ))
        
        success = await test_coordinator._load_model_on_worker(
            mock_worker_info, "deepseek-7b"
        )
        assert success is True
        mock_worker_info.stub.LoadModel.assert_called_once()


# ============================================================================
# Discovery Tests
# ============================================================================

class TestWorkerDiscovery:
    """Test worker discovery mechanisms."""

    @pytest.mark.asyncio
    async def test_static_discovery(self, test_settings):
        """Test static discovery from config."""
        provider = StaticDiscoveryProvider(test_settings)
        await provider.start()
        
        workers = await provider.discover()
        assert len(workers) > 0
        assert workers[0].address == "127.0.0.1:50051"
        
        await provider.stop()

    @pytest.mark.asyncio
    async def test_static_discovery_with_file(self, tmp_path, test_settings):
        """Test static discovery from YAML file."""
        # Create test config
        config_file = tmp_path / "workers.yaml"
        config = {
            "workers": [
                {"address": "192.168.1.10:50051", "id": "worker-1", "gpu_count": 2},
                {"address": "192.168.1.11:50051", "id": "worker-2", "gpu_count": 4},
            ]
        }
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        # Patch config path
        with patch.object(StaticDiscoveryProvider, "config_path", config_file):
            provider = StaticDiscoveryProvider(test_settings)
            await provider.start()
            
            workers = await provider.discover()
            assert len(workers) == 2
            assert workers[0].worker_id == "worker-1"
            assert workers[0].gpu_count == 2
            
            await provider.stop()

    @pytest.mark.asyncio
    async def test_broadcast_discovery(self, test_settings):
        """Test broadcast discovery."""
        provider = BroadcastDiscoveryProvider(test_settings)
        await provider.start()
        
        # Mock receiving a broadcast response
        await provider._handle_broadcast_response(
            json.dumps({
                "type": "worker_announce",
                "worker_id": "test-worker",
                "port": 50051,
                "gpus": 2,
                "memory_gb": 32,
            }).encode(),
            ("192.168.1.100", 50051)
        )
        
        workers = await provider.discover()
        assert len(workers) == 1
        assert workers[0].address == "192.168.1.100:50051"
        assert workers[0].worker_id == "test-worker"
        
        await provider.stop()


# ============================================================================
# Router Tests
# ============================================================================

class TestRequestRouter:
    """Test request routing logic."""

    @pytest.fixture
    def mock_get_workers(self, mock_worker_info):
        """Mock get_workers callback."""
        def _get_workers():
            return {"test-1": mock_worker_info}
        return _get_workers

    @pytest.mark.asyncio
    async def test_round_robin_routing(self, mock_get_workers, test_settings):
        """Test round-robin load balancing."""
        router = RequestRouter(mock_get_workers, test_settings)
        router.strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        # Create multiple workers
        workers = {
            "worker-1": Mock(id="worker-1", is_available=True, loaded_models={}),
            "worker-2": Mock(id="worker-2", is_available=True, loaded_models={}),
        }
        
        def mock_get():
            return workers
        
        router.get_workers = mock_get
        
        # Route multiple requests
        selected = set()
        for i in range(4):
            worker, _ = await router.route_request(f"req-{i}", "test-model", "prompt", {})
            selected.add(worker.id)
        
        # Should have used both workers
        assert len(selected) == 2

    @pytest.mark.asyncio
    async def test_least_load_routing(self, mock_get_workers, test_settings):
        """Test least-load routing."""
        router = RequestRouter(mock_get_workers, test_settings)
        router.strategy = LoadBalancingStrategy.LEAST_LOAD
        
        # Create workers with different loads
        workers = {
            "worker-1": Mock(id="worker-1", is_available=True, loaded_models={}, active_requests=5),
            "worker-2": Mock(id="worker-2", is_available=True, loaded_models={}, active_requests=1),
        }
        
        def mock_get():
            return workers
        
        router.get_workers = mock_get
        
        # Should pick least loaded worker
        worker, _ = await router.route_request("req-1", "test-model", "prompt", {})
        assert worker.id == "worker-2"

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, mock_get_workers, test_settings):
        """Test circuit breaker functionality."""
        router = RequestRouter(mock_get_workers, test_settings)
        
        # Record failures
        for _ in range(5):
            router.record_failure("test-1")
        
        # Circuit should be open
        assert router.circuit_breakers["test-1"].state.value == "open"
        
        # Request should be rejected
        worker, queued = await router.route_request("req-1", "test-model", "prompt", {})
        assert worker is None
        assert queued is not None

    @pytest.mark.asyncio
    async def test_request_queue(self, mock_get_workers, test_settings):
        """Test request queuing."""
        router = RequestRouter(mock_get_workers, test_settings)
        await router.start()
        
        # Queue a request
        worker, queued = await router.route_request(
            "req-1", "test-model", "prompt", {},
            priority=QueuePriority.HIGH
        )
        
        assert worker is None
        assert queued is not None
        assert queued.priority == QueuePriority.HIGH
        
        # Check queue stats
        stats = router.get_queue_stats()
        assert stats["queues"]["2"]["size"] == 1  # HIGH priority
        
        await router.stop()

    @pytest.mark.asyncio
    async def test_affinity_routing(self, mock_get_workers, test_settings):
        """Test affinity-based routing."""
        router = RequestRouter(mock_get_workers, test_settings)
        router.strategy = LoadBalancingStrategy.AFFINITY
        
        workers = {
            "worker-1": Mock(id="worker-1", is_available=True, loaded_models={}),
            "worker-2": Mock(id="worker-2", is_available=True, loaded_models={}),
        }
        
        def mock_get():
            return workers
        
        router.get_workers = mock_get
        
        # Same request ID should route to same worker
        worker1, _ = await router.route_request("same-id", "test-model", "prompt", {})
        worker2, _ = await router.route_request("same-id", "test-model", "prompt", {})
        
        assert worker1.id == worker2.id


# ============================================================================
# Model Registry Tests
# ============================================================================

class TestModelRegistry:
    """Test model registry."""

    def test_get_model(self):
        """Test getting model config."""
        model = ModelRegistry.get_model("deepseek-7b")
        assert model is not None
        assert model.name == "deepseek-7b"
        assert model.family.value == "deepseek"
        assert model.parameters == "7B"

    def test_list_models(self):
        """Test listing all models."""
        models = ModelRegistry.list_models()
        assert len(models) > 0
        assert "deepseek-7b" in models
        assert "llama3-8b" in models

    def test_find_by_family(self):
        """Test finding models by family."""
        deepseek_models = ModelRegistry.find_models_by_family("deepseek")
        assert len(deepseek_models) > 0
        assert all(m.family.value == "deepseek" for m in deepseek_models)

    def test_validate_requirements(self):
        """Test hardware requirement validation."""
        # Should pass with enough memory
        valid, msg = ModelRegistry.validate_requirements(
            "deepseek-7b", 32*1e9, 1
        )
        assert valid is True
        
        # Should fail with insufficient memory
        valid, msg = ModelRegistry.validate_requirements(
            "deepseek-7b", 8*1e9, 1
        )
        assert valid is False
        assert "Insufficient memory" in msg


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.asyncio
    async def test_request_latency(self, test_coordinator, mock_worker_info):
        """Test request latency under normal conditions."""
        # Add worker
        test_coordinator.workers["test-1"] = mock_worker_info
        
        # Mock inference
        mock_worker_info.stub.Infer = AsyncMock(return_value=AsyncMock(
            __aiter__=lambda: iter([Mock(
                text="test",
                tokens_generated=10,
                finished=True,
            )])
        ))
        
        start = time.time()
        result = await test_coordinator.infer(
            "deepseek-7b",
            "Test prompt",
            max_tokens=10,
        )
        latency = (time.time() - start) * 1000
        
        assert result is not None
        assert latency < 100  # Should be under 100ms for mock

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_coordinator, mock_worker_info):
        """Test handling multiple concurrent requests."""
        # Add worker
        test_coordinator.workers["test-1"] = mock_worker_info
        
        # Mock inference
        mock_worker_info.stub.Infer = AsyncMock(return_value=AsyncMock(
            __aiter__=lambda: iter([Mock(
                text="test",
                tokens_generated=10,
                finished=True,
            )])
        ))
        
        # Fire off multiple concurrent requests
        tasks = []
        for i in range(10):
            tasks.append(test_coordinator.infer(
                "deepseek-7b",
                f"Test prompt {i}",
                max_tokens=10,
            ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(not isinstance(r, Exception) for r in results)
        assert len(results) == 10


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_model_not_found(self, test_coordinator):
        """Test handling of unknown model."""
        with pytest.raises(Exception) as exc:
            await test_coordinator.infer(
                "non-existent-model",
                "Test prompt",
            )
        assert "Unknown model" in str(exc.value)

    @pytest.mark.asyncio
    async def test_no_workers_available(self, test_coordinator):
        """Test handling when no workers are available."""
        # Clear workers
        test_coordinator.workers.clear()
        
        with pytest.raises(Exception) as exc:
            await test_coordinator.infer(
                "deepseek-7b",
                "Test prompt",
            )
        assert "No available workers" in str(exc.value)

    @pytest.mark.asyncio
    async def test_request_timeout(self, test_coordinator, mock_worker_info):
        """Test request timeout handling."""
        # Add worker
        test_coordinator.workers["test-1"] = mock_worker_info
        
        # Mock slow inference
        async def slow_infer(*args, **kwargs):
            await asyncio.sleep(5)
            return Mock(text="test", tokens_generated=10, finished=True)
        
        mock_worker_info.stub.Infer = AsyncMock(return_value=AsyncMock(
            __aiter__=lambda: iter([await slow_infer()])
        ))
        
        with pytest.raises(TimeoutError):
            await test_coordinator.infer(
                "deepseek-7b",
                "Test prompt",
                timeout=1,  # Short timeout
            )


# ============================================================================
# WebSocket Tests (if applicable)
# ============================================================================

@pytest.mark.asyncio
async def test_websocket_streaming(client: TestClient):
    """Test WebSocket streaming endpoint."""
    with client.websocket_connect("/v1/stream") as websocket:
        # Send request
        websocket.send_json({
            "model": "deepseek-7b",
            "prompt": "Test prompt",
            "stream": True,
        })
        
        # Receive streaming responses
        responses = []
        for _ in range(5):
            data = websocket.receive_json()
            responses.append(data)
        
        assert len(responses) > 0
        websocket.close()


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_inference_flow(self, test_coordinator, mock_worker_info):
        """Test complete inference flow from request to response."""
        # Setup
        test_coordinator.workers["test-1"] = mock_worker_info
        
        # Mock model load
        mock_worker_info.stub.LoadModel = AsyncMock(return_value=Mock(
            success=True,
            message="Loaded",
            memory_used=8*1e9,
            loaded_on_gpus=[0],
        ))
        
        # Mock inference
        mock_worker_info.stub.Infer = AsyncMock(return_value=AsyncMock(
            __aiter__=lambda: iter([
                Mock(text="This ", tokens_generated=1, finished=False),
                Mock(text="is ", tokens_generated=2, finished=False),
                Mock(text="a ", tokens_generated=3, finished=False),
                Mock(text="test", tokens_generated=4, finished=True),
            ])
        ))
        
        # Execute
        result = await test_coordinator.infer(
            "deepseek-7b",
            "Test prompt",
            max_tokens=10,
        )
        
        # Verify
        assert result is not None
        assert result["text"] == "This is a test"
        assert result["tokens_generated"] == 4
        assert "processing_time_ms" in result
        assert "worker_id" in result


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Test configuration loading and validation."""

    def test_settings_from_env(self, monkeypatch):
        """Test loading settings from environment."""
        monkeypatch.setenv("COORDINATOR_HOST", "192.168.1.100")
        monkeypatch.setenv("COORDINATOR_PORT", "9000")
        monkeypatch.setenv("DISCOVERY_METHOD", "mdns")
        
        settings = Settings()
        assert settings.host == "192.168.1.100"
        assert settings.port == 9000
        assert settings.discovery_method == "mdns"

    def test_static_workers_parsing(self):
        """Test parsing static worker list."""
        settings = Settings(static_workers="192.168.1.10:50051,192.168.1.11:50051")
        assert len(settings.static_workers) == 2
        assert "192.168.1.10:50051" in settings.static_workers

    def test_invalid_port(self):
        """Test port validation."""
        with pytest.raises(ValueError):
            Settings(port=99999)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=coordinator"])