"""Configuration management for the coordinator."""

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Union, cast

import toml
import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class DiscoveryMethod(str, Enum):
    """Worker discovery methods."""

    STATIC = "static"
    MDNS = "mdns"
    BROADCAST = "broadcast"
    CONSUL = "consul"


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_prefix="COORDINATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # the shared .env also carries worker vars (HF_TOKEN, GPU_INDEX, ...)
        json_schema_extra={
            "example": {
                "host": "0.0.0.0",
                "port": 8000,
                "discovery_method": "mdns",
                "static_workers": ["192.168.1.10:50051"],
                "health_check_interval": 30,
            }
        },
    )

    # Server settings
    host: str = Field("0.0.0.0", description="Host to bind to")
    port: int = Field(8000, description="Port to bind to", ge=1, le=65535)

    # Worker discovery
    discovery_method: DiscoveryMethod = Field(
        DiscoveryMethod.STATIC, description="How to discover workers"
    )
    static_workers: List[str] = Field(
        default_factory=list, description="Static worker addresses (host:port)"
    )
    discovery_interval: int = Field(30, description="Worker discovery interval (seconds)", ge=5)

    # Health monitoring
    health_check_interval: int = Field(30, description="Health check interval (seconds)", ge=5)
    health_check_timeout: int = Field(5, description="Health check timeout (seconds)", ge=1)
    max_failures: int = Field(
        3, description="Max consecutive failures before marking unhealthy", ge=1
    )

    # Request routing
    request_timeout: int = Field(300, description="Request timeout (seconds)", ge=1)
    max_queue_size: int = Field(1000, description="Maximum queued requests", ge=1)

    # Model management
    models_config: Path = Field(
        Path("config/models.toml"), description="Path to models configuration"
    )

    # CORS
    # NoDecode: pydantic-settings otherwise JSON-decodes any complex-typed env
    # value before our validator runs, which rejects non-JSON strings like "*".
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins. Use specific origins in production.",
    )

    # Per-worker concurrency limit
    max_concurrent_requests_per_worker: int = Field(
        10, description="Maximum concurrent requests per worker", ge=1
    )

    # Request routing (consumed by coordinator.router.RequestRouter)
    routing_strategy: str = Field(
        "least_load",
        description=(
            "Load balancing strategy: least_load, round_robin, random, affinity, power_of_two"
        ),
    )
    circuit_breaker_failure_threshold: int = Field(
        5, description="Consecutive failures before a worker's circuit opens", ge=1
    )
    circuit_breaker_recovery_timeout: float = Field(
        30.0, description="Seconds before an open circuit half-opens", gt=0
    )
    affinity_ttl_seconds: float = Field(
        600.0, description="How long a session sticks to a worker (affinity strategy)", gt=0
    )

    @field_validator("static_workers", mode="before")
    @classmethod
    def validate_static_workers(cls, v: Union[str, List[str]]) -> List[str]:
        """Validate static worker addresses."""
        if isinstance(v, str):
            # Parse comma-separated list
            return [addr.strip() for addr in v.split(",") if addr.strip()]
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Accept '*', comma-separated strings, or JSON array strings from env vars.

        NoDecode (see the field annotation) means we always receive the raw
        value here — a list already built by the caller, or the untouched
        env-var string.
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                return [str(origin).strip() for origin in parsed if str(origin).strip()]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    def load_models_config(self) -> Dict[str, Any]:
        """Load models configuration from file."""
        if not self.models_config.exists():
            return {}

        if self.models_config.suffix == ".toml":
            with open(self.models_config) as f:
                return toml.load(f)
        elif self.models_config.suffix in (".yaml", ".yml"):
            with open(self.models_config) as f:
                return cast(Dict[str, Any], yaml.safe_load(f))
        else:
            raise ValueError(f"Unsupported config format: {self.models_config.suffix}")
