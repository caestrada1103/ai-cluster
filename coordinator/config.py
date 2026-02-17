"""Configuration management for the coordinator."""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
import toml


class DiscoveryMethod(str, Enum):
    """Worker discovery methods."""
    STATIC = "static"
    MDNS = "mdns"
    BROADCAST = "broadcast"
    CONSUL = "consul"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="COORDINATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Server settings
    host: str = Field("0.0.0.0", description="Host to bind to")
    port: int = Field(8000, description="Port to bind to", ge=1, le=65535)
    
    # Worker discovery
    discovery_method: DiscoveryMethod = Field(
        DiscoveryMethod.STATIC, description="How to discover workers"
    )
    static_workers: List[str] = Field(
        default_factory=list,
        description="Static worker addresses (host:port)"
    )
    discovery_interval: int = Field(
        30, description="Worker discovery interval (seconds)", ge=5
    )
    
    # Health monitoring
    health_check_interval: int = Field(
        30, description="Health check interval (seconds)", ge=5
    )
    health_check_timeout: int = Field(
        5, description="Health check timeout (seconds)", ge=1
    )
    max_failures: int = Field(
        3, description="Max consecutive failures before marking unhealthy", ge=1
    )
    
    # Request routing
    default_model: str = Field(
        "deepseek-7b", description="Default model for inference"
    )
    request_timeout: int = Field(
        60, description="Request timeout (seconds)", ge=1
    )
    max_queue_size: int = Field(
        1000, description="Maximum queued requests", ge=1
    )
    
    # Model management
    models_config: Path = Field(
        Path("config/models.toml"), description="Path to models configuration"
    )
    model_cache_dir: Path = Field(
        Path("./models"), description="Directory for cached models"
    )
    auto_load_models: bool = Field(
        False, description="Automatically load models on startup"
    )
    
    # Performance
    enable_batching: bool = Field(
        True, description="Enable continuous batching"
    )
    max_batch_size: int = Field(
        32, description="Maximum batch size", ge=1, le=256
    )
    batch_timeout_ms: int = Field(
        50, description="Maximum wait time for batching (ms)", ge=1, le=1000
    )
    
    # Security
    enable_auth: bool = Field(False, description="Enable API authentication")
    api_keys: List[str] = Field(
        default_factory=list, description="Valid API keys"
    )
    rate_limit_per_minute: int = Field(
        60, description="Rate limit per API key (requests/minute)", ge=1
    )
    
    # Logging
    log_level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    log_format: str = Field(
        "json", description="Log format (json or text)", pattern="^(json|text)$"
    )
    
    # Monitoring
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    metrics_port: int = Field(9090, description="Metrics port", ge=1, le=65535)
    
    @field_validator("static_workers", mode="before")
    @classmethod
    def validate_static_workers(cls, v: Union[str, List[str]]) -> List[str]:
        """Validate static worker addresses."""
        if isinstance(v, str):
            # Parse comma-separated list
            return [addr.strip() for addr in v.split(",") if addr.strip()]
        return v
    
    def load_models_config(self) -> Dict:
        """Load models configuration from file."""
        if not self.models_config.exists():
            return {}
        
        if self.models_config.suffix == ".toml":
            with open(self.models_config) as f:
                return toml.load(f)
        elif self.models_config.suffix in (".yaml", ".yml"):
            with open(self.models_config) as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {self.models_config.suffix}")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "host": "0.0.0.0",
                "port": 8000,
                "discovery_method": "mdns",
                "static_workers": ["192.168.1.10:50051"],
                "health_check_interval": 30,
                "default_model": "deepseek-7b",
            }
        }