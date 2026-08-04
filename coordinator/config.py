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
    model_load_timeout: int = Field(
        3600,
        description=(
            "LoadModel gRPC deadline (seconds). Must exceed the worker's "
            "GGUF download time (weight_size / link_speed)."
        ),
        ge=1,
    )

    # CORS: defaults to no explicit origins, but main.py always applies a
    # loopback allow_origin_regex, so local browser calls work with zero
    # config while any other origin needs an explicit opt-in. See
    # docs/configuration.md.
    # NoDecode: pydantic-settings would otherwise JSON-decode this before our
    # validator runs, rejecting non-JSON strings like "*".
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=list,
        description=(
            "Allowed CORS origins beyond the always-on loopback default. "
            "Empty by default; set to '*' or specific origins to widen."
        ),
    )

    # Per-worker concurrency limit
    max_concurrent_requests_per_worker: int = Field(
        10, description="Maximum concurrent requests per worker", ge=1
    )

    # Disabled by default: a maliciously registered "worker" here can be
    # routed real traffic or used for SSRF. See docs/configuration.md.
    allow_manual_worker_registration: bool = Field(
        False,
        description=(
            "Enable POST /v1/workers/manual (env "
            "COORDINATOR_ALLOW_MANUAL_WORKER_REGISTRATION). Off by default; the route "
            "also independently requires a valid COORDINATOR_API_KEYS credential "
            "regardless of whether global auth is enabled, and addresses are "
            "shape-validated and capped."
        ),
    )
    manual_worker_allowed_hosts: Annotated[List[str], NoDecode] = Field(
        default_factory=list,
        description=(
            "Optional allowlist of hosts/CIDRs POST /v1/workers/manual may register "
            "(comma-separated). Empty (default) means any well-formed host:port is "
            "accepted once the feature is enabled and authenticated."
        ),
    )

    # Off by default: an unregistered model_name would otherwise become an
    # arbitrary HuggingFace repo pull.
    allow_unregistered_model_pull: bool = Field(
        False,
        description=(
            "Allow POST /v1/models/load for a model_name absent from the registry "
            "(env COORDINATOR_ALLOW_UNREGISTERED_MODEL_PULL) — the worker treats it "
            "as an arbitrary HuggingFace repo id to download. Off by default."
        ),
    )

    # 25 MB is generous for a real chat/agentic request while still
    # bounding worst-case memory use. See coordinator/body_limit.py.
    max_request_body_bytes: int = Field(
        25_000_000,
        description=(
            "Maximum HTTP request body size in bytes (env COORDINATOR_MAX_REQUEST_BODY_BYTES)"
        ),
        ge=1,
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

    # When True (default), a proxied request for an unloaded llamaserver
    # model triggers a load first. When False it 404s instead.
    llamaserver_autoload: bool = Field(
        True,
        description=(
            "Auto-load an unloaded engine=llamaserver model on demand before proxying "
            "(env COORDINATOR_LLAMASERVER_AUTOLOAD)"
        ),
    )

    # Context compression middleware (coordinator/context_compression/). Off
    # by default; only applies when estimated prompt tokens exceed the
    # budget. Tune the budget to roughly `n_ctx - max_tokens - 512`.
    context_compression_enabled: bool = Field(
        False, description="Server-wide default: compress oversized prompts before forwarding"
    )
    context_compression_token_budget: int = Field(
        8192,
        description="Estimated-token threshold that triggers compression",
        ge=1,
    )
    context_compression_safety_margin: float = Field(
        0.20,
        description="Fractional inflation applied to the token estimate before comparing to budget",
        ge=0.0,
        le=1.0,
    )
    context_compression_active_segments: int = Field(
        1,
        description=(
            "Most-recent code segments kept fully intact (never compressed); 0 protects none"
        ),
        ge=0,
    )
    context_compression_techniques: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["skeletonize"],
        description=(
            "Ordered techniques to try in priority order: skeletonize, summarize, llmlingua"
        ),
    )
    context_compression_summarizer_model: str = Field(
        "qwen2.5-0.5b-gguf",
        description="Registry model name used to summarize NL history (Phase 2)",
    )
    context_compression_summarizer_max_tokens: int = Field(
        256, description="max_tokens budget for each summarizer sub-call", ge=16
    )
    context_compression_llmlingua_enabled: bool = Field(
        False, description="Opt-in LLMLingua token compression for NL segments only (Phase 3)"
    )
    context_compression_llmlingua_model: str = Field(
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        description=(
            "LLMLingua compressor model (downloaded from HF on first use unless pre-cached)"
        ),
    )
    context_compression_llmlingua_rate: float = Field(
        0.5,
        description="LLMLingua target compression rate (fraction of tokens kept)",
        gt=0.0,
        lt=1.0,
    )

    @field_validator("static_workers", mode="before")
    @classmethod
    def validate_static_workers(cls, v: Union[str, List[str]]) -> List[str]:
        """Validate static worker addresses."""
        if isinstance(v, str):
            # Parse comma-separated list
            return [addr.strip() for addr in v.split(",") if addr.strip()]
        return v

    @field_validator("manual_worker_allowed_hosts", mode="before")
    @classmethod
    def validate_manual_worker_allowed_hosts(cls, v: Union[str, List[str]]) -> List[str]:
        """Accept a comma-separated env-var string or a list."""
        if isinstance(v, str):
            return [h.strip() for h in v.split(",") if h.strip()]
        return v

    @field_validator("context_compression_techniques", mode="before")
    @classmethod
    def validate_context_compression_techniques(cls, v: Union[str, List[str]]) -> List[str]:
        """Accept a comma-separated env-var string or a list; reject unknown names."""
        if isinstance(v, str):
            v = [t.strip() for t in v.split(",") if t.strip()]
        allowed = {"skeletonize", "summarize", "llmlingua"}
        unknown = [t for t in v if t not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown context_compression technique(s) {unknown}; "
                f"expected any of {sorted(allowed)}"
            )
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
