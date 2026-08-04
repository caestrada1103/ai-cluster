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
            "LoadModel gRPC deadline (seconds). Covers the worker's GGUF "
            "download, so it must exceed weight_size / link_speed: a 22 GB "
            "model over a 10 MB/s link needs ~37 min. The previous hardcoded "
            "300s capped API-loadable models at roughly 3 GB."
        ),
        ge=1,
    )

    # CORS
    # M3: secure by default — no explicit origins (`[]`), not `"*"`. Combined
    # with `main.py`'s always-on loopback `allow_origin_regex`
    # (`http(s)://(localhost|127.0.0.1)[:port]`), the DEFAULT deployment
    # allows browser calls from the machine the coordinator itself runs on
    # (matches this project's "local debug is permissive but opt-in" story —
    # a developer hitting the API from a page served on localhost keeps
    # working with zero config) while any other origin is rejected until an
    # operator explicitly sets COORDINATOR_CORS_ORIGINS. Set it to `"*"` to
    # restore the old wide-open behavior (still forces
    # `allow_credentials=False`, same anti-pattern guard as before), or to a
    # comma-separated/JSON list of specific origins for a real deployment.
    # NoDecode: pydantic-settings otherwise JSON-decodes any complex-typed env
    # value before our validator runs, which rejects non-JSON strings like "*".
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

    # C3 — POST /v1/workers/manual (unauthenticated arbitrary worker
    # registration in the audit finding). Disabled by default: a rogue
    # "worker" registered this way self-reports loaded_models and can be
    # selected by find_worker_for_model for real routed traffic (prompt
    # exfiltration + poisoned completions), and the address is also handed
    # straight to grpc.aio.insecure_channel / the llamaserver proxy (SSRF).
    # Opt in only for deployments that actually need runtime worker
    # registration (most don't — COORDINATOR_STATIC_WORKERS covers the
    # documented single-host/LAN-cluster setups).
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

    # H4 — an unregistered model_name sent to POST /v1/models/load is passed
    # to the worker as an arbitrary HuggingFace repo id to download and load
    # (coordinator.py: "If unknown, it's a HuggingFace pull"). Off by
    # default: only models in config/models.toml are loadable unless a
    # deployment explicitly wants ad hoc HF pulls.
    allow_unregistered_model_pull: bool = Field(
        False,
        description=(
            "Allow POST /v1/models/load for a model_name absent from the registry "
            "(env COORDINATOR_ALLOW_UNREGISTERED_MODEL_PULL) — the worker treats it "
            "as an arbitrary HuggingFace repo id to download. Off by default."
        ),
    )

    # H5: no request-body size cap previously existed anywhere on the HTTP
    # surface (api.py's _read_json_body buffers the whole body via
    # request.body()); this also covers the engine="llamaserver" proxy path,
    # which has no max_tokens/queue admission control of its own. 25 MB is
    # generous for a real chat/agentic request (long conversation history,
    # embedded code, tool results) while still bounding worst-case memory use
    # per request. See coordinator/body_limit.py.
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

    # llama-server auto-load-on-demand (Plan 13 Task 5). When True (default), a
    # proxied request for an `engine="llamaserver"` model that no worker reports
    # loaded triggers the standard load path on a healthy worker (single-flight
    # per model) before proxying. When False the coordinator preserves Phase-1
    # behavior — it 404s and the client must POST /models/load first.
    llamaserver_autoload: bool = Field(
        True,
        description=(
            "Auto-load an unloaded engine=llamaserver model on demand before proxying "
            "(env COORDINATOR_LLAMASERVER_AUTOLOAD)"
        ),
    )

    # Context compression middleware (coordinator/context_compression/) — see
    # pending-work/12-context-compression-pipeline.md. Off by default: a
    # request is only ever touched when (a) this is True or the request sets
    # `compress_context: true`, AND (b) its estimated prompt tokens exceed
    # context_compression_token_budget. Tune the budget per deployment —
    # roughly `n_ctx - max_tokens - 512` for whatever model you route to.
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
