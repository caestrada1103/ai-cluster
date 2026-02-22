# Configuration Guide

## Table of Contents
1. [Overview](#overview)
2. [Configuration Hierarchy](#configuration-hierarchy)
3. [Coordinator Configuration](#coordinator-configuration)
4. [Worker Configuration](#worker-configuration)
5. [Model Configuration](#model-configuration)
6. [Environment Variables](#environment-variables)
7. [Configuration Files](#configuration-files)
8. [Dynamic Configuration](#dynamic-configuration)
9. [Security Configuration](#security-configuration)
10. [Performance Tuning](#performance-tuning)
11. [Monitoring Configuration](#monitoring-configuration)
12. [Advanced Options](#advanced-options)
13. [Troubleshooting](#troubleshooting)

---

## Overview

The AI Cluster uses a hierarchical configuration system that allows for flexible deployment scenarios. Configuration can be provided through:

- **Configuration files** (YAML, TOML, JSON)
- **Environment variables**
- **Command-line arguments**
- **Dynamic API updates** (for supported settings)

### Configuration Priority (Highest to Lowest)

1. Command-line arguments
2. Environment variables
3. Dynamic API updates
4. Configuration files
5. Default values

---

## Configuration Hierarchy

```
┌──────────────────────────────────────────────────────────────┐
│                   Global Configuration                       │
│   ┌─────────────────────────────────────────────────────┐    │
│   │                  coordinator.yaml                   │    │
│   │  • Server settings    • Discovery method            │    │
│   │  • API configuration  • Model registry              │    │
│   │  • Security           • Monitoring                  │    │
│   └─────────────────────────────────────────────────────┘    │
│                             │                                │
│          ┌──────────────────┼───────────────────┐            │
│          ▼                  ▼                   ▼            │
│ ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ |
│ │  worker.toml    │  │  models.toml    │  │  logging.yaml  │ |
│ │ • GPU settings  │  │ • Model configs │  │ • Log levels   │ |
│ │ • Parallelism   │  │ • Quantization  │  │ • Outputs      │ |
│ │ • Cache         │  │ • Paths         │  │ • Formats      │ |
│ └─────────────────┘  └─────────────────┘  └────────────────┘ |
│                             │                                │
│          ┌──────────────────┴──────────────────┐             │
│          ▼                                     ▼             │
│ ┌─────────────────┐                    ┌─────────────────┐   │
│ │  prometheus.yml │                    │   alerts.yml    │   │
│ │ • Scrape config │                    │ • Alert rules   │   │
│ │ • Targets       │                    │ • Notifications │   │
│ └─────────────────┘                    └─────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Coordinator Configuration

### Main Configuration File: `config/coordinator.yaml`

```yaml
# AI Cluster Coordinator Configuration
# Version: 1.0
# Last Updated: 2024-01-15

# ============================================================================
# Server Settings
# ============================================================================

server:
  # Host to bind to (0.0.0.0 for all interfaces)
  host: "0.0.0.0"
  
  # Port for HTTP API
  port: 8000
  
  # Number of worker processes (auto = CPU count)
  workers: auto
  
  # Request timeout in seconds
  timeout_seconds: 60
  
  # Maximum request size in MB
  max_request_size_mb: 100
  
  # Keep-alive timeout for persistent connections
  keep_alive_seconds: 5
  
  # Maximum number of simultaneous connections
  max_connections: 1000
  
  # Enable gzip compression for responses
  enable_compression: true
  
  # Compression level (1-9)
  compression_level: 6

# ============================================================================
# API Settings
# ============================================================================

api:
  # API version prefix
  prefix: "/v1"
  
  # Enable interactive API documentation
  enable_docs: true
  
  # Enable CORS
  cors:
    enabled: true
    allow_origins:
      - "*"  # In production, specify exact origins
    allow_methods:
      - "GET"
      - "POST"
      - "DELETE"
      - "OPTIONS"
    allow_headers:
      - "Authorization"
      - "Content-Type"
    allow_credentials: false
    max_age: 3600
  
  # Rate limiting
  rate_limits:
    enabled: true
    default:
      requests_per_minute: 60
      tokens_per_minute: 10000
    tiers:
      free:
        requests_per_minute: 60
        tokens_per_minute: 10000
      pro:
        requests_per_minute: 600
        tokens_per_minute: 100000
      enterprise:
        requests_per_minute: 6000
        tokens_per_minute: 1000000

# ============================================================================
# Worker Discovery
# ============================================================================

discovery:
  # Discovery method: static, mdns, broadcast, consul, kubernetes
  method: "mdns"
  
  # Discovery interval in seconds
  interval_seconds: 30
  
  # Static discovery (used when method = "static")
  static:
    workers:
      - "192.168.1.10:50051"
      - "192.168.1.11:50051"
      - "worker1.local:50051"
  
  # mDNS discovery (used when method = "mdns")
  mdns:
    service_name: "_ai-worker._tcp.local."
    discovery_interval_seconds: 30
    interface: "eth0"  # Optional, specify network interface
  
  # Broadcast discovery (used when method = "broadcast")
  broadcast:
    port: 50052
    interface: "eth0"
    broadcast_address: "255.255.255.255"
    ttl: 3
  
  # Consul discovery (used when method = "consul")
  consul:
    host: "consul.service.consul"
    port: 8500
    service_name: "ai-worker"
    datacenter: "dc1"
    token: ""  # Optional Consul token
    tls:
      enabled: false
      ca_cert: "/etc/consul/ca.pem"
      cert: "/etc/consul/cert.pem"
      key: "/etc/consul/key.pem"
  
  # Kubernetes discovery (used when method = "kubernetes")
  kubernetes:
    namespace: "ai-cluster"
    label_selector: "app=ai-worker"
    kubeconfig: "~/.kube/config"  # Optional, for out-of-cluster
    use_service_endpoints: true

# ============================================================================
# Health Monitoring
# ============================================================================

health:
  # Health check interval in seconds
  check_interval_seconds: 30
  
  # Health check timeout in seconds
  check_timeout_seconds: 5
  
  # Maximum consecutive failures before marking unhealthy
  max_failures: 3
  
  # Recovery timeout after failure (seconds)
  recovery_timeout_seconds: 60
  
  # Startup grace period (seconds)
  startup_grace_period_seconds: 10
  
  # Health check endpoints
  endpoints:
    liveness: "/health/live"
    readiness: "/health/ready"
    startup: "/health/startup"

# ============================================================================
# Request Routing
# ============================================================================

routing:
  # Load balancing strategy: round_robin, least_load, random, affinity, power_of_two
  strategy: "least_load"
  
  # Request queue size
  queue_size: 1000
  
  # Request timeout in seconds
  request_timeout_seconds: 60
  
  # Maximum retries on failure
  max_retries: 3
  
  # Retry delay in milliseconds
  retry_delay_ms: 100
  
  # Circuit breaker configuration
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout_seconds: 30
    half_open_requests: 3
  
  # Affinity routing (for session persistence)
  affinity:
    enabled: true
    ttl_seconds: 3600
    cookie_name: "WORKER_AFFINITY"
  
  # Request prioritization
  priorities:
    critical_weight: 10
    high_weight: 5
    normal_weight: 1
    low_weight: 0.5
    batch_weight: 0.1

# ============================================================================
# Model Management
# ============================================================================

models:
  # Model registry configuration file
  config_file: "config/models.toml"
  
  # Model cache directory
  cache_dir: "/data/models"
  
  # Download directory for new models
  download_dir: "/data/downloads"
  
  # Automatically load models on startup
  auto_load_on_startup: true
  
  # Automatically unload idle models (minutes, 0 = disable)
  auto_unload_after_idle_minutes: 60
  
  # Maximum number of models loaded per worker
  max_loaded_models_per_worker: 5
  
  # Default model for inference when not specified
  default_model: "deepseek-7b"
  
  # Warm up models on load (run sample inference)
  warm_up_models: true
  
  # Warm up prompts (varied to test different paths)
  warm_up_prompts:
    - "Hello, how are you?"
    - "Explain quantum computing"
    - "Write a short poem"
    - "What is 2+2?"
    - "Translate 'hello' to Spanish"
  
  # Model versioning
  versioning:
    enabled: true
    default_version: "latest"
    allow_multiple_versions: false

# ============================================================================
# Performance Tuning
# ============================================================================

performance:
  # Enable continuous batching
  enable_batching: true
  
  # Maximum batch size
  max_batch_size: 32
  
  # Batch timeout in milliseconds
  batch_timeout_ms: 50
  
  # Enable response streaming
  enable_streaming: true
  
  # Enable response compression
  enable_compression: true
  
  # Compression threshold in bytes
  compression_threshold_bytes: 1024
  
  # Pre-warm workers on startup
  prewarm_workers: true
  
  # Maximum concurrent requests per worker
  max_concurrent_requests_per_worker: 10
  
  # Connection pooling
  connection_pool:
    max_size: 100
    idle_timeout_seconds: 60
  
  # Request timeouts by priority
  timeouts:
    critical_seconds: 120
    high_seconds: 60
    normal_seconds: 30
    low_seconds: 15
    batch_seconds: 300

# ============================================================================
# Security Configuration
# ============================================================================

security:
  # Enable TLS for API endpoints
  enable_tls: false
  
  # TLS certificate files
  tls:
    cert_file: "/etc/ai-cluster/cert.pem"
    key_file: "/etc/ai-cluster/key.pem"
    ca_file: "/etc/ai-cluster/ca.pem"
    min_version: "1.2"
    cipher_suites:
      - "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
      - "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
  
  # Authentication
  auth:
    enabled: false
    method: "api_key"  # api_key, jwt, oauth2, mtls
    
    # API Key authentication
    api_key:
      header_name: "Authorization"
      header_prefix: "Bearer "
      keys_file: "/etc/ai-cluster/api_keys.txt"
      keys:  # In production, use keys_file
        - "sk-1234567890abcdef"
        - "sk-0987654321fedcba"
    
    # JWT authentication
    jwt:
      secret: "your-secret-key-here"  # Use environment variable in production
      algorithm: "HS256"
      issuer: "ai-cluster"
      audience: "api"
      leeway_seconds: 60
    
    # OAuth2 authentication
    oauth2:
      provider: "google"
      client_id: "your-client-id"
      client_secret: "your-client-secret"
      token_url: "https://oauth2.googleapis.com/token"
      userinfo_url: "https://www.googleapis.com/oauth2/v3/userinfo"
    
    # mTLS authentication
    mtls:
      require_client_cert: true
      verify_chain: true
      client_ca_file: "/etc/ai-cluster/client-ca.pem"
  
  # Authorization (RBAC)
  rbac:
    enabled: false
    roles_file: "/etc/ai-cluster/roles.yaml"
    default_role: "user"
    roles:
      admin:
        permissions: ["*"]
      user:
        permissions: ["inference:read", "models:list"]
      guest:
        permissions: ["inference:read"]
  
  # Rate limiting per user
  rate_limits_per_user:
    enabled: true
    defaults:
      requests_per_minute: 60
      tokens_per_minute: 10000
    overrides:
      "admin@example.com":
        requests_per_minute: 600
        tokens_per_minute: 100000

# ============================================================================
# Logging Configuration
# ============================================================================

logging:
  # Log level: debug, info, warning, error
  level: "info"
  
  # Log format: json, text
  format: "json"
  
  # Log output: stdout, stderr, file
  output: "stdout"
  
  # Log file (if output = "file")
  file: "/var/log/ai-cluster/coordinator.log"
  
  # Log rotation
  rotation:
    max_size_mb: 100
    max_backups: 10
    max_age_days: 30
    compress: true
  
  # Structured logging fields
  fields:
    service: "coordinator"
    environment: "production"
    version: "0.1.0"
  
  # Audit logging (security events)
  audit:
    enabled: true
    file: "/var/log/ai-cluster/audit.log"
    events:
      - "model_load"
      - "model_unload"
      - "auth_failure"
      - "rate_limit_exceeded"
      - "config_change"

# ============================================================================
# Monitoring & Metrics
# ============================================================================

monitoring:
  # Enable metrics collection
  enabled: true
  
  # Metrics server port
  metrics_port: 9090
  
  # Metrics path
  metrics_path: "/metrics"
  
  # Prometheus configuration
  prometheus:
    enabled: true
    histogram_buckets: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    summary_max_age_seconds: 600
    summary_age_buckets: 5
  
  # Grafana integration
  grafana:
    dashboard_path: "monitoring/grafana-dashboards"
    datasource_name: "Prometheus"
    annotations: true
  
  # Distributed tracing
  tracing:
    enabled: false
    provider: "jaeger"  # jaeger, zipkin, otlp
    jaeger:
      host: "jaeger.service.consul"
      port: 6831
      service_name: "ai-coordinator"
      sample_rate: 0.1
      max_tag_value_length: 256
    zipkin:
      endpoint: "http://zipkin:9411/api/v2/spans"
    otlp:
      endpoint: "http://otel-collector:4317"
      protocol: "grpc"
  
  # Health check metrics
  health_metrics:
    enabled: true
    include_gpu_stats: true
    include_memory_stats: true
    include_request_stats: true

# ============================================================================
# Resource Limits
# ============================================================================

resources:
  # Maximum memory usage (GB)
  max_memory_gb: 128
  
  # Maximum CPU cores to use
  max_cpu_cores: 32
  
  # Disk cache size (GB)
  disk_cache_gb: 500
  
  # Maximum open file descriptors
  max_open_files: 65535
  
  # File system cache
  fs_cache:
    enabled: true
    max_size_gb: 10
    ttl_seconds: 3600

# ============================================================================
# Database Configuration (Optional)
# ============================================================================

database:
  # Enable request tracking and analytics
  enabled: false
  
  # Database type: sqlite, postgres, mysql
  type: "sqlite"
  
  # SQLite configuration
  sqlite:
    path: "/data/coordinator.db"
    journal_mode: "WAL"
    synchronous: "NORMAL"
  
  # PostgreSQL configuration
  postgres:
    host: "localhost"
    port: 5432
    database: "ai-cluster"
    username: "coordinator"
    password: ""  # Use environment variable
    ssl_mode: "disable"
    max_connections: 10
    connection_timeout_seconds: 5
  
  # MySQL configuration
  mysql:
    host: "localhost"
    port: 3306
    database: "ai-cluster"
    username: "coordinator"
    password: ""  # Use environment variable
    charset: "utf8mb4"
    max_connections: 10
  
  # Connection pool
  pool:
    min_size: 2
    max_size: 10
    idle_timeout_seconds: 300
    max_lifetime_seconds: 3600

# ============================================================================
# Cache Configuration
# ============================================================================

cache:
  # Enable response caching
  enabled: true
  
  # Cache type: memory, redis, memcached
  type: "memory"
  
  # In-memory cache
  memory:
    max_size_mb: 1024
    eviction_policy: "lru"  # lru, lfu, fifo, ttl
    ttl_seconds: 3600
  
  # Redis cache
  redis:
    host: "redis.service.consul"
    port: 6379
    password: ""  # Use environment variable
    db: 0
    max_connections: 10
    ssl: false
  
  # Memcached cache
  memcached:
    hosts:
      - "memcached1:11211"
      - "memcached2:11211"
    timeout_seconds: 1
    pool_size: 5
  
  # Cache keys
  keys:
    model_info_ttl_seconds: 300
    worker_status_ttl_seconds: 10
    inference_prefix: "infer:"
    model_prefix: "model:"

# ============================================================================
# Notifications
# ============================================================================

notifications:
  # Enable notifications
  enabled: false
  
  # Slack integration
  slack:
    webhook_url: "https://hooks.slack.com/services/xxx/yyy/zzz"
    channel: "#ai-cluster-alerts"
    username: "AI Cluster Bot"
    icon_emoji: ":robot_face:"
  
  # Email notifications
  email:
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    username: "alerts@example.com"
    password: ""  # Use environment variable
    from: "ai-cluster@example.com"
    to:
      - "admin@example.com"
      - "oncall@example.com"
    tls: true
  
  # PagerDuty integration
  pagerduty:
    routing_key: "your-routing-key"
    service_id: "your-service-id"
    severity: "critical"
  
  # Notification rules
  rules:
    worker_down:
      channels: ["slack", "pagerduty"]
      throttle_seconds: 300
    model_load_failed:
      channels: ["slack"]
      throttle_seconds: 60
    high_error_rate:
      channels: ["slack", "email"]
      threshold: 0.05
      throttle_seconds: 600

# ============================================================================
# Advanced Settings
# ============================================================================

advanced:
  # Debug mode (enables additional logging)
  debug_mode: false
  
  # Memory profiling
  profile_memory: false
  
  # CPU profiling
  profile_cpu: false
  
  # Enable pprof endpoints
  enable_pprof: false
  pprof_port: 6060
  
  # gRPC reflection for debugging
  enable_reflection: true
  
  # Enable health check endpoints
  enable_health_check: true
  
  # Shutdown timeout in seconds
  shutdown_timeout_seconds: 30
  
  # Graceful shutdown
  graceful_shutdown: true
  
  # Background task concurrency
  background_tasks: 10
  
  # Feature flags
  features:
    enable_streaming: true
    enable_batching: true
    enable_quantization: true
    enable_speculative: false
    enable_experimental: false
  
  # Custom headers
  custom_headers:
    X-Server-Name: "AI-Cluster"
    X-API-Version: "v1"

# ============================================================================
# Backup Configuration
# ============================================================================

backup:
  # Enable automatic backups
  enabled: false
  
  # Backup directory
  directory: "/backups"
  
  # Backup schedule (cron expression)
  schedule: "0 2 * * *"  # 2 AM daily
  
  # Retention days
  retention_days: 30
  
  # Backup content
  content:
    - "config"
    - "models"
    - "database"
  
  # Compression
  compress: true
  compression_type: "gzip"
  
  # S3 backup (optional)
  s3:
    enabled: false
    bucket: "ai-cluster-backups"
    region: "us-east-1"
    access_key: ""  # Use environment variable
    secret_key: ""  # Use environment variable
    endpoint: ""  # Optional custom endpoint
```

---

## Worker Configuration

### Main Configuration File: `config/worker.toml`

```toml
# AI Cluster Worker Configuration
# Version: 1.0
# Last Updated: 2024-01-15

# ============================================================================
# Worker Identity
# ============================================================================

[worker]
# Worker ID (auto-generated if not specified)
id = "worker-1"

# Worker version
version = "0.1.0"

# Environment: development, staging, production
environment = "production"

# Worker tags for discovery and routing
tags = { region = "us-east", rack = "rack-1", type = "gpu" }

# Description for monitoring
description = "AMD GPU Worker Node 1"

# ============================================================================
# gRPC Server Settings
# ============================================================================

[grpc]
# Server port
port = 50051

# Maximum message size in MB
max_message_size_mb = 100

# Keepalive settings
keepalive_time_ms = 10000
keepalive_timeout_ms = 5000
max_pings_without_data = 0
keepalive_permit_without_calls = true

# Connection settings
concurrent_streams = 100
max_connection_age_ms = 86400000  # 24 hours
max_connection_age_grace_ms = 60000  # 1 minute grace period

# gRPC server options
server_options = [
    { key = "grpc.max_concurrent_streams", value = 1000 },
    { key = "grpc.initial_window_size", value = 65535 },
    { key = "grpc.initial_conn_window_size", value = 65535 }
]

# ============================================================================
# Metrics Server
# ============================================================================

[metrics]
# Enable metrics server
enabled = true

# Metrics server port
port = 9091

# Metrics path
path = "/metrics"

# Prometheus histogram buckets
histogram_buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Metrics collection interval
collection_interval_seconds = 5

# Include system metrics
include_system_metrics = true

# Include process metrics
include_process_metrics = true

# ============================================================================
# Logging Configuration
# ============================================================================

[logging]
# Log level: debug, info, warn, error
level = "info"

# Log format: json, text
format = "json"

# Log output: stdout, stderr, file
output = "stdout"

# Log file path (if output = "file")
file_path = "/var/log/ai-cluster/worker.log"

# Log rotation
max_size_mb = 100
max_backups = 10
max_age_days = 30
compress = true

# Structured logging fields
[logging.fields]
service = "worker"
component = "inference"

# ============================================================================
# GPU Configuration
# ============================================================================

[gpu]
# GPU device IDs to use. Useful for mixed-GPU setups (e.g., selecting only specific cards)
# If empty, uses all available GPUs.
device_ids = [0, 1]

# Minimum free memory to keep (GB)
min_memory_free_gb = 1.0

# Maximum fraction of GPU memory to use (0.0-1.0)
memory_fraction = 0.9

# Enable memory pooling for faster allocations
enable_memory_pooling = true

# Memory pool size per GPU (GB)
memory_pool_size_gb = 2.0

# Compute preference: performance, power_saver, balanced
compute_preference = "performance"

# Enable peer-to-peer access between GPUs
enable_peer_access = true

# Enable unified memory (for devices that support it)
enable_unified_memory = false

# Enable NVIDIA MPS (Multi-Process Service)
enable_mps = false

# MPS pipe directory
mps_pipe_directory = "/tmp/mps"

# Enable MIG (Multi-Instance GPU) for A100/H100
enable_migs = false

# MIG device configuration
mig_devices = [
    { gpu = 0, slice = "1g.5gb" },
    { gpu = 0, slice = "1g.5gb" }
]

# GPU synchronization settings
sync_streams = true
default_stream_priority = 0
max_concurrent_kernels = 32

# GPU compute modes: default, exclusive, shared
compute_mode = "shared"

# GPU clock settings
[gpu.clock]
# Clock limits (0 = use default)
max_clock_mhz = 0
memory_clock_mhz = 0
power_limit_watts = 0

# GPU profiling
[gpu.profiling]
enable_profiling = false
profile_dir = "/tmp/gpu_profiles"
profile_interval_ms = 100

# GPU-specific optimizations
[gpu.optimizations]
enable_tensor_cores = true
enable_tf32 = true
enable_cudnn = true
enable_cublas = true
cudnn_benchmark = false
cudnn_deterministic = false

# AMD-specific settings
[gpu.amd]
enable_rocm = true
rocm_path = "/opt/rocm"
enable_xgmi = true
enable_pcie_atomics = true
hip_visible_devices = "0,1"

# NVIDIA-specific settings
[gpu.nvidia]
enable_cuda = true
cuda_path = "/usr/local/cuda"
enable_nvlink = true
enable_infiniband = true
nvml_monitoring = true

# ============================================================================
# Model Loader Configuration
# ============================================================================

[model_loader]
# Cache directory for downloaded models
cache_dir = "/data/models"

# Download directory for temporary files
download_dir = "/data/downloads"

# Maximum concurrent model loads
max_concurrent_loads = 2

# Load timeout in seconds
load_timeout_seconds = 300

# Verify checksums after download
verify_checksums = true

# Use memory mapping for model files
enable_mmap = true

# Pin memory for faster transfers
pin_memory = true

# Prefetch size in GB
prefetch_size_gb = 2.0

# Model format: safetensors, mpk, pytorch
preferred_format = "safetensors"

# Keep models in memory after loading
keep_in_memory = true

# Model warm-up on load
warm_up = true
warm_up_prompts = [
    "Hello, world!",
    "What is AI?",
    "Explain quantum computing"
]

# HuggingFace configuration
[model_loader.huggingface]
enabled = true
cache_dir = "/data/huggingface"
token = ""  # Use environment variable
offline_mode = false

# Download retry settings
[model_loader.retry]
max_attempts = 3
initial_delay_ms = 1000
max_delay_ms = 10000
backoff_multiplier = 2.0

# ============================================================================
# Inference Settings
# ============================================================================

[inference]
# Default generation parameters
default_max_tokens = 512
default_temperature = 0.7
default_top_p = 0.95
default_top_k = 40
default_repetition_penalty = 1.0
default_length_penalty = 1.0

# Performance settings
max_batch_size = 32
batch_timeout_ms = 50
enable_continuous_batching = true
max_queued_requests = 1000
request_timeout_seconds = 60

# Token streaming
enable_streaming = true
stream_chunk_size = 1  # Tokens per chunk

# Sampling methods
sampling_methods = ["temperature", "top_k", "top_p", "typical"]

# Stop sequences
stop_sequences = ["</s>", "<|endoftext|>"]

# Prefix caching
enable_prefix_caching = true
max_prefix_cache_size = 1000

# Speculative decoding
[inference.speculative]
enabled = false
draft_model_path = "/data/models/tiny-model"
num_speculative_tokens = 5
verify_probability = 0.8
max_draft_tokens = 100

# Quantization
[inference.quantization]
supported = ["fp16", "int8", "int4", "fp8"]
default = "fp16"
dynamic_quantization = true
calibration_size = 100
calibration_prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is the future.",
    "Once upon a time in a faraway land."
]

# ============================================================================
# KV Cache Configuration
# ============================================================================

[kv_cache]
# Enable KV cache for efficient generation
enabled = true

# Maximum cache size in GB
max_cache_size_gb = 8.0

# Cache block size in tokens
block_size = 16

# Enable prefix caching (share cache for common prefixes)
enable_prefix_caching = true

# Maximum prefix cache size
max_prefix_cache_size = 1000

# Eviction policy: lru, lfu, fifo
eviction_policy = "lru"

# Enable cache compression
enable_compression = false

# Compression algorithm: zstd, lz4, gzip
compression_algorithm = "zstd"

# Compression level (1-22)
compression_level = 3

# Cache statistics tracking
track_stats = true

# Cache warming
warm_on_load = true

# Memory limits per model
[kv_cache.per_model]
"deepseek-7b" = 4.0
"llama3-8b" = 4.0
"default" = 2.0

# ============================================================================
# Parallelism Configuration
# ============================================================================

[parallelism]
# Default strategy: auto, single, pipeline, tensor, data, expert
default_strategy = "auto"

# Pipeline parallelism settings
[parallelism.pipeline]
# Number of pipeline stages
num_stages = 2

# Number of microbatches for pipeline parallelism
num_microbatches = 4

# Enable interleaved schedule (1F1B)
enable_interleaved_schedule = true

# Communication method: p2p, nccl, rccl
communication = "nccl"

# Gradient accumulation steps
gradient_accumulation_steps = 1

# Tensor parallelism settings
[parallelism.tensor]
# Tensor parallel size
size = 2

# Enable sequence parallelism
enable_sequence_parallel = false

# All-gather communication method: nccl, rccl, custom
all_gather_comm = "nccl"

# Reduce-scatter communication method
reduce_scatter_comm = "nccl"

# Data parallelism settings
[parallelism.data]
# Number of data parallel replicas
replicas = 1

# Gradient sync mode: all_reduce, reduce_scatter, all_gather
gradient_sync_mode = "all_reduce"

# Sync interval in steps
sync_interval_steps = 1

# Average gradients across replicas
average_gradients = true

# Expert parallelism (for MoE models)
[parallelism.expert]
# Number of experts per GPU
num_experts_per_gpu = 8

# Expert routing algorithm: top_k, random, round_robin
expert_routing = "top_k"

# Enable load balancing
enable_load_balancing = true

# Load balancing threshold
load_balance_threshold = 0.2

# Expert capacity factor
capacity_factor = 1.2

# ============================================================================
# Communication Settings
# ============================================================================

[communication]
# Default timeout in seconds
timeout_seconds = 30

# NCCL settings (NVIDIA)
[communication.nccl]
# NCCL communicator timeout
timeout_seconds = 30

# Minimum CTAs per thread block
min_ctas = 1

# Maximum CTAs per thread block
max_ctas = 32

# CGA cluster size
cga_cluster_size = 2

# Enable NVLink
enable_nvlink = true

# Enable InfiniBand
enable_infiniband = true

# NCCL debug
debug = false

# NCCL environment variables
environment = [
    "NCCL_DEBUG=INFO",
    "NCCL_IB_DISABLE=0",
    "NCCL_NET_GDR_LEVEL=5"
]

# RCCL settings (AMD)
[communication.rccl]
# RCCL communicator timeout
timeout_seconds = 30

# Enable XGMI
enable_xgmi = true

# Enable PCIe
enable_pcie = true

# Maximum parallel transfers
max_parallel_transfers = 4

# RCCL environment variables
environment = [
    "RCCL_DEBUG=INFO",
    "RCCL_ENABLE_XGMI=1"
]

# Custom communication settings
[communication.custom]
# Enable GPUDirect RDMA
enable_gdr = false

# Enable RDMA over Converged Ethernet
enable_roce = false

# RoCE interface
roce_interface = "eth0"

# MPI path
mpi_path = "/usr/lib/x86_64-linux-gnu/openmpi"

# Use TCP for fallback
use_tcp = false

# TCP interface
tcp_interface = "eth0"

# ============================================================================
# Tensor Operations
# ============================================================================

[tensor]
# Enable TF32 (NVIDIA)
enable_tf32 = true

# Enable FP16
enable_fp16 = true

# Enable BF16
enable_bf16 = true

# Enable INT8
enable_int8 = true

# Enable FP8 (H100 only)
enable_fp8 = false

# Enable tensor cores
enable_tensor_cores = true

# Enable cuDNN
enable_cudnn = true

# Enable cuBLAS
enable_cublas = true

# Enable Triton kernels
enable_triton = false

# Triton kernel cache
triton_cache_dir = "/tmp/triton"

# cuDNN benchmark
cudnn_benchmark = false

# cuDNN deterministic
cudnn_deterministic = false

# cuBLAS workspace size
cublas_workspace_size_mb = 32

# Math mode: default, fast, compatible
math_mode = "fast"

# ============================================================================
# Cache Settings
# ============================================================================

[cache]
# Model cache
[cache.model]
enabled = true
size_gb = 50
ttl_seconds = 3600
eviction_policy = "lru"

# Weight cache
[cache.weights]
enabled = true
size_gb = 10
compress = true
compression_level = 3

# Kernel cache
[cache.kernel]
enabled = true
size = 1000
directory = "/tmp/ai-worker/kernels"
persist = false

# Graph cache
[cache.graph]
enabled = true
size = 100
directory = "/tmp/ai-worker/graphs"
autotune = true

# ============================================================================
# Monitoring
# ============================================================================

[monitoring]
# Enable profiling
enable_profiling = false

# Profile directory
profile_dir = "/tmp/profiles"

# Profile interval in milliseconds
profile_interval_ms = 100

# Trace GPU kernels
trace_gpu_kernels = true

# Trace CPU operations
trace_cpu_ops = true

# Trace memory allocations
trace_memory = false

# Export traces
trace_export_format = "json"
trace_export_dir = "/tmp/traces"

# Alert thresholds
[monitoring.alerts]
# Alert on out-of-memory
alert_on_oom = true

# Alert on timeout
alert_on_timeout = true

# Alert on error rate above threshold
alert_on_error_rate = 0.1

# Alert on GPU temperature above threshold
alert_on_gpu_temp = 85

# Alert on power limit
alert_on_power_limit = true

# Alert on memory fragmentation
alert_on_memory_fragmentation = 0.3

# ============================================================================
# Security
# ============================================================================

[security]
# Enable process isolation
enable_isolation = false

# Sandbox path
sandbox_path = "/tmp/worker-sandbox"

# Enable seccomp
enable_seccomp = true

# Seccomp profile
seccomp_profile = "default.json"

# Enable AppArmor
enable_apparmor = false

# AppArmor profile
apparmor_profile = "ai-worker"

# Enable SELinux
enable_selinux = false

# SELinux context
selinux_context = "system_u:system_r:ai_worker_t:s0"

# Authentication
[security.auth]
# Require authentication
require_auth = false

# Auth token (shared secret)
token = "your-secret-token-here"

# Allowed coordinator IPs
allowed_coordinators = ["192.168.1.100", "192.168.1.101"]

# mTLS settings
[security.mtls]
enabled = false
cert_file = "/etc/ai-worker/cert.pem"
key_file = "/etc/ai-worker/key.pem"
ca_file = "/etc/ai-worker/ca.pem"
verify_client = true

# ============================================================================
# Resource Limits
# ============================================================================

[resources]
# Maximum memory usage in GB
max_memory_gb = 128

# Maximum CPU cores to use
max_cpu_cores = 32

# Maximum threads
max_threads = 64

# Maximum open file descriptors
max_open_files = 65535

# Nice level (-20 to 19, lower = higher priority)
nice_level = -5

# IO priority (0-7, lower = higher priority)
io_priority = 0

# CPU affinity (comma-separated list of cores)
cpu_affinity = "0-15"

# Memory policy: default, interleave, local
memory_policy = "local"

# Scheduler policy: other, fifo, rr
scheduler_policy = "other"

# Scheduler priority (for FIFO/RR, 1-99)
scheduler_priority = 1

# ============================================================================
# Advanced Settings
# ============================================================================

[advanced]
# Debug mode (extra logging)
debug = false

# Trace memory allocations
trace_allocations = false

# Validate kernels after compilation
validate_kernels = false

# Synchronize after each operation (slow!)
sync_after_each_op = false

# Detect anomalies in computations
detect_anomalies = false

# Compile with verbose output
compile_verbose = false

# Dump computation graphs
dump_graphs = false

# Graph dump directory
graph_dump_dir = "/tmp/graphs"

# JIT compilation settings
[advanced.jit]
enable_jit = true
cache_size = 100
fusion_strategy = "aggressive"

# Error handling
[advanced.error_handling]
panic_on_error = false
recover_from_oom = true
auto_restart_on_failure = true
max_restarts = 5
restart_delay_seconds = 10
```

---

## Model Configuration

### Model Registry: `config/models.toml`

```toml
# AI Cluster Model Registry
# Version: 1.0
# Last Updated: 2024-01-15

# ============================================================================
# Default Settings (Applied to all models)
# ============================================================================

[defaults]
# Default quantization
quantization = "fp16"

# Default parallelism strategy
parallelism = "auto"

# Maximum batch size
max_batch_size = 32

# Maximum sequence length
max_seq_len = 4096

# Cache size in GB
cache_size_gb = 8

# Warm-up prompts (varied to test different paths)
warm_up_prompts = [
    "Hello, world!",
    "What is artificial intelligence?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about technology.",
    "Translate 'hello' to Spanish."
]

# Stop sequences
stop_sequences = ["</s>", "<|endoftext|>", "Human:", "Assistant:"]

# Generation defaults
generation_defaults = { temperature = 0.7, top_p = 0.95, top_k = 40 }

# ============================================================================
# DeepSeek Models
# ============================================================================

[models."deepseek-7b"]
# Basic information
family = "deepseek"
description = "DeepSeek 7B Base Model with MoE architecture"
parameters = "7B"
version = "1.0"
release_date = "2024-01-01"

# Hardware requirements
min_memory_gb = 16
recommended_gpus = 1
max_gpus = 2
multi_gpu_scaling = "good"

# Model architecture
[models."deepseek-7b".architecture]
num_layers = 30
hidden_size = 4096
num_attention_heads = 32
num_kv_heads = 32
vocab_size = 32256
max_seq_len = 4096
intermediate_size = 11008
rms_norm_eps = 1e-6
rope_theta = 10000.0
is_moe = true
num_experts = 64
num_experts_per_tok = 6
moe_top_k = 6
moe_capacity_factor = 1.2

# File paths
[models."deepseek-7b".paths]
config = "/data/models/deepseek-7b/config.json"
weights = "/data/models/deepseek-7b/model.safetensors"
tokenizer = "/data/models/deepseek-7b/tokenizer.json"
vocab = "/data/models/deepseek-7b/vocab.json"
shard_index = "/data/models/deepseek-7b/model.safetensors.index.json"

# Quantization support
[models."deepseek-7b".quantization]
supported = ["fp16", "int8", "int4"]
default = "fp16"
int8_algorithm = "symmetric"
int4_group_size = 32
calibration_size = 100

# Parallelism support
[models."deepseek-7b".parallelism]
supported = ["single", "pipeline", "expert"]
default = "auto"
pipeline_stages = [1, 2]
tensor_parallel_sizes = [1]
expert_parallel = true

# Performance characteristics
[models."deepseek-7b".performance]
throughput_tokens_per_sec = 45
latency_p50_ms = 120
latency_p95_ms = 250
batch_1_tokens_per_sec = 45
batch_8_tokens_per_sec = 210
batch_32_tokens_per_sec = 450

# HuggingFace source
[models."deepseek-7b".hf]
repo_id = "deepseek-ai/deepseek-llm-7b-base"
filename = "model.safetensors"
revision = "main"
token = ""  # Optional, use environment variable

# Model tags for routing
[models."deepseek-7b".tags]
capabilities = ["chat", "code", "reasoning"]
languages = ["en", "zh"]
license = "deepseek"

# ============================================================================
# DeepSeek 67B Model
# ============================================================================

[models."deepseek-67b"]
family = "deepseek"
description = "DeepSeek 67B Model with extensive MoE"
parameters = "67B"
version = "1.0"
min_memory_gb = 140
recommended_gpus = 4
max_gpus = 8

[models."deepseek-67b".architecture]
num_layers = 95
hidden_size = 8192
num_attention_heads = 64
num_kv_heads = 64
vocab_size = 32256
max_seq_len = 4096
intermediate_size = 22016
rms_norm_eps = 1e-6
rope_theta = 10000.0
is_moe = true
num_experts = 128
num_experts_per_tok = 8

[models."deepseek-67b".paths]
config = "/data/models/deepseek-67b/config.json"
weights = "/data/models/deepseek-67b/model.safetensors"
tokenizer = "/data/models/deepseek-67b/tokenizer.json"

[models."deepseek-67b".quantization]
supported = ["int8", "int4", "fp8"]
default = "int8"

[models."deepseek-67b".parallelism]
supported = ["pipeline", "tensor", "expert"]
default = "auto"
pipeline_stages = [2, 4, 8]
tensor_parallel_sizes = [2, 4]

[models."deepseek-67b".hf]
repo_id = "deepseek-ai/deepseek-llm-67b-base"
filename = "model.safetensors.index.json"

# ============================================================================
# Llama 3 Models
# ============================================================================

[models."llama3-8b"]
family = "llama"
description = "Meta Llama 3 8B Instruct"
parameters = "8B"
version = "3.0"
min_memory_gb = 16
recommended_gpus = 1
max_gpus = 2

[models."llama3-8b".architecture]
num_layers = 32
hidden_size = 4096
num_attention_heads = 32
num_kv_heads = 8  # GQA
vocab_size = 128256
max_seq_len = 8192
intermediate_size = 14336
rms_norm_eps = 1e-5
rope_theta = 500000.0
rope_scaling = { type = "linear", factor = 8.0 }
is_moe = false

[models."llama3-8b".paths]
config = "/data/models/llama3-8b/config.json"
weights = "/data/models/llama3-8b/model.safetensors"
tokenizer = "/data/models/llama3-8b/tokenizer.json"

[models."llama3-8b".quantization]
supported = ["fp16", "int8", "int4"]
default = "fp16"

[models."llama3-8b".parallelism]
supported = ["single", "pipeline", "tensor"]
default = "auto"

[models."llama3-8b".hf]
repo_id = "meta-llama/Meta-Llama-3-8B"
filename = "model-00001-of-00002.safetensors"

[models."llama3-8b".tags]
capabilities = ["chat", "instruct", "code"]
languages = ["en"]
license = "llama3"

# Llama 3 70B
[models."llama3-70b"]
family = "llama"
description = "Meta Llama 3 70B Instruct"
parameters = "70B"
min_memory_gb = 140
recommended_gpus = 4
max_gpus = 8

[models."llama3-70b".architecture]
num_layers = 80
hidden_size = 8192
num_attention_heads = 64
num_kv_heads = 8
vocab_size = 128256
max_seq_len = 8192
intermediate_size = 28672
rms_norm_eps = 1e-5
rope_theta = 500000.0

[models."llama3-70b".quantization]
supported = ["int8", "int4"]
default = "int8"

[models."llama3-70b".parallelism]
supported = ["pipeline", "tensor"]
default = "auto"
pipeline_stages = [2, 4, 8]
tensor_parallel_sizes = [2, 4]

[models."llama3-70b".hf]
repo_id = "meta-llama/Meta-Llama-3-70B"
filename = "model-00001-of-00030.safetensors"

# ============================================================================
# Mistral Models
# ============================================================================

[models."mistral-7b"]
family = "mistral"
description = "Mistral 7B v0.2"
parameters = "7B"
min_memory_gb = 14
recommended_gpus = 1
max_gpus = 2

[models."mistral-7b".architecture]
num_layers = 32
hidden_size = 4096
num_attention_heads = 32
num_kv_heads = 8
vocab_size = 32000
max_seq_len = 32768
intermediate_size = 14336
rms_norm_eps = 1e-5
rope_theta = 10000.0
sliding_window = 4096
is_moe = false

[models."mistral-7b".paths]
config = "/data/models/mistral-7b/config.json"
weights = "/data/models/mistral-7b/model.safetensors"
tokenizer = "/data/models/mistral-7b/tokenizer.json"

[models."mistral-7b".quantization]
supported = ["fp16", "int8"]
default = "fp16"

[models."mistral-7b".hf]
repo_id = "mistralai/Mistral-7B-v0.1"
filename = "model.safetensors"

# ============================================================================
# Mixtral MoE Model
# ============================================================================

[models."mixtral-8x7b"]
family = "mistral"
description = "Mistral Mixtral 8x7B MoE"
parameters = "46.7B"
min_memory_gb = 90
recommended_gpus = 2
max_gpus = 4

[models."mixtral-8x7b".architecture]
num_layers = 32
hidden_size = 4096
num_attention_heads = 32
num_kv_heads = 8
vocab_size = 32000
max_seq_len = 32768
intermediate_size = 14336
rms_norm_eps = 1e-5
rope_theta = 1000000.0
is_moe = true
num_experts = 8
num_experts_per_tok = 2

[models."mixtral-8x7b".quantization]
supported = ["int8", "int4"]
default = "int8"

[models."mixtral-8x7b".parallelism]
supported = ["pipeline", "tensor", "expert"]
default = "auto"

[models."mixtral-8x7b".hf]
repo_id = "mistralai/Mixtral-8x7B-v0.1"
filename = "model-00001-of-00019.safetensors"

# ============================================================================
# Gemma Models
# ============================================================================

[models."gemma-2b"]
family = "gemma"
description = "Google Gemma 2B"
parameters = "2B"
min_memory_gb = 4
recommended_gpus = 1
max_gpus = 1

[models."gemma-2b".architecture]
num_layers = 18
hidden_size = 2048
num_attention_heads = 8
num_kv_heads = 1  # MQA
vocab_size = 256000
max_seq_len = 8192
intermediate_size = 16384
rms_norm_eps = 1e-6
rope_theta = 10000.0
is_moe = false

[models."gemma-2b".hf]
repo_id = "google/gemma-2b"
filename = "model.safetensors"

[models."gemma-7b"]
family = "gemma"
description = "Google Gemma 7B"
parameters = "7B"
min_memory_gb = 14
recommended_gpus = 1
max_gpus = 2

[models."gemma-7b".architecture]
num_layers = 28
hidden_size = 3072
num_attention_heads = 16
num_kv_heads = 16
vocab_size = 256000
max_seq_len = 8192
intermediate_size = 24576
rms_norm_eps = 1e-6
rope_theta = 10000.0

[models."gemma-7b".hf]
repo_id = "google/gemma-7b"
filename = "model.safetensors"

# ============================================================================
# Phi Models
# ============================================================================

[models."phi-2"]
family = "phi"
description = "Microsoft Phi-2 (2.7B)"
parameters = "2.7B"
min_memory_gb = 6
recommended_gpus = 1
max_gpus = 1

[models."phi-2".architecture]
num_layers = 32
hidden_size = 2560
num_attention_heads = 32
num_kv_heads = 32
vocab_size = 51200
max_seq_len = 2048
intermediate_size = 10240
layer_norm_eps = 1e-5
is_moe = false

[models."phi-2".hf]
repo_id = "microsoft/phi-2"
filename = "model.safetensors"

[models."phi-3-mini"]
family = "phi"
description = "Microsoft Phi-3 Mini (3.8B)"
parameters = "3.8B"
min_memory_gb = 8
recommended_gpus = 1
max_gpus = 1

[models."phi-3-mini".architecture]
num_layers = 32
hidden_size = 3072
num_attention_heads = 32
num_kv_heads = 32
vocab_size = 32064
max_seq_len = 4096
intermediate_size = 8192
rms_norm_eps = 1e-5

[models."phi-3-mini".hf]
repo_id = "microsoft/Phi-3-mini-4k-instruct"
filename = "model.safetensors"

# ============================================================================
# Qwen Models
# ============================================================================

[models."qwen-7b"]
family = "qwen"
description = "Qwen 7B"
parameters = "7B"
min_memory_gb = 14
recommended_gpus = 1
max_gpus = 2

[models."qwen-7b".architecture]
num_layers = 32
hidden_size = 4096
num_attention_heads = 32
num_kv_heads = 32
vocab_size = 151936
max_seq_len = 8192
intermediate_size = 11008
rms_norm_eps = 1e-6
is_moe = false

[models."qwen-7b".hf]
repo_id = "Qwen/Qwen-7B"
filename = "model.safetensors"

[models."qwen-14b"]
family = "qwen"
description = "Qwen 14B"
parameters = "14B"
min_memory_gb = 28
recommended_gpus = 1
max_gpus = 2

[models."qwen-14b".architecture]
num_layers = 40
hidden_size = 5120
num_attention_heads = 40
num_kv_heads = 40
vocab_size = 151936
max_seq_len = 8192
intermediate_size = 13696

[models."qwen-14b".hf]
repo_id = "Qwen/Qwen-14B"
filename = "model.safetensors"

# ============================================================================
# Custom Model Example
# ============================================================================

[models."my-custom-model"]
family = "custom"
description = "My custom fine-tuned model"
parameters = "7B"
version = "1.2"
min_memory_gb = 16
recommended_gpus = 1
max_gpus = 2

[models."my-custom-model".architecture]
num_layers = 32
hidden_size = 4096
num_attention_heads = 32
num_kv_heads = 8
vocab_size = 50000
max_seq_len = 4096
intermediate_size = 11008
rms_norm_eps = 1e-5
rope_theta = 10000.0
is_moe = false

[models."my-custom-model".paths]
config = "/data/models/my-custom-model/config.json"
weights = "/data/models/my-custom-model/model.safetensors"
tokenizer = "/data/models/my-custom-model/tokenizer.json"

[models."my-custom-model".quantization]
supported = ["fp16", "int8"]
default = "fp16"

[models."my-custom-model".tags]
owner = "my-team"
project = "fine-tuning"
version = "1.2"
```

---

## Environment Variables

### Coordinator Environment Variables

```bash
# ============================================================================
# Coordinator Environment Variables
# ============================================================================

# Server settings
export COORDINATOR_HOST="0.0.0.0"
export COORDINATOR_PORT="8000"
export COORDINATOR_WORKERS="4"
export COORDINATOR_TIMEOUT="60"

# Discovery
export DISCOVERY_METHOD="mdns"
export STATIC_WORKERS="192.168.1.10:50051,192.168.1.11:50051"

# Security
export API_KEYS="sk-1234567890abcdef,sk-0987654321fedcba"
export JWT_SECRET="your-secret-key-here"
export ENABLE_AUTH="false"

# Database
export DATABASE_URL="postgresql://user:pass@localhost:5432/ai-cluster"
export REDIS_URL="redis://:password@localhost:6379/0"

# Logging
export LOG_LEVEL="info"
export LOG_FORMAT="json"

# Model paths
export MODEL_CACHE_DIR="/data/models"
export MODEL_DOWNLOAD_DIR="/data/downloads"

# Performance
export MAX_BATCH_SIZE="32"
export BATCH_TIMEOUT_MS="50"
export ENABLE_BATCHING="true"

# Monitoring
export METRICS_PORT="9090"
export ENABLE_METRICS="true"
export JAEGER_HOST="jaeger.service.consul"
```

### Worker Environment Variables

```bash
# ============================================================================
# Worker Environment Variables
# ============================================================================

# Worker identity
export WORKER_ID="worker-1"
export WORKER_TAGS="region=us-east,rack=rack-1"

# gRPC settings
export GRPC_PORT="50051"
export GRPC_MAX_MESSAGE_SIZE_MB="100"

# Metrics
export METRICS_PORT="9091"
export ENABLE_METRICS="true"

# GPU selection
export GPU_IDS="0,1"
export ROCM_VISIBLE_DEVICES="0,1"  # AMD
export NVIDIA_VISIBLE_DEVICES="0,1"  # NVIDIA
export CUDA_VISIBLE_DEVICES="0,1"  # NVIDIA

# Logging
export RUST_LOG="info"
export RUST_BACKTRACE="1"

# Model paths
export MODEL_CACHE_DIR="/data/models"
export MODEL_DOWNLOAD_DIR="/data/downloads"

# Performance
export MAX_BATCH_SIZE="32"
export BATCH_TIMEOUT_MS="50"
export REQUEST_TIMEOUT_SECS="60"

# Memory
export GPU_MEMORY_FRACTION="0.9"
export KV_CACHE_SIZE_GB="8"

# Security
export AUTH_TOKEN="your-secret-token-here"
export REQUIRE_AUTH="false"

# HuggingFace
export HF_TOKEN="your-hf-token-here"
export HF_HOME="/data/huggingface"
```

---

## Configuration Validation

### Validate Configuration

```bash
# Validate coordinator configuration
python -m coordinator.config.validate --file config/coordinator.yaml

# Validate worker configuration
cd worker && cargo run -- validate-config --file config/worker.toml

# Validate model registry
python scripts/validate_models.py --config config/models.toml
```

### Configuration Schema Validation

```python
# config_schema.py
from pydantic import BaseModel, validator
from typing import List, Optional, Dict

class GPUMemoryConfig(BaseModel):
    min_memory_free_gb: float = 1.0
    memory_fraction: float = 0.9
    
    @validator('memory_fraction')
    def check_fraction(cls, v):
        if not 0 < v <= 1:
            raise ValueError('memory_fraction must be between 0 and 1')
        return v

class WorkerConfig(BaseModel):
    worker_id: Optional[str]
    gpu: GPUMemoryConfig
    max_batch_size: int = 32
    
    @validator('max_batch_size')
    def check_batch_size(cls, v):
        if v < 1 or v > 256:
            raise ValueError('max_batch_size must be between 1 and 256')
        return v
```

---

## Dynamic Configuration

### API Endpoints for Runtime Configuration

```bash
# Get current configuration
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:8000/v1/config

# Update configuration at runtime
curl -X POST http://localhost:8000/v1/config \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "routing": {
      "strategy": "least_load",
      "max_batch_size": 64
    }
  }'

# Reload configuration from file
curl -X POST http://localhost:8000/v1/config/reload \
  -H "Authorization: Bearer $API_KEY"

# Get worker configuration
curl http://localhost:9091/config
```

### Dynamic Worker Configuration

```rust
// worker/src/config/dynamic.rs
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicConfig {
    pub max_batch_size: usize,
    pub batch_timeout_ms: u64,
    pub enable_streaming: bool,
    pub log_level: String,
}

pub struct DynamicConfigManager {
    config: Arc<RwLock<DynamicConfig>>,
}

impl DynamicConfigManager {
    pub async fn update(&self, new_config: DynamicConfig) {
        let mut config = self.config.write().await;
        *config = new_config;
        tracing::info!("Dynamic configuration updated");
    }
}
```

---

## Configuration Best Practices

### 1. **Environment-Based Configuration**

```yaml
# config/coordinator.yaml
server:
  host: ${COORDINATOR_HOST:-0.0.0.0}
  port: ${COORDINATOR_PORT:-8000}
  workers: ${COORDINATOR_WORKERS:-auto}
```

### 2. **Secrets Management**

```bash
# Never commit secrets to git!
# Use environment variables or secrets management

# .env file (git-ignored)
export DATABASE_PASSWORD="secure-password"
export API_KEY="sk-..."
export JWT_SECRET="random-secret"

# Or use HashiCorp Vault
export VAULT_ADDR="https://vault.example.com"
export DB_PASSWORD=$(vault kv get -field=password secret/database)
```

### 3. **Configuration Validation Hooks**

```python
# pre_start.py
def validate_config_before_start():
    """Validate configuration before starting services."""
    required_files = [
        "config/coordinator.yaml",
        "config/worker.toml",
        "config/models.toml"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            raise RuntimeError(f"Missing configuration file: {file}")
    
    # Validate critical settings
    check_gpu_availability()
    check_disk_space()
    check_network_connectivity()
```

### 4. **Configuration Versioning**

```yaml
# config/coordinator.yaml
metadata:
  version: "1.2.0"
  last_updated: "2024-01-15T10:30:00Z"
  updated_by: "admin@example.com"
  changelog: |
    - Increased max batch size to 32
    - Enabled speculative decoding
    - Updated model registry
```

### 5. **Configuration Templates**

```bash
# Create configuration from template
cp config/coordinator.yaml.example config/coordinator.yaml
cp config/worker.toml.example config/worker.toml
cp config/models.toml.example config/models.toml

# Edit with your settings
vim config/coordinator.yaml
```

---

## Troubleshooting Configuration

### Common Issues

#### 1. **GPU Not Detected**

```bash
# Check GPU visibility
rocm-smi  # AMD
nvidia-smi  # NVIDIA

# Check environment variables
echo $ROCM_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# Test in container
docker run --rm --gpus all nvidia/cuda:12.1-runtime nvidia-smi
```

#### 2. **Configuration File Not Found**

```bash
# Check file permissions
ls -la config/
chmod 644 config/*.yaml config/*.toml

# Check paths in config
grep "path" config/*.yaml
```

#### 3. **Invalid YAML/TOML Syntax**

```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('config/coordinator.yaml'))"

# Validate TOML
pip install toml
python -c "import toml; toml.load('config/worker.toml')"
```

#### 4. **Port Conflicts**

```bash
# Check if ports are in use
sudo netstat -tulpn | grep -E "8000|50051|9090|9091"

# Change ports in configuration
sed -i 's/port: 8000/port: 8001/' config/coordinator.yaml
```

### Configuration Debugging

```bash
# Enable debug logging
export RUST_LOG=debug
export LOG_LEVEL=debug

# Dump configuration on startup
./worker/target/release/ai-worker --dump-config

# Test configuration parsing
./worker/target/release/ai-worker --test-config config/worker.toml
```

---

## Configuration Migration

### Upgrading from v0.1 to v0.2

```python
# migrate_config.py
import yaml
import toml

def migrate_coordinator_config(old_path, new_path):
    with open(old_path) as f:
        old_config = yaml.safe_load(f)
    
    # New structure
    new_config = {
        "server": old_config.get("server", {}),
        "api": {
            "prefix": "/v1",
            "rate_limits": old_config.get("rate_limiting", {})
        },
        "discovery": {
            "method": old_config.get("discovery_method", "static"),
            "static_workers": old_config.get("static_workers", [])
        }
    }
    
    with open(new_path, 'w') as f:
        yaml.dump(new_config, f)
```

---

## Reference

### Configuration File Locations

| Component | Default Path | Environment Variable |
|-----------|-------------|---------------------|
| Coordinator | `config/coordinator.yaml` | `COORDINATOR_CONFIG` |
| Worker | `config/worker.toml` | `WORKER_CONFIG` |
| Models | `config/models.toml` | `MODELS_CONFIG` |
| Logging | `config/logging.yaml` | `LOGGING_CONFIG` |
| Prometheus | `config/prometheus.yml` | `PROMETHEUS_CONFIG` |

### Configuration Precedence

1. Command-line arguments (`--port 8080`)
2. Environment variables (`COORDINATOR_PORT=8080`)
3. Dynamic API updates
4. Configuration files (`coordinator.yaml`)
5. Default values

### Hot-Reloadable Settings

- Log levels
- Rate limits
- Batch sizes
- Routing strategies
- Cache sizes
- Worker priorities

---

For more information, see:
- [Architecture Guide](architecture.md)
- [API Reference](api_reference.md)
- [Deployment Guide](deployment.md)
- [Troubleshooting](troubleshooting.md)