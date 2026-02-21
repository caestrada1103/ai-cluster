# Docker Deployment for AI Cluster

This directory contains Dockerfiles and compose configurations for deploying the AI cluster in containers.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.20+
- NVIDIA Container Toolkit (for NVIDIA GPUs)
- ROCm Container Runtime (for AMD GPUs)

## Quick Start

### 1. Build the images

```bash
# Build coordinator
docker build -f docker/Dockerfile.coordinator -t ai-coordinator:latest .

# Build worker (choose backend)
docker build -f docker/Dockerfile.worker --build-arg GPU_BACKEND=hip -t ai-worker:amd .
docker build -f docker/Dockerfile.worker --build-arg GPU_BACKEND=cuda -t ai-worker:nvidia .
```

### 2. Start the cluster

```bash
# Start the cluster with monitoring and workers
docker compose up -d
```

### 3. Check status

```bash
# View logs
docker compose logs -f

# Check health
curl http://localhost:8000/health
curl http://localhost:8000/v1/workers
```

## Image Details

### Coordinator Image (`ai-coordinator`)

- Based on Python 3.10 slim
- Exposes port 8000 (HTTP API & metrics)
- Includes health check endpoint
- Runs as non-root user

### Worker Image (`ai-worker`)

Multi-architecture support:
- `amd` variant: ROCm backend for AMD GPUs
- `nvidia` variant: CUDA backend for NVIDIA GPUs
- `cpu` variant: WGPU fallback for CPU-only

Features:
- GPU passthrough support
- Memory limits and isolation
- Health checks
- Metrics endpoint

## GPU Support

### NVIDIA GPUs

Requires NVIDIA Container Toolkit:
```bash
# Install runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### AMD GPUs

Requires ROCm and Docker runtime:
```bash
# Install ROCm
sudo apt-get update
sudo apt-get install -y rocm-dev

# Configure Docker
sudo usermod -a -G video,render $USER
sudo systemctl restart docker
```

## Environment Variables

### Coordinator
- `COORDINATOR_HOST`: Bind address (default: 0.0.0.0)
- `COORDINATOR_PORT`: HTTP port (default: 8000)
- `DISCOVERY_METHOD`: Worker discovery (static/mdns/broadcast/consul)
- `CONFIG_FILE`: Path to config file

### Worker
- `WORKER_ID`: Unique worker identifier
- `GRPC_PORT`: gRPC server port (default: 50051)
- `METRICS_PORT`: Metrics port (default: 9091)
- `GPU_IDS`: Comma-separated GPU indices (default: 0)
- `RUST_LOG`: Logging level (debug/info/warn/error)
- `ROCM_VISIBLE_DEVICES`: AMD GPU selection
- `NVIDIA_VISIBLE_DEVICES`: NVIDIA GPU selection

## Volumes

- `/app/config`: Configuration files
- `/app/models`: Model weights (read-only recommended)
- `/app/data`: Runtime data
- `/app/logs`: Log files

## Networking

Services communicate over the `ai-cluster-net` bridge network:
- Coordinator: port 8000 (HTTP API & metrics)
- Workers: ports 50051-50054 (gRPC), 9091-9094 (metrics)
- Monitoring: 9099 (Prometheus), 3000 (Grafana)

## Health Checks

All services include health checks:
- Coordinator: HTTP GET /health
- Worker: HTTP GET /health (metrics port)
- Monitoring: Native Docker health checks

## Production Considerations

1. **Security**:
   - Use non-root users
   - Read-only root filesystems
   - Seccomp profiles
   - AppArmor/SELinux

2. **Resource Limits**:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '8'
         memory: 32G
       reservations:
         memory: 16G
   ```

3. **Logging**:
   - JSON format for log aggregation
   - Log rotation configured
   - Separate error logs

4. **Monitoring**:
   - Prometheus metrics
   - Grafana dashboards
   - Alerting rules

5. **Backup**:
   - Persistent volumes for data
   - Regular model cache backups

## Troubleshooting

### GPU not detected
```bash
# Check GPU visibility
docker run --rm --gpus all nvidia/cuda:12.1-runtime nvidia-smi
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/dev-ubuntu-22.04 rocm-smi
```

### Permission issues
```bash
# Ensure user is in correct groups
sudo usermod -a -G docker $USER
sudo usermod -a -G video,render $USER
```

### Network issues
```bash
# Check container connectivity
docker exec ai-worker-amd-0 ping ai-coordinator
docker exec ai-coordinator curl http://ai-worker-amd-0:9091/health
```

## Performance Tuning

1. **CPU Pinning**:
   ```yaml
   cpuset: '0-3'  # Pin to specific CPUs
   ```

2. **Memory HugePages**:
   ```yaml
   volumes:
     - /dev/hugepages:/dev/hugepages
   ```

3. **I/O Priority**:
   ```yaml
   blkio_config:
     weight: 1000
     weight_device:
       - path: /dev/sda
         weight: 500
   ```

## Example: Production Deployment

```bash
# Deploy with Docker Swarm
docker stack deploy -c docker-compose.yml ai-cluster

# Rolling updates
docker service update --image ai-worker:new-version ai-cluster_worker-gpu-0
```

## Building for Production

```bash
# Multi-arch build (AMD64 + ARM64)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t registry/ai-worker:latest \
  --push \
  -f docker/Dockerfile.worker \
  --build-arg GPU_BACKEND=hip \
  .
```

## Cleanup

```bash
# Stop all containers
docker compose down -v

# Remove images
docker rmi ai-coordinator:latest ai-worker:amd ai-worker:nvidia

# Clean volumes
docker volume prune -f
```

## Support

For issues:
1. Check logs: `docker compose logs -f`
2. Verify GPU access
3. Check configuration files
4. Ensure network connectivity

For more information, see the [main documentation](../README.md).
```

---

These Docker files provide:

1. **`Dockerfile.coordinator`** - Multi-stage Python build:
   - Poetry for dependency management
   - gRPC code generation
   - Non-root user
   - Health checks
   - Small final image

2. **`Dockerfile.worker`** - Multi-stage Rust build with GPU support:
   - Built to target `wgpu`/Vulkan for universal compatibility
   - Non-root user with GPU group access

3. **`docker-compose.yml`** (root) - Complete stack:
   - Coordinator
   - Multiple explicitly-defined GPU workers (`worker-gpu-0`, `worker-gpu-1`, etc.)
   - Prometheus + Grafana monitoring

4. **Documentation** - Comprehensive README with:
   - Build instructions
   - GPU setup
   - Environment variables
   - Production considerations
   - Troubleshooting
   - Performance tuning

These files are production-ready with security best practices (non-root users, health checks, resource limits) and support for both AMD and NVIDIA GPUs.