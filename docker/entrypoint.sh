#!/bin/sh
# =============================================================================
# AI Worker Entrypoint
# =============================================================================
# Computes unique ports and worker ID for each GPU replica, then launches
# the worker binary.
#
# Environment variables:
#   GPU_INDEX        - GPU device index for this replica (default: 0)
#   GRPC_BASE_PORT   - Base gRPC port (default: 50051)
#   METRICS_BASE_PORT - Base metrics port (default: 9091)
#   WORKER_ID        - Worker ID override (auto-generated if not set)
#   RUST_LOG         - Log level (default: info)
# =============================================================================

set -e

# Defaults
GPU_INDEX="${GPU_INDEX:-0}"
GRPC_BASE_PORT="${GRPC_BASE_PORT:-50051}"
METRICS_BASE_PORT="${METRICS_BASE_PORT:-9091}"

# Compute ports: base + GPU index offset
GRPC_PORT=$((GRPC_BASE_PORT + GPU_INDEX))
METRICS_PORT=$((METRICS_BASE_PORT + GPU_INDEX))

# Auto-generate worker ID if not set
if [ -z "$WORKER_ID" ]; then
    HOSTNAME=$(hostname -s 2>/dev/null || echo "worker")
    WORKER_ID="${HOSTNAME}-gpu-${GPU_INDEX}"
fi

# Export for healthcheck
export METRICS_PORT

echo "============================================="
echo "  AI Worker Starting"
echo "============================================="
echo "  Worker ID:    ${WORKER_ID}"
echo "  GPU Index:    ${GPU_INDEX}"
echo "  gRPC Port:    ${GRPC_PORT}"
echo "  Metrics Port: ${METRICS_PORT}"
echo "  Log Level:    ${RUST_LOG:-info}"
echo "============================================="

# Launch the worker
exec /usr/local/bin/ai-worker \
    --worker-id "${WORKER_ID}" \
    --port "${GRPC_PORT}" \
    --metrics-port "${METRICS_PORT}" \
    --gpu-ids "${GPU_INDEX}" \
    "$@"
