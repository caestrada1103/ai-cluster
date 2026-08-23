#!/usr/bin/env bash
# Starts stable-diffusion.cpp's sd-server in Docker. See docs/diffusion.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && set -a && . "$REPO_ROOT/.env" && set +a

NAME="${SD_CONTAINER_NAME:-ai-sd-server}"
PORT="${SD_SERVER_PORT:-8090}"
HOST="${SD_SERVER_HOST:-127.0.0.1}"
MODELS_DIR="${SD_MODELS_DIR:-$REPO_ROOT/models/diffusion}"
OUTPUT_DIR="${SD_OUTPUT_DIR:-$REPO_ROOT/data/diffusion}"
MODEL="${SD_MODEL:-}"
EXTRA_ARGS="${SD_EXTRA_ARGS:-}"

# Match the tag to the GPU: -spark is built for GB10 (sm_121).
TAG="${SD_IMAGE_TAG:-master-cuda-spark}"
IMAGE="${SD_IMAGE:-ghcr.io/leejet/stable-diffusion.cpp:$TAG}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--model PATH] [--name NAME] [--tag TAG] [--port N] [--stop] [--logs]

  --model PATH   Single-file model, relative to \$SD_MODELS_DIR. Omit for video
                 models, which pass --diffusion-model/--vae/--t5xxl via
                 \$SD_EXTRA_ARGS instead.
  --name NAME    Container name (default $NAME). One model per instance.
  --tag TAG      master-cuda-spark (GB10) | master-cuda | master-vulkan | master-sycl
  --port N       Host port (default $PORT)
  --stop         Stop and remove the container
  --logs         Follow container logs
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --tag) TAG="$2"; IMAGE="ghcr.io/leejet/stable-diffusion.cpp:$TAG"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --stop) docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME"; exit 0 ;;
    --logs) exec docker logs -f "$NAME" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; usage >&2; exit 1 ;;
  esac
done

MODEL_ARGS=()
if [ -n "$MODEL" ]; then
  if [ ! -f "$MODELS_DIR/$MODEL" ]; then
    echo "Not found: $MODELS_DIR/$MODEL" >&2
    exit 1
  fi
  MODEL_ARGS=(-m "/models/$MODEL")
elif [ -z "$EXTRA_ARGS" ]; then
  echo "No model. Pass --model, or set SD_MODEL / SD_EXTRA_ARGS in .env; see docs/diffusion.md." >&2
  exit 1
fi

GPU_ARGS=()
case "$TAG" in
  *cuda*) GPU_ARGS=(--gpus all) ;;
  *vulkan*|*sycl*|*musa*) GPU_ARGS=(--device /dev/dri) ;;
esac

mkdir -p "$MODELS_DIR" "$OUTPUT_DIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
# -w keeps relative model scans off /proc; sd-server walks "." for LoRAs.
docker run -d --name "$NAME" --restart unless-stopped \
  --entrypoint /sd-server -w /models \
  "${GPU_ARGS[@]}" \
  -p "$HOST:$PORT:8080" \
  -v "$MODELS_DIR:/models:ro" \
  -v "$OUTPUT_DIR:/output" \
  "$IMAGE" \
  --listen-ip 0.0.0.0 --listen-port 8080 --lora-model-dir /models/loras \
  "${MODEL_ARGS[@]}" $EXTRA_ARGS

echo "$NAME up on $HOST:$PORT  (model: ${MODEL:-via SD_EXTRA_ARGS}, image: $IMAGE)"
echo "Logs: $(basename "$0") --logs"
