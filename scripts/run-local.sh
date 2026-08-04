#!/usr/bin/env bash
# Start the worker and coordinator locally on loopback. Ctrl-C stops both.
#
#   ./scripts/run-local.sh              # start
#   ./scripts/run-local.sh --build      # rebuild the worker first
#   ./scripts/run-local.sh --model NAME # also load a model once healthy
#
# See docs/deployment.md for the container path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKER_PORT="${WORKER_PORT:-50051}"
COORDINATOR_PORT="${COORDINATOR_PORT:-8000}"
API_KEY="${COORDINATOR_API_KEYS:-sk-local-dev}"
FEATURES="${WORKER_FEATURES:-wgpu,llamacpp,llamacpp-cuda}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
BUILD=0
LOAD_MODEL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --build) BUILD=1; shift ;;
        --model) LOAD_MODEL="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,8p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }

# --- prerequisites ---------------------------------------------------------
[ -d .venv ] || die "no .venv — run: python3 -m venv .venv && . .venv/bin/activate && pip install -r coordinator/requirements-dev.txt"

if [ "$BUILD" = "1" ]; then
    [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
    command -v cargo >/dev/null || die "cargo not found — install rustup"
    export PROTOC="${PROTOC:-$HOME/.local/bin/protoc}"
    export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-18/lib}"
    echo "==> building worker (--features $FEATURES)"
    ( cd worker && cargo build --release --features "$FEATURES" )
fi

WORKER_BIN="$REPO_ROOT/target/release/ai-worker"
[ -x "$WORKER_BIN" ] || die "worker binary missing — rerun with --build"

# llama-server is only needed for engine = "llamaserver" models.
export PATH="$HOME/.local/bin:$PATH"
command -v llama-server >/dev/null || echo "note: llama-server not on PATH; engine=\"llamaserver\" models will fail to load"

mkdir -p "$LOG_DIR"
WORKER_LOG="$LOG_DIR/worker.local.log"
COORD_LOG="$LOG_DIR/coordinator.local.log"

# --- lifecycle -------------------------------------------------------------
WORKER_PID=""
COORD_PID=""
cleanup() {
    echo
    echo "==> stopping"
    [ -n "$COORD_PID" ] && kill "$COORD_PID" 2>/dev/null || true
    [ -n "$WORKER_PID" ] && kill "$WORKER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Both bind loopback by default; nothing is reachable off this machine.
echo "==> worker        :$WORKER_PORT   -> $WORKER_LOG"
RUST_LOG="${RUST_LOG:-info}" "$WORKER_BIN" --port "$WORKER_PORT" >"$WORKER_LOG" 2>&1 &
WORKER_PID=$!

for _ in $(seq 1 60); do
    grep -q "gRPC server listening" "$WORKER_LOG" 2>/dev/null && break
    kill -0 "$WORKER_PID" 2>/dev/null || { tail -20 "$WORKER_LOG"; die "worker exited during startup"; }
    sleep 1
done
grep -q "gRPC server listening" "$WORKER_LOG" || { tail -20 "$WORKER_LOG"; die "worker did not become ready"; }
grep -oE "Initialized GPU 0.*" "$WORKER_LOG" | sed 's/^/    /' || true

echo "==> coordinator   :$COORDINATOR_PORT   -> $COORD_LOG"
# shellcheck disable=SC1091
. .venv/bin/activate
export COORDINATOR_STATIC_WORKERS="[\"localhost:$WORKER_PORT\"]"
export COORDINATOR_API_KEYS="$API_KEY"
uvicorn coordinator.main:app --host 127.0.0.1 --port "$COORDINATOR_PORT" >"$COORD_LOG" 2>&1 &
COORD_PID=$!

for _ in $(seq 1 60); do
    curl -fsS --max-time 2 "localhost:$COORDINATOR_PORT/health" >/dev/null 2>&1 && break
    kill -0 "$COORD_PID" 2>/dev/null || { tail -20 "$COORD_LOG"; die "coordinator exited during startup"; }
    sleep 1
done
curl -fsS --max-time 2 "localhost:$COORDINATOR_PORT/health" >/dev/null 2>&1 \
    || { tail -20 "$COORD_LOG"; die "coordinator did not become ready"; }

AUTH="Authorization: Bearer $API_KEY"

if [ -n "$LOAD_MODEL" ]; then
    echo "==> loading $LOAD_MODEL (first run downloads the weights)"
    curl -fsS --max-time 3600 -H "$AUTH" -H 'Content-Type: application/json' \
        -X POST "localhost:$COORDINATOR_PORT/v1/models/load" \
        -d "{\"model_name\":\"$LOAD_MODEL\"}" | sed 's/^/    /'
    echo
fi

cat <<EOF

    ready. api key: $API_KEY

    curl -H "$AUTH" localhost:$COORDINATOR_PORT/v1/models
    curl -H "$AUTH" -X POST localhost:$COORDINATOR_PORT/v1/models/load \\
      -H 'Content-Type: application/json' -d '{"model_name":"qwen2.5-0.5b-gguf"}'
    curl -H "$AUTH" -X POST localhost:$COORDINATOR_PORT/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"qwen2.5-0.5b-gguf","messages":[{"role":"user","content":"hi"}],"max_tokens":200}'

    logs: $WORKER_LOG
          $COORD_LOG
    ctrl-c to stop both.

EOF

wait
