#!/usr/bin/env bash
# Start HALIM serve for HANOON PRIME 3.0.
# This starts the external MLX/Metal model server that HalimBridge connects to.
#
# The serve is detached via scripts/launch_detached.py (double-fork + setsid)
# so it survives the launching terminal/session closing while keeping
# Downloads access (launchd would be TCC-blocked on ~/Downloads).
set -uo pipefail

# Halim is self-contained inside the HANOON PRIME repo (2026-09-03, moved
# from hanoon_rebuild): the serve code lives in halim/halim, the Kaggle-trained
# weights + LoRA in halim/data/checkpoints (qwen3.5_4b_v2 merged).
HALIM_REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$HALIM_REPO"
PID_DIR="$ROOT/runtime/pids"
LOG_DIR="$ROOT/logs"
PID_FILE="$PID_DIR/halim_serve.pid"
HALIM_URL="${HALIM_SERVER_URL:-http://127.0.0.1:${HALIM_SERVE_PORT:-8765}}"
mkdir -p "$PID_DIR" "$LOG_DIR"

# Check if already running
if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "HALIM serve already running (PID $PID)"
    exit 0
  fi
  rm -f "$PID_FILE"
fi
if pgrep -f "halim/halim/serve.py" >/dev/null 2>&1; then
  echo "Stale HALIM serve found — killing before restart"
  pkill -f "halim/halim/serve.py" 2>/dev/null || true
  sleep 2
fi

if [[ ! -f "$HALIM_REPO/scripts/halim_serve.sh" ]]; then
  echo "WARNING: HALIM serve script not found at $HALIM_REPO/scripts/halim_serve.sh"
  echo "HalimBridge will run in advisory-only mode (no external LM)"
  exit 0
fi

echo "Starting HALIM serve (detached daemon)..."
echo "  Logs: $LOG_DIR/halim_serve.log"

PYTHON_BIN=$(command -v python3 || echo /opt/homebrew/bin/python3)
# launch_detached runs the launcher in its own session. The launcher sources
# the repo env (HALIM_MODEL_PATH, MLX_GPU_CACHE_LIMIT_MB, HALIM_LM_BACKEND=mlx)
# then execs halim_serve.sh, which execs `nice python3 halim/halim/serve.py`.
# stdio MUST be redirected to a file: the detached serve daemon inherits
# these FDs for life — if a caller captures stdout/stderr via pipes, the
# daemon holds them open forever and deadlocks the caller's pipe drain.
"$PYTHON_BIN" "$ROOT/scripts/launch_detached.py" "$PID_FILE" "$LOG_DIR/halim_serve.log" \
  /bin/bash "$ROOT/scripts/halim_serve_launcher.sh" \
  </dev/null >>"$LOG_DIR/halim_start.log" 2>&1

PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
if [[ -n "$PID" ]]; then
  echo "HALIM serve started (PID $PID)"
else
  echo "⚠️  HALIM serve failed to start — check $LOG_DIR/halim_serve.log"
  tail -5 "$LOG_DIR/halim_serve.log" 2>/dev/null
fi

# Wait for /health to come up (model load can take 60-90s)
for _i in $(seq 1 60); do
  if curl -sf --max-time 2 "$HALIM_URL/health" >/dev/null 2>&1; then
    echo "✅ HALIM serve healthy at $HALIM_URL (after ${_i}x2s)"
    exit 0
  fi
  sleep 2
done
echo "⚠️  HALIM serve not healthy after 120s — check $LOG_DIR/halim_serve.log"
exit 0
