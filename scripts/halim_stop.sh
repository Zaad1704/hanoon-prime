#!/usr/bin/env bash
# Stop HALIM serve for HANOON PRIME 3.0.
# Sends SIGTERM first (for safe MLX/Metal buffer release), then SIGKILL after grace,
# then removes the launchd registration (label com.hanoon.halim).
set -uo pipefail

# Halim self-contained in this repo — see halim_start.sh.
HALIM_REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$HALIM_REPO"
PID_DIR="$ROOT/runtime/pids"
LOG_DIR="$ROOT/logs"
LABEL="com.hanoon.halim"
mkdir -p "$PID_DIR" "$LOG_DIR"

# Get the serve PID (launchd first, then PID file, then pgrep)
PID=""
if launchctl list 2>/dev/null | grep -q "$LABEL"; then
  PID=$(launchctl list 2>/dev/null | grep "$LABEL" | awk '{print $1}')
elif [[ -f "$PID_DIR/halim_serve.pid" ]]; then
  PID=$(tr -d '[:space:]' < "$PID_DIR/halim_serve.pid" 2>/dev/null || echo "")
fi
if [[ -z "$PID" ]]; then
  PID=$(pgrep -f "halim/halim/serve.py" 2>/dev/null | head -1 || true)
fi

if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  echo "Stopping HALIM serve (PID $PID)..."
  kill -TERM "$PID" 2>/dev/null || true
  _waited=0
  while [[ $_waited -lt 10 ]]; do
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
    sleep 1
    _waited=$((_waited + 1))
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "HALIM serve ignored SIGTERM — SIGKILL"
    kill -9 "$PID" 2>/dev/null || true
  fi
fi

# Remove launchd registration
launchctl remove "$LABEL" 2>/dev/null || true
rm -f "$PID_DIR/halim_serve.pid"

# Fallback: kill by process pattern
pkill -f "halim/halim/serve.py" 2>/dev/null || true

echo "HALIM serve stopped"
