#!/usr/bin/env bash
# start.command — bring the full HANOON PRIME 3.0 stack online:
#   HALIM serve → trading bot → production guardian/bug-catcher monitor.
#
# The bot and monitor are detached via scripts/launch_detached.py (double-fork
# + setsid) so they survive this terminal closing while keeping Downloads
# access (launchd would be TCC-blocked on ~/Downloads). PIDs live in
# runtime/pids, logs in logs/. Re-running is idempotent: anything already up
# is left alone.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT/runtime/pids"
LOG_DIR="$ROOT/logs"
mkdir -p "$PID_DIR" "$LOG_DIR"
BOT_PID="$PID_DIR/hanoon_prime.pid"
MONITOR_PID="$PID_DIR/production_monitor.pid"
PYTHON_BIN="$ROOT/.venv/bin/python"
TELEMETRY="http://127.0.0.1:8080"

_PID_ALIVE() { # $1 pidfile → 0 if running
  [[ -s "$1" ]] || return 1
  local pid; pid=$(tr -d '[:space:]' < "$1" 2>/dev/null || echo "")
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

echo "== HANOON PRIME stack — starting =="

# 1) HALIM serve (idempotent; already-running detected internally)
"$ROOT/scripts/halim_start.sh"

# 1.5) Re-anchor the journal chain if broken — only valid between stop and start,
#      while the bot is down (never while live). Failure is non-fatal.
echo "== re-anchoring journal chain =="
"$PYTHON_BIN" -m hanoon_prime.inspection reanchor >>"$LOG_DIR/reanchor.log" 2>&1 || echo "warn: re-anchor step failed (rc $?)"

# 2) Trading bot — only if not already serving telemetry
if curl -sf --max-time 2 "$TELEMETRY/health" >/dev/null 2>&1; then
  echo "Bot already online (telemetry $TELEMETRY healthy) — skipping start"
elif _PID_ALIVE "$BOT_PID"; then
  echo "Bot already running (PID $(tr -d '[:space:]' < "$BOT_PID")) — skipping start"
else
  rm -f "$BOT_PID"
  echo "Starting bot (detached, log: $LOG_DIR/hanoon_prime.log)..."
  "$PYTHON_BIN" "$ROOT/scripts/launch_detached.py" "$BOT_PID" "$LOG_DIR/hanoon_prime.log" \
    "$PYTHON_BIN" -m hanoon_prime.cli </dev/null >>"$LOG_DIR/hanoon_start.log" 2>&1
  _waited=0
  while ! curl -sf --max-time 2 "$TELEMETRY/health" >/dev/null 2>&1; do
    _waited=$((_waited + 2))
    if [[ $_waited -ge 120 ]]; then
      echo "⚠️  Bot not healthy after 120s — tail $LOG_DIR/hanoon_prime.log"
      break
    fi
    sleep 2
  done
  echo "Bot PID $(tr -d '[:space:]' < "$BOT_PID" 2>/dev/null) (after ${_waited}s)"
fi

# 3) Production guardian + bug catcher (idempotent, pidfile-guarded)
if _PID_ALIVE "$MONITOR_PID"; then
  echo "Monitor already running (PID $(tr -d '[:space:]' < "$MONITOR_PID")) — skipping start"
else
  rm -f "$MONITOR_PID"
  echo "Starting guardian monitor (log: $LOG_DIR/production_monitor.log)..."
  "$PYTHON_BIN" "$ROOT/scripts/launch_detached.py" "$MONITOR_PID" "$LOG_DIR/production_monitor.log" \
    "$PYTHON_BIN" "$ROOT/scripts/production_monitor.py" --daemon </dev/null >>"$LOG_DIR/production_monitor_start.log" 2>&1
  echo "Monitor PID $(tr -d '[:space:]' < "$MONITOR_PID" 2>/dev/null)"
fi

echo
echo "== Stack status =="
HALIM_OK=$("$ROOT/.venv/bin/python" -c "
import urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2); print('healthy')
except Exception: print('DOWN')")
BOT_OK=$(curl -sf --max-time 2 "$TELEMETRY/health" >/dev/null 2>&1 && echo online || echo DOWN)
MON_OK=$(_PID_ALIVE "$MONITOR_PID" && echo running || echo DOWN)
echo "  HALIM   : $HALIM_OK"
echo "  Bot     : $BOT_OK"
echo "  Monitor : $MON_OK"
echo "  To stop:  double-click scripts/stop.command"