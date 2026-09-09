#!/usr/bin/env bash
# stop.command — graceful teardown of the HANOON PRIME 3.0 stack:
#   production monitor → trading bot → HALIM serve.
# SIGTERM first (graceful shutdown), SIGKILL after grace, stale pidfiles
# removed. A pgrep fallback catches a bot started without a pidfile.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT/runtime/pids"
BOT_PID="$PID_DIR/hanoon_prime.pid"
MONITOR_PID="$PID_DIR/production_monitor.pid"

_STOP_PIDFILE() { # $1 label, $2 pidfile, [$3 pgrep pattern]
  local label="$1" pidfile="$2" pattern="${3:-}" pid=""
  if [[ -s "$pidfile" ]]; then
    pid=$(tr -d '[:space:]' < "$pidfile" 2>/dev/null || echo "")
  fi
  if [[ -z "$pid" || "$pid" == "0" ]] && [[ -n "$pattern" ]]; then
    pid=$(pgrep -f "$pattern" 2>/dev/null | head -1 || echo "")
  fi
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $label (PID $pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    _w=0
    while [[ $_w -lt 10 ]] && kill -0 "$pid" 2>/dev/null; do sleep 1; _w=$((_w + 1)); done
    if [[ $_w -ge 10 ]] && kill -0 "$pid" 2>/dev/null; then
      echo "  $label ignored SIGTERM — SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
    fi
  else
    echo "$label not running (pidfile ${pidfile:-none})"
  fi
  rm -f "$pidfile"
  if [[ -n "$pattern" ]]; then
    pkill -f "$pattern" 2>/dev/null || true
  fi
}

echo "== HANOON PRIME stack — stopping =="

_STOP_PIDFILE "guardian monitor" "$MONITOR_PID" "scripts/production_monitor.py --daemon"

# Bot may be pidfile-less (legacy nohup start) → pgrep fallback.
_STOP_PIDFILE "trading bot" "$BOT_PID" "hanoon_prime.cli"

"$ROOT/scripts/halim_stop.sh"

echo
echo "== Stack status =="
curl -sf --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && echo "  Bot     : still ONLINE (check)?" || echo "  Bot     : stopped"
curl -sf --max-time 2 http://127.0.0.1:8765/health >/dev/null 2>&1 && echo "  HALIM   : still ONLINE" || echo "  HALIM   : stopped"
pgrep -f "production_monitor.py --daemon" >/dev/null 2>&1 && echo "  Monitor : still running" || echo "  Monitor : stopped"