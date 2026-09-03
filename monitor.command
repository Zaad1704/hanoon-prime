#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  monitor.command — HANOON PRIME 3.0 — Live Monitor
#
#  Shows running process status + tails the primary bot log.
#  Uses tmux if available; falls back to plain tail.
#
#  Usage:  bash monitor.command
#  Stop:   Ctrl+C
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail

BOT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$BOT_ROOT/runtime/pids"
LOG_DIR="$BOT_ROOT/logs"

log()  { echo -e "\033[1;36m[$(date '+%H:%M:%S')]\033[0m $*"; }
ok()   { echo -e "\033[32m[$(date '+%H:%M:%S')]\033[0m ✅ $*"; }
warn() { echo -e "\033[33m[$(date '+%H:%M:%S')]\033[0m ⚠️  $*"; }
hdr()  { echo -e "\033[1;37m$*\033[0m"; }

# Filter noisy ib_insync messages
LOG_FILTER='grep -v -E "(commissionReport|Error 10147.*OrderId 0.*not found|Error 10092|Deep market data|execDetails|orderStatus)" || true'

# ── Process status ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━████
show_status() {
  echo ""
  hdr "═══════════════════════════════════════════════════════════════"
  hdr "  HANOON PRIME 3.0 — RUNNING PROCESSES"
  hdr "═══════════════════════════════════════════════════════════════"
  echo ""

  local found=false

  # Prime bot
  if [[ -f "$PID_DIR/hanoon_prime.pid" ]]; then
    _pid=$(cat "$PID_DIR/hanoon_prime.pid" 2>/dev/null || echo "")
    if [[ -n "$_pid" ]] && kill -0 "$_pid" 2>/dev/null; then
      ok "Prime bot  ✓ PID $_pid  (hanoon_prime.cli)"
      found=true
    else
      warn "Prime bot  ✗ PID file stale"
    fi
  else
    _pid=$(pgrep -f "hanoon_prime.cli" 2>/dev/null | head -1)
    if [[ -n "$_pid" ]]; then
      ok "Prime bot  ✓ PID $_pid  (running but no PID file)"
      found=true
    else
      warn "Prime bot  ✗ Not running"
    fi
  fi

  # IB Gateway watchdog
  if pgrep -f "ib-gateway-watchdog" >/dev/null 2>&1; then
    _pid=$(pgrep -f "ib-gateway-watchdog" | head -1)
    ok "IB watchdog  ✓ PID $_pid"
    found=true
  else
    warn "IB watchdog  ✗ Not running"
  fi

  # HALIM serve
  _halim_pid=$(pgrep -f "halim/halim/serve.py" | head -1)
  if [[ -n "$_halim_pid" ]]; then
    ok "HALIM serve  ✓ PID $_halim_pid"
    found=true
  else
    warn "HALIM serve  ✗ Not running"
  fi

  # Cloudflare tunnel
  _cf_pid=$(pgrep -f "cloudflared tunnel.*run" | head -1)
  if [[ -n "$_cf_pid" ]]; then
    ok "Cloudflared  ✓ PID $_cf_pid"
    found=true
  else
    warn "Cloudflared  ✗ Not running"
  fi

  # Telegram monitor
  _tg_pid=$(pgrep -f "telegram_bot_monitor" | head -1)
  if [[ -n "$_tg_pid" ]]; then
    ok "Telegram     ✓ PID $_tg_pid"
    found=true
  else
    warn "Telegram     ✗ Not running"
  fi

  # TelemetryAPI
  if curl -sf --max-time 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    ok "Telemetry    ✓ http://127.0.0.1:8080"
    curl -sf --max-time 2 "http://127.0.0.1:8080/health" 2>&1 | python3 -c \
      "import sys,json; d=json.load(sys.stdin); print('  Status:', d.get('status','?'), 'Positions:', len(d.get('positions',[])))" 2>/dev/null || true
  else
    warn "Telemetry    ✗ Not responding"
  fi

  # HALIM health
  if curl -sf --max-time 2 "http://127.0.0.1:8765/health" >/dev/null 2>&1; then
    ok "HALIM API    ✓ http://127.0.0.1:8765"
  else
    warn "HALIM API    ✗ Not responding"
  fi

  if [[ "$found" == "false" ]]; then
    warn "No HANOON processes found. Start with: bash start.command"
  fi
  echo ""
}

# ── Main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
show_status

hdr "═══════════════════════════════════════════════════════════════"
hdr "  Tail: $LOG_DIR/hanoon_prime.log  (Ctrl+C to exit)"
hdr "═══════════════════════════════════════════════════════════════"
echo ""

# Use tmux if available for a nicer split-pane experience
if command -v tmux >/dev/null 2>&1 && [[ -z "${TMUX:-}" ]]; then
  if [[ ! -f "$LOG_DIR/hanoon_prime.log" ]]; then
    warn "No log file at $LOG_DIR/hanoon_prime.log"
    log "Waiting for bot to start... (run: bash start.command)"
    touch "$LOG_DIR/hanoon_prime.log"
  fi

  _session="hanoon_monitor"
  tmux kill-session -t "$_session" 2>/dev/null || true

  # Main bot log (left pane) - filter noisy ib_insync messages
  tmux new-session -d -s "$_session" \
    "tail -f '$LOG_DIR/hanoon_prime.log' 2>/dev/null | grep --line-buffered -v -E '(commissionReport|Error 10147.*OrderId 0.*not found|Error 10092|Deep market data|execDetails|orderStatus)'"

  # HALIM log (right pane, if exists)
  if [[ -f "$LOG_DIR/halim_serve.log" ]]; then
    tmux split-window -t "$_session" -h \
      "tail -f '$LOG_DIR/halim_serve.log' 2>/dev/null | grep --line-buffered -v -E '(commissionReport|Error 10147.*OrderId 0.*not found)'"
  fi

  # Health check bottom pane (if overnight monitor exists)
  if [[ -f "$LOG_DIR/overnight_monitor.log" ]]; then
    tmux split-window -t "$_session" -v \
      "tail -f '$LOG_DIR/overnight_monitor.log' 2>/dev/null | grep --line-buffered -v -E '(commissionReport|Error 10147.*OrderId 0.*not found)'"
  fi

  tmux select-layout -t "$_session" tiled 2>/dev/null
  tmux attach-session -t "$_session"
else
  # Fallback: plain tail with color highlighting
  if [[ -f "$LOG_DIR/hanoon_prime.log" ]]; then
    tail -f "$LOG_DIR/hanoon_prime.log" 2>/dev/null | grep --line-buffered -v -E "(commissionReport|Error 10147.*OrderId 0.*not found|Error 10092|Deep market data|execDetails|orderStatus)"
  else
    warn "No log file at $LOG_DIR/hanoon_prime.log"
    log "Waiting for bot to start... (run: bash start.command)"
    touch "$LOG_DIR/hanoon_prime.log"
    tail -f "$LOG_DIR/hanoon_prime.log" 2>/dev/null
  fi
fi
