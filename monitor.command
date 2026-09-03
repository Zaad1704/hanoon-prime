#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  monitor.command — HANOON PRIME 3.0 — Live Monitor
#
#  Shows running process status + tails the primary bot log.
#  Usage:  bash monitor.command
#  Stop:   Ctrl+C
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail

BOT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$BOT_ROOT/runtime/pids"
LOG_DIR="$BOT_ROOT/logs"
BOT_LOG="$LOG_DIR/hanoon_prime.log"

log()  { echo -e "\033[1;36m[$(date '+%H:%M:%S')]\033[0m $*"; }
ok()   { echo -e "\033[32m[$(date '+%H:%M:%S')]\033[0m ✅ $*"; }
warn() { echo -e "\033[33m[$(date '+%H:%M:%S')]\033[0m ⚠️  $*"; }
hdr()  { echo -e "\033[1;37m$*\033[0m"; }

# Filter noisy ib_insync messages
FILTER='grep --line-buffered -v -E "(commissionReport|Error 10147|Error 10092|Deep market data|execDetails|orderStatus|PendingNew|PendingCancel|ApiCancelled)"'

# ── Process status ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━████
show_status() {
  echo ""
  hdr "═══════════════════════════════════════════════════════════════"
  hdr "  HANOON PRIME 3.0 — RUNNING PROCESSES"
  hdr "═══════════════════════════════════════════════════════════════"
  echo ""

  # Prime bot
  if [[ -f "$PID_DIR/hanoon_prime.pid" ]]; then
    _pid=$(cat "$PID_DIR/hanoon_prime.pid" 2>/dev/null || echo "")
    if [[ -n "$_pid" ]] && kill -0 "$_pid" 2>/dev/null; then
      ok "Prime bot  ✓ PID $_pid"
    else
      warn "Prime bot  ✗ PID file stale"
    fi
  else
    _pid=$(pgrep -f "hanoon_prime.cli" 2>/dev/null | head -1)
    if [[ -n "$_pid" ]]; then
      ok "Prime bot  ✓ PID $_pid"
    else
      warn "Prime bot  ✗ Not running"
    fi
  fi

  # IB watchdog
  _wd_pid=$(pgrep -f "ib_gateway_watchdog" | head -1)
  if [[ -n "$_wd_pid" ]]; then
    ok "IB watchdog ✓ PID $_wd_pid"
  else
    warn "IB watchdog ✗ Not running"
  fi

  # HALIM serve
  _halim_pid=$(pgrep -f "halim/halim/serve.py" | head -1)
  if [[ -n "$_halim_pid" ]]; then
    ok "HALIM serve ✓ PID $_halim_pid"
  else
    warn "HALIM serve ✗ Not running"
  fi

  # Cloudflare tunnel
  _cf_pid=$(pgrep -f "cloudflared tunnel.*run" | head -1)
  if [[ -n "$_cf_pid" ]]; then
    ok "Cloudflared ✓ PID $_cf_pid"
  else
    warn "Cloudflared ✗ Not running"
  fi

  # TelemetryAPI
  if curl -sf --max-time 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    ok "Telemetry   ✓ http://127.0.0.1:8080"
    curl -sf --max-time 2 "http://127.0.0.1:8080/health" 2>&1 | python3 -c \
      "import sys,json; d=json.load(sys.stdin); print('  Positions:', len(d.get('positions',[])), '| Tickers:', len(d.get('tickers',[])))" 2>/dev/null || true
  else
    warn "Telemetry   ✗ Not responding"
  fi

  # HALIM health
  if curl -sf --max-time 2 "http://127.0.0.1:8765/health" >/dev/null 2>&1; then
    ok "HALIM API   ✓ http://127.0.0.1:8765"
  else
    warn "HALIM API   ✗ Not responding"
  fi

  echo ""
}

# ── Main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
show_status

hdr "═══════════════════════════════════════════════════════════════"
hdr "  Tailing: $BOT_LOG  (Ctrl+C to exit)"
hdr "═══════════════════════════════════════════════════════════════"
echo ""

# Ensure log file exists
touch "$BOT_LOG" 2>/dev/null

# Use tmux if available for split-pane view
if command -v tmux >/dev/null 2>&1 && [[ -z "${TMUX:-}" ]]; then
  _session="hanoon_monitor"
  tmux kill-session -t "$_session" 2>/dev/null || true

  # Main bot log (left pane)
  tmux new-session -d -s "$_session" "tail -f '$BOT_LOG' 2>/dev/null | $FILTER"

  # IB watchdog log (right pane)
  tmux split-window -t "$_session" -h "tail -f '$LOG_DIR/ib_gateway_watchdog.log' 2>/dev/null | $FILTER" 2>/dev/null || true

  tmux select-layout -t "$_session" tiled 2>/dev/null
  tmux attach-session -t "$_session"
else
  # Fallback: plain tail
  tail -f "$BOT_LOG" 2>/dev/null | $FILTER
fi
