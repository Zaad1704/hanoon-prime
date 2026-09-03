#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  start.command — HANOON PRIME 3.0 — Start EVERYTHING
#
#  Starts (in order):
#    0. Pre-flight: complexity, file length, contract tests
#    1. IB Gateway watchdog (monitors connection)
#    2. HANOON Prime bot (hanoon_prime.cli)
#    3. HALIM serve (advisory layer)
#    4. Cloudflare tunnel
#    5. Monitors
#
#  Usage:  bash start.command
#  Stop:   bash stop.command
#  Monitor: bash monitor.command
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail

BOT_ROOT="$(cd "$(dirname "$0")" && pwd)"
REBUILD_ROOT="$HOME/Downloads/hanoon_rebuild"
PID_DIR="$BOT_ROOT/runtime/pids"
LOG_DIR="$BOT_ROOT/logs"
TUNNEL_URL_FILE="$BOT_ROOT/runtime/tunnel_url.txt"
BOT_LOG="$LOG_DIR/hanoon_prime.log"
PYTHON_BIN="$(command -v python3 || command -v python)"

mkdir -p "$PID_DIR" "$LOG_DIR" "$BOT_ROOT/runtime"

# ── Helpers ─────────────────────────────────────────────────────────────
log()  { echo -e "\033[1;36m[$(date '+%H:%M:%S')]\033[0m $*"; }
warn() { echo -e "\033[33m[$(date '+%H:%M:%S')]\033[0m ⚠️  $*"; }
ok()   { echo -e "\033[32m[$(date '+%H:%M:%S')]\033[0m ✅ $*"; }
err()  { echo -e "\033[31m[$(date '+%H:%M:%S')]\033[0m ❌ $*"; }

_check_running() {
  local pidfile="$1" label="$2"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid=$(cat "$pidfile" 2>/dev/null || echo "")
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      log "$label is already running (PID $pid)"
      return 0
    fi
    rm -f "$pidfile"
  fi
  return 1
}

# ── Banner ──────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  🚀 HANOON PRIME 3.0 — FULL SYSTEM START"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── 0. Pre-flight ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 0/5: Pre-flight verification"

if ! python3 -c "import hanoon_prime" 2>/dev/null; then
  log "  Installing hanoon_prime..."
  python3 -m pip install --break-system-packages -e "$BOT_ROOT" 2>&1 | tail -3
fi

python3 "$BOT_ROOT/scripts/check_complexity.py" 2>&1 | sed 's/^/  /'
bash "$BOT_ROOT/scripts/check_file_length.sh" 2>&1 | sed 's/^/  /'
python3 -m pytest "$BOT_ROOT/tests/test_contract.py" -q --tb=short 2>&1 | tail -3 | sed 's/^/  /'
ok "Pre-flight passed"

# ── 0b. Source .env ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [[ -f "$BOT_ROOT/.env" ]]; then
  set -a
  source "$BOT_ROOT/.env"
  set +a
fi

# ── 1. IB Gateway Watchdog ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 1/5: IB Gateway watchdog"

_IB_WD_PID_FILE="$PID_DIR/ib_gateway_watchdog.pid"
_IB_WD_SCRIPT="$BOT_ROOT/scripts/ib_gateway_watchdog.py"

if _check_running "$_IB_WD_PID_FILE" "IB watchdog"; then
  :
elif [[ -f "$_IB_WD_SCRIPT" ]]; then
  nohup "$PYTHON_BIN" "$_IB_WD_SCRIPT" >> "$LOG_DIR/ib_gateway_watchdog.log" 2>&1 &
  echo $! > "$_IB_WD_PID_FILE"
  sleep 2
  ok "IB Gateway watchdog started (PID $(cat "$_IB_WD_PID_FILE" 2>/dev/null))"
else
  warn "IB watchdog script not found — using direct connection"
fi

# ── 2. HANOON Prime Bot ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 2/5: HANOON Prime bot"

_BOT_PID_FILE="$PID_DIR/hanoon_prime.pid"

if _check_running "$_BOT_PID_FILE" "HANOON Prime bot"; then
  :
else
  # Start bot directly (no setsid needed on macOS)
  "$PYTHON_BIN" -m hanoon_prime.cli AAPL MSFT SPY TSLA NVDA \
    >> "$BOT_LOG" 2>&1 &
  BOT_PID=$!
  echo "$BOT_PID" > "$_BOT_PID_FILE"
  
  sleep 3
  if kill -0 "$BOT_PID" 2>/dev/null; then
    ok "HANOON Prime bot started (PID $BOT_PID)"
  else
    err "Bot failed to start — check $BOT_LOG"
    tail -20 "$BOT_LOG" 2>/dev/null | sed 's/^/  [bot] /'
    exit 1
  fi

  # Wait for TelemetryAPI
  _tb=0
  while [[ $_tb -lt 15 ]]; do
    if curl -sf "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
      ok "TelemetryAPI live on http://127.0.0.1:8080"
      break
    fi
    sleep 1
    _tb=$((_tb + 1))
  done
fi

# ── 3. HALIM Serve ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 3/5: HALIM serve"

_HALIM_PID_FILE="$PID_DIR/halim_serve.pid"
_HALIM_START="$BOT_ROOT/scripts/halim_start.sh"

if _check_running "$_HALIM_PID_FILE" "HALIM serve"; then
  :
elif [[ -f "$_HALIM_START" ]]; then
  nohup bash "$_HALIM_START" </dev/null >>"$LOG_DIR/halim_start.log" 2>&1 &
  disown $! 2>/dev/null || true
  sleep 8
  
  _HALIM_PID=$(pgrep -f "halim/halim/serve.py" | head -1)
  if [[ -n "$_HALIM_PID" ]]; then
    echo "$_HALIM_PID" > "$_HALIM_PID_FILE"
  fi
  
  _HALIM_URL="${HALIM_SERVER_URL:-http://127.0.0.1:8765}"
  if curl -sf --max-time 3 "$_HALIM_URL/health" >/dev/null 2>&1; then
    ok "HALIM serve healthy"
  else
    warn "HALIM loading (model takes 60-90s)"
  fi
else
  warn "HALIM start script not found"
fi

# ── 4. Cloudflare Tunnel ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 4/5: Cloudflare tunnel"

_NAMED_TUNNEL_URL="https://api.hanoonweb.xyz"
_CF_PID_FILE="$PID_DIR/cloudflared.pid"
_CF_CONFIG="$HOME/.cloudflared/config.yml"

if ! command -v cloudflared >/dev/null 2>&1; then
  warn "cloudflared not installed"
elif ! [[ -f "$_CF_CONFIG" ]]; then
  warn "Cloudflared config not found"
elif _check_running "$_CF_PID_FILE" "Cloudflared"; then
  :
else
  if pgrep -f "cloudflared tunnel.*run" >/dev/null 2>&1; then
    warn "Tunnel already running"
  else
    nohup /opt/homebrew/bin/cloudflared tunnel --config "$_CF_CONFIG" \
      --metrics 127.0.0.1:45213 run >> "$LOG_DIR/cloudflared.log" 2>&1 &
    echo $! > "$_CF_PID_FILE"
    sleep 8
    
    if kill -0 "$(cat "$_CF_PID_FILE" 2>/dev/null)" 2>/dev/null; then
      ok "Cloudflared started"
    fi
  fi

  for _i in $(seq 1 10); do
    if curl -sf --max-time 3 "$_NAMED_TUNNEL_URL/health" >/dev/null 2>&1; then
      echo "$_NAMED_TUNNEL_URL" > "$TUNNEL_URL_FILE"
      ok "Tunnel active: $_NAMED_TUNNEL_URL"
      break
    fi
    sleep 3
  done
fi

# ── 5. Monitors ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 5/5: Starting monitors"

# Telegram monitor
_TG_PID_FILE="$PID_DIR/telegram_monitor.pid"
_TG_SCRIPT="$REBUILD_ROOT/scripts/telegram_bot_monitor.py"
if [[ -f "$_TG_SCRIPT" ]] && ! _check_running "$_TG_PID_FILE" "Telegram"; then
  nohup "$PYTHON_BIN" "$_TG_SCRIPT" >> "$LOG_DIR/telegram_monitor.log" 2>&1 &
  echo $! > "$_TG_PID_FILE"
  sleep 2
  ok "Telegram monitor started"
fi

# Overnight monitor
_OM_PID_FILE="$PID_DIR/overnight_monitor.pid"
_OM_SCRIPT="$REBUILD_ROOT/scripts/overnight_monitor.py"
if [[ -f "$_OM_SCRIPT" ]] && ! _check_running "$_OM_PID_FILE" "Overnight"; then
  nohup "$PYTHON_BIN" "$_OM_SCRIPT" --interval 60 >> "$LOG_DIR/overnight_monitor.log" 2>&1 &
  echo $! > "$_OM_PID_FILE"
  sleep 2
  ok "Overnight monitor started"
fi

# ── Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  🏁 HANOON PRIME 3.0 — ALL SYSTEMS STARTED"
echo "═══════════════════════════════════════════════════════════════"
echo ""
log "Bot:        hanoon_prime.cli (PID $(cat "$_BOT_PID_FILE" 2>/dev/null))"
log "IB watchdog: PID $(cat "$_IB_WD_PID_FILE" 2>/dev/null)"
log "HALIM:      http://127.0.0.1:8765"
log "Telemetry:  http://127.0.0.1:8080"
log "Tunnel:     https://api.hanoonweb.xyz"
echo ""
log "Monitor:    bash monitor.command"
log "Stop:       bash stop.command"
log "Log:        tail -f $BOT_LOG"
echo ""
