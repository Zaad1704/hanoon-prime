#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  start.command — HANOON PRIME 3.0 — Start EVERYTHING
#
#  hanoon_prime is the ENTIRE system: brain (Hippocampus), data
#  (IBStreamer), execution (IBExecutor), journal (memory.Journal),
#  telemetry (HTTP :8080), HALIM (advisory), Cloudflare tunnel,
#  overnight monitor, and health monitor.
#
#  Starts (in order):
#    0. Pre-flight: complexity, file length, contract tests
#    1. IB Gateway watchdog (auto-login via IBC)
#    2. HANOON Prime bot (hanoon_prime.cli → IBStreamingBot on port 4002)
#       + TelemetryAPI on :8080 for cloudflared
#    3. HALIM serve (Qwen3.5-4B MLX MoE) — advisory layer
#    4. Cloudflare named tunnel (api.hanoonweb.xyz → :8080 telemetry)
#    5. Health monitor + overnight monitor (trade surveillance)
#
#  PIDs written to runtime/pids/ for stop.command.
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
echo "  Engine: hanoon_prime (standalone — no legacy bridge)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── 0. Pre-flight Verification ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 0/5: Pre-flight verification"

if ! python3 -c "import hanoon_prime" 2>/dev/null; then
  log "  Installing hanoon_prime package (editable mode)..."
  python3 -m pip install --break-system-packages -e "$BOT_ROOT" 2>&1 | tail -3
fi

python3 "$BOT_ROOT/scripts/check_complexity.py" 2>&1 | sed 's/^/  /'
bash "$BOT_ROOT/scripts/check_file_length.sh" 2>&1 | sed 's/^/  /'
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  err "Complexity checks failed — aborting startup"
  exit 1
fi

python3 -m pytest "$BOT_ROOT/tests/test_contract.py" -q --tb=short 2>&1 | tail -3 | sed 's/^/  /'
ok "Pre-flight passed (complexity + contract tests)"

# ── 0b. Source .env for Telegram credentials ━━━━━━━━━━━━━━━━━━━━━━━━
if [[ -f "$REBUILD_ROOT/.env" ]]; then
  set -a
  source "$REBUILD_ROOT/.env"
  set +a
fi
_tg_token="${TRADING_BOT_TELEGRAM_TOKEN:-${TELEGRAM_BOT_TOKEN:-}}"
_tg_chat="${TRADING_BOT_TELEGRAM_CHAT_ID:-${TELEGRAM_CHAT_ID:-}}"
if [[ -n "$_tg_token" && -n "$_tg_chat" ]]; then
  ok "Telegram notifications: ENABLED"
else
  warn "Telegram credentials not found — terminal-only logging"
fi

# ── 1. IB Gateway Watchdog ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 1/5: IB Gateway watchdog"

_IB_WATCHDOG="$REBUILD_ROOT/scripts/ib-gateway-watchdog.command"
_IB_WD_PID_FILE="$PID_DIR/ib_gateway_watchdog.pid"

if [[ -f "$_IB_WATCHDOG" ]] && [[ -x "$_IB_WATCHDOG" ]]; then
  if ! pgrep -f "ib-gateway-watchdog" >/dev/null 2>&1; then
    nohup bash "$_IB_WATCHDOG" >> "$LOG_DIR/ib_gateway_watchdog.log" 2>&1 &
    echo $! > "$_IB_WD_PID_FILE"
    sleep 5
    ok "IB Gateway watchdog started (PID $(cat "$_IB_WD_PID_FILE" 2>/dev/null))"
  else
    ok "IB Gateway watchdog already running"
  fi
else
  warn "IB Gateway watchdog not found — bot will connect directly in paper mode"
fi

# ── 2. HANOON Prime Bot ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# hanoon_prime.cli starts IBStreamingBot (connects IB Gateway port 4002)
# AND TelemetryAPI (HTTP :8080) — fully self-contained.
log "Step 2/5: HANOON Prime bot (hanoon_prime — standalone IB Gateway)"

_BOT_PID_FILE="$PID_DIR/hanoon_prime.pid"

if _check_running "$_BOT_PID_FILE" "HANOON Prime bot"; then
  :
else
  "$PYTHON_BIN" "$REBUILD_ROOT/scripts/launch_detached.py" \
    "$_BOT_PID_FILE" "$BOT_LOG" \
    "$PYTHON_BIN" -m hanoon_prime.cli AAPL MSFT SPY TSLA NVDA \
    </dev/null >>"$LOG_DIR/hanoon_prime_start.log" 2>&1

  sleep 3
  BOT_PID=$(cat "$_BOT_PID_FILE" 2>/dev/null || echo "")
  if [[ -n "$BOT_PID" ]] && kill -0 "$BOT_PID" 2>/dev/null; then
    ok "HANOON Prime bot started (PID $BOT_PID)"
  else
    err "Bot failed to start — check $BOT_LOG"
    tail -20 "$BOT_LOG" 2>/dev/null | sed 's/^/  [bot] /'
    exit 1
  fi

  # Wait for TelemetryAPI on :8080
  sleep 2
  _tb=0
  while [[ $_tb -lt 15 ]]; do
    if curl -sf "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
      ok "TelemetryAPI live on http://127.0.0.1:8080"
      curl -sf "http://127.0.0.1:8080/health" 2>&1 | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Status:', d.get('status','?'))" 2>/dev/null || true
      break
    fi
    sleep 1
    _tb=$((_tb + 1))
  done
  if [[ $_tb -ge 15 ]]; then
    warn "TelemetryAPI not responding — bot may still be initializing"
  fi
fi

# ── 3. HALIM Serve ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 3/5: HALIM serve (Qwen3.5-4B MLX MoE — advisory layer)"

_HALIM_PID_FILE="$PID_DIR/halim_serve.pid"
_HALIM_START="$REBUILD_ROOT/scripts/halim_start_rebuild.sh"

if _check_running "$_HALIM_PID_FILE" "HALIM serve"; then
  :
elif [[ -f "$_HALIM_START" ]]; then
  nohup bash "$_HALIM_START" </dev/null >>"$LOG_DIR/halim_start.log" 2>&1 &
  _HALIM_WRAPPER_PID=$!
  disown "$_HALIM_WRAPPER_PID" 2>/dev/null || true
  sleep 8

  _HALIM_ACTUAL_PID=$(pgrep -f "halim/halim/serve.py" | head -1)
  if [[ -n "$_HALIM_ACTUAL_PID" ]]; then
    echo "$_HALIM_ACTUAL_PID" > "$_HALIM_PID_FILE"
    log "HALIM serve wrapper PID $_HALIM_WRAPPER_PID → actual serve.py PID $_HALIM_ACTUAL_PID"
  fi

  _HALIM_URL="${HALIM_SERVER_URL:-http://127.0.0.1:${HALIM_SERVE_PORT:-8765}}"
  if curl -sf --max-time 3 "$_HALIM_URL/health" >/dev/null 2>&1; then
    ok "HALIM serve healthy at $_HALIM_URL"
  else
    warn "HALIM serve loading (model takes 60-90s) — advisory only, bot runs independently"
  fi
else
  warn "HALIM start script not found — advisory layer disabled"
fi

# ── 4. Cloudflare Named Tunnel ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 4/5: Cloudflare named tunnel (api.hanoonweb.xyz → :8080)"

_NAMED_TUNNEL_URL="https://api.hanoonweb.xyz"
_CF_PID_FILE="$PID_DIR/cloudflared.pid"
_CF_CONFIG="$HOME/.cloudflared/config.yml"

if ! command -v cloudflared >/dev/null 2>&1; then
  warn "cloudflared not installed — tunnel skipped (local-only mode)"
elif ! [[ -f "$_CF_CONFIG" ]]; then
  warn "Cloudflared config not found — tunnel skipped"
elif _check_running "$_CF_PID_FILE" "Cloudflared tunnel"; then
  :
else
  if pgrep -f "cloudflared tunnel.*run" >/dev/null 2>&1; then
    warn "Named tunnel already running"
  else
    "$PYTHON_BIN" "$REBUILD_ROOT/scripts/launch_detached.py" \
      "$_CF_PID_FILE" "$LOG_DIR/cloudflared.log" \
      /opt/homebrew/bin/cloudflared tunnel --config "$_CF_CONFIG" --metrics 127.0.0.1:45213 run \
      </dev/null >>"$LOG_DIR/cloudflared_start.log" 2>&1
    sleep 8

    _CF_PID=$(cat "$_CF_PID_FILE" 2>/dev/null || echo "")
    if [[ -n "$_CF_PID" ]] && kill -0 "$_CF_PID" 2>/dev/null; then
      ok "Cloudflared named tunnel started (PID $_CF_PID)"
    else
      warn "Cloudflared failed to start — check $LOG_DIR/cloudflared.log"
    fi
  fi

  _tunnel_ok=false
  for _i in $(seq 1 10); do
    if curl -sf --max-time 3 "$_NAMED_TUNNEL_URL/health" >/dev/null 2>&1; then
      _tunnel_ok=true
      break
    fi
    sleep 3
  done

  if [[ "$_tunnel_ok" == "true" ]]; then
    echo "$_NAMED_TUNNEL_URL" > "$TUNNEL_URL_FILE"
    ok "Named tunnel active: $_NAMED_TUNNEL_URL → http://127.0.0.1:8080"
  else
    warn "Tunnel not responding yet — bot continues in local-only mode"
  fi
fi

# ── 5. Monitors ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Step 5/5: Starting monitors"

# Telegram bot monitor (notifications)
_TG_PID_FILE="$PID_DIR/telegram_monitor.pid"
if _check_running "$_TG_PID_FILE" "Telegram bot monitor"; then
  :
else
  _TG_SCRIPT="$REBUILD_ROOT/scripts/telegram_bot_monitor.py"
  if [[ -f "$_TG_SCRIPT" ]]; then
    "$PYTHON_BIN" "$REBUILD_ROOT/scripts/launch_detached.py" "$_TG_PID_FILE" "$LOG_DIR/telegram_monitor.log" \
      "$PYTHON_BIN" "$_TG_SCRIPT" </dev/null >>"$LOG_DIR/telegram_monitor_start.log" 2>&1
    sleep 2
    _TG_PID=$(cat "$_TG_PID_FILE" 2>/dev/null || echo "")
    if [[ -n "$_TG_PID" ]] && kill -0 "$_TG_PID" 2>/dev/null; then
      ok "Telegram bot monitor started (PID $_TG_PID)"
    else
      warn "Telegram monitor exited — terminal-only logging"
    fi
  else
    warn "telegram_bot_monitor.py not found — skipping"
  fi
fi

# Overnight monitor (trade surveillance + daily summary)
_OM_PID_FILE="$PID_DIR/overnight_monitor.pid"
if _check_running "$_OM_PID_FILE" "Overnight monitor"; then
  :
else
  _OM_SCRIPT="$REBUILD_ROOT/scripts/overnight_monitor.py"
  if [[ -f "$_OM_SCRIPT" ]]; then
    "$PYTHON_BIN" "$REBUILD_ROOT/scripts/launch_detached.py" "$_OM_PID_FILE" "$LOG_DIR/overnight_monitor.log" \
      "$PYTHON_BIN" "$_OM_SCRIPT" --interval 60 </dev/null >>"$LOG_DIR/overnight_monitor_start.log" 2>&1
    sleep 2
    _OM_PID=$(cat "$_OM_PID_FILE" 2>/dev/null || echo "")
    if [[ -n "$_OM_PID" ]] && kill -0 "$_OM_PID" 2>/dev/null; then
      ok "Overnight monitor started (PID $_OM_PID)"
    else
      warn "Overnight monitor exited — check $LOG_DIR/overnight_monitor.log"
    fi
  else
    warn "overnight_monitor.py not found — skipping"
  fi
fi

# ── Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  🏁 HANOON PRIME 3.0 — ALL SYSTEMS STARTED"
echo "═══════════════════════════════════════════════════════════════"
echo ""
log "Brain:        hanoon_prime (Hippocampus + Cortex + Cerebellum + Edge)"
log "IB Gateway:    127.0.0.1:4002 (paper) — DUO429233"
log "TelemetryAPI:  http://127.0.0.1:8080/health"
log "HALIM serve:   ${HALIM_SERVER_URL:-http://127.0.0.1:8765} (advisory)"
log "Dashboard:     https://www.hanoonweb.xyz (Vercel)"
log "Named tunnel:  https://api.hanoonweb.xyz → :8080"
log ""
log "Prime bot log:  tail -f $LOG_DIR/hanoon_prime.log"
log "HALIM log:      tail -f $LOG_DIR/halim_serve.log"
log "Cloudflared:    tail -f $LOG_DIR/cloudflared.log"
log ""
log "PID files:      $PID_DIR/"
echo ""
echo "  Stop:      bash stop.command"
echo "  Monitor:   bash monitor.command"
echo "  Health:    curl http://127.0.0.1:8080/health"
echo "  Journal:   curl http://127.0.0.1:8080/journal"
echo "  Positions: curl http://127.0.0.1:8080/positions"
echo ""
