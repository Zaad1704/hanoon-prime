#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  stop.command — HANOON PRIME 3.0 — Stop EVERYTHING
#
#  CRITICAL: launchd agents are booted out FIRST (before killing processes)
#  so KeepAlive doesn't respawn them. Order:
#    0. Boot out ALL launchd auto-restart agents (bot, csv-refresh,
#       halim, tunnel, monitors) — prevents respawn
#    1. Web dashboard / monitors (health, csv refresh, overnight)
#    2. HANOON bot (hanoon_rebuild/main.py) — graceful SIGINT, then SIGTERM/SIGKILL
#    3. HALIM serve (Qwen3.5-4B MLX MoE) — SIGKILL after SIGTERM grace
#    4. IB Gateway watchdog, cloudflared, all orphans — force kill
#    5. Cleanup PID files + tunnel URL
#
#  After this script: no HANOON processes should be running.
#  The only thing left untouched is the REAL IB Gateway (if running).
#
#  Usage: double-click, or  bash stop.command
#  Restart: start.command
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail

# ── Resolve directories ─────────────────────────────────────────────────
BOT_ROOT="$(cd "$(dirname "$0")" && pwd)"
REBUILD_ROOT="$HOME/Downloads/hanoon_rebuild"
PID_DIR="$BOT_ROOT/runtime/pids"
LOG_DIR="$BOT_ROOT/logs"

mkdir -p "$PID_DIR" "$LOG_DIR"

# ── Helpers ─────────────────────────────────────────────────────────────
log()  { echo -e "\033[1;36m[$(date '+%H:%M:%S')]\033[0m $*"; }
ok()   { echo -e "\033[32m[$(date '+%H:%M:%S')]\033[0m ✅ $*"; }
warn() { echo -e "\033[33m[$(date '+%H:%M:%S')]\033[0m ⚠️  $*"; }
err()  { echo -e "\033[31m[$(date '+%H:%M:%S')]\033[0m ❌ $*"; }

# ── Graceful stop helper ━────────────────────────────────────────────────
# Sends SIGTERM, waits up to $grace seconds, then SIGKILL.
_graceful_stop() {
  local pids="$1" label="$2" grace="${3:-5}"
  if [[ -z "$pids" ]]; then
    return
  fi
  local alive_pids=""
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      alive_pids="$alive_pids $pid"
    fi
  done
  if [[ -z "$alive_pids" ]]; then
    return
  fi
  log "Stopping $label (SIGTERM)… PIDs:$alive_pids"
  # shellcheck disable=SC2086
  kill $alive_pids 2>/dev/null || true
  local waited=0
  while [[ $waited -lt $grace ]]; do
    local still_alive=""
    for pid in $alive_pids; do
      if kill -0 "$pid" 2>/dev/null; then
        still_alive="$still_alive $pid"
      fi
    done
    if [[ -z "$still_alive" ]]; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  warn "$label still running after ${grace}s — sending SIGKILL…"
  # shellcheck disable=SC2086
  kill -9 $alive_pids 2>/dev/null || true
  sleep 1
}

_stop_pid_file() {
  local file="$1" label="$2" grace="${3:-5}"
  if [[ -f "$file" ]]; then
    local pid
    pid=$(tr -d '[:space:]' <"$file" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      _graceful_stop "$pid" "$label" "$grace"
    fi
    rm -f "$file"
  fi
}

_stop_pgrep() {
  local pattern="$1" label="$2" grace="${3:-5}"
  local pids
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    _graceful_stop "$pids" "$label" "$grace"
  fi
}

# ── Banner ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  🛑 HANOON PRIME 3.0 — FULL SYSTEM STOP"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── 0. CRITICAL: Boot out ALL launchd agents FIRST ━━━━━━━━━━━━━━━━━━━
log "Disabling ALL launchd auto-restart agents..."
_UID_GUI="gui/$(id -u)"
for _lbl in com.hanoon.bot-autorestart com.hanoon.csv-refresh \
            com.hanoon.halim com.hanoon.tunnel com.hanoon.tunnel-sync \
            com.hanoon.bridge-keepalive com.hanoon.ws-keepalive \
            com.hanoon.health-monitor; do
  launchctl bootout "$_UID_GUI/$_lbl" 2>/dev/null \
    || launchctl remove "$_lbl" 2>/dev/null || true
done
sleep 2
ok "All launchd agents booted out (auto-restart disabled)"

# ── 1. Monitors ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Stopping monitors..."
_stop_pid_file "$PID_DIR/health_monitor.pid" "Health monitor" 3
_stop_pid_file "$PID_DIR/csv_refresh.pid" "CSV refresh" 3
_stop_pid_file "$PID_DIR/overnight_monitor.pid" "Overnight monitor" 3
_stop_pid_file "$PID_DIR/telegram_monitor.pid" "Telegram monitor" 2
_stop_pgrep "health_monitor.py" "Health monitor (orphan)" 2
_stop_pgrep "fetch_us_ticker_csvs.py" "CSV refresh (orphan)" 2
_stop_pgrep "overnight_monitor.py" "Overnight monitor (orphan)" 2
_stop_pgrep "telegram_bot_monitor\|send_telegram_alert" "Telegram monitor (orphan)" 2
ok "Monitors stopped"

# ── 2. HANOON Prime Bot ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Stopping HANOON Prime bot (graceful shutdown)..."
_BOT_PID_FILE="$PID_DIR/hanoon_prime.pid"
_BOT_PID=""

if [[ -f "$_BOT_PID_FILE" ]]; then
  _BOT_PID=$(tr -d '[:space:]' < "$_BOT_PID_FILE" 2>/dev/null || true)
fi

if [[ -n "$_BOT_PID" ]] && kill -0 "$_BOT_PID" 2>/dev/null; then
  log "  Sending SIGINT to Prime bot (PID $_BOT_PID)..."
  kill -INT "$_BOT_PID" 2>/dev/null || true

  waited=0
  while [[ $waited -lt 30 ]]; do
    if ! kill -0 "$_BOT_PID" 2>/dev/null; then
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done

  if kill -0 "$_BOT_PID" 2>/dev/null; then
    warn "Bot didn't exit gracefully — sending SIGTERM → SIGKILL"
    _graceful_stop "$_BOT_PID" "HANOON Prime bot" 5
  else
    ok "Prime bot exited gracefully (IB streaming cancelled, journal flushed)"
  fi
else
  warn "Prime bot PID not found — searching by process name"
  _stop_pgrep "hanoon_prime\.cli" "HANOON Prime bot" 5
fi

rm -f "$_BOT_PID_FILE"
ok "HANOON Prime bot stopped"

# ── 2b. Learning Vault Release ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Releasing learning vault..."
if [[ -f "$REBUILD_ROOT/scripts/vault_release.sh" ]]; then
  bash "$REBUILD_ROOT/scripts/vault_release.sh" 2>&1 | sed 's/^/  [vault] /'
  ok "Learning vault released"
else
  warn "vault_release.sh not found — snapshot skipped"
fi

# ── 3. HALIM Serve ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Stopping HALIM serve (M. A. Halim)..."
_HALIM_PID_FILE="$PID_DIR/halim_serve.pid"

# Prefer prime's own stop script (proper MLX/Metal cleanup)
if [[ -f "$BOT_ROOT/scripts/halim_stop.sh" ]]; then
  bash "$BOT_ROOT/scripts/halim_stop.sh" >> "$LOG_DIR/halim_stop.log" 2>&1 || true
  ok "HALIM serve stopped"
elif [[ -f "$REBUILD_ROOT/scripts/halim_stop_rebuild.sh" ]]; then
  bash "$REBUILD_ROOT/scripts/halim_stop_rebuild.sh" >> "$LOG_DIR/halim_stop.log" 2>&1 || true
  ok "HALIM serve stopped (legacy rebuild script)"
fi
_stop_pid_file "$_HALIM_PID_FILE" "HALIM serve" 5
_stop_pgrep "halim/halim/serve.py\|halim_serve\|ensure_halim" "HALIM serve (orphan)" 5
ok "HALIM serve stopped"

# ── 3b. IB Gateway Watchdog ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Stopping IB Gateway watchdog..."
_stop_pid_file "$PID_DIR/ib_gateway_watchdog.pid" "IB Gateway watchdog" 3
_stop_pgrep "ib-gateway-watchdog" "IB Gateway watchdog" 3
_stop_pgrep "IBC.jar" "IBC auto-login" 3
ok "IB Gateway watchdog stopped"

# ── 3c. Cloudflare Tunnel ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Stopping Cloudflare tunnel..."
_stop_pid_file "$PID_DIR/cloudflared.pid" "Cloudflared tunnel" 3
_stop_pgrep "cloudflared tunnel" "Cloudflared" 3
ok "Cloudflared tunnel stopped"

# ── 4. Remaining orphans ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Killing remaining orphans..."
_stop_pgrep "next-server\|next dev\|node.*dev" "Web dashboard (orphan)" 2
_stop_pgrep "freebuff" "freebuff (orphan)" 2
ok "All orphans killed"

# ── 5. Cleanup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "Cleaning up stale PID files..."
rm -f "$PID_DIR"/*.pid
rm -f "$BOT_ROOT/runtime/tunnel_url.txt"
ok "Cleanup complete"

# ── Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ HANOON PRIME 3.0 — ALL SYSTEMS STOPPED"
echo "═══════════════════════════════════════════════════════════════"
echo ""
log "Launchd agents booted out — nothing will auto-restart"
log "Bot TelemetryAPI stopped (port 8080 released)"
log "HALIM stopped (MLX/Metal buffers released)"
log "Monitoring stopped"
log ""
log "Logs preserved at: $LOG_DIR/"
echo ""
echo "  Restart:  bash start.command"
echo "  Or:       cd $BOT_ROOT && bash start.command"
echo ""
