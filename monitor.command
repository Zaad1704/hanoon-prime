#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  monitor.command — HANOON PRIME 3.0 — Status + Live Log Viewer
#
#  Shows bot status, then tails the running log with color coding.
#  READ-ONLY — never starts/stops the bot.
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/logs/hanoon_prime.log"

echo "═══════════════════════════════════════════════════════════════"
echo "  HANOON PRIME 3.0 — STATUS"
echo "═══════════════════════════════════════════════════════════════"

# Check bot process
BOT_PID=$(pgrep -f "hanoon_prime.cli" 2>/dev/null | head -1)
if [ -n "$BOT_PID" ]; then
    ELAPSED=$(ps -o etime= -p "$BOT_PID" 2>/dev/null | xargs)
    echo "[$(date +%H:%M:%S)] ✅ Prime bot  ✓ PID $BOT_PID  (uptime: $ELAPSED)"
else
    echo "[$(date +%H:%M:%S)] ❌ Prime bot  ✗ Not running"
fi

# Check telemetry
HEALTH=$(curl -s --max-time 2 http://127.0.0.1:8080/health 2>/dev/null)
if [ -n "$HEALTH" ]; then
    POSITIONS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('positions',[])))" 2>/dev/null)
    STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null)
    echo "[$(date +%H:%M:%S)] ✅ Telemetry    ✓ http://127.0.0.1:8080  ($STATUS, $POSITIONS positions)"
else
    echo "[$(date +%H:%M:%S)] ❌ Telemetry    ✗ Not responding"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Tail: $LOG  (Ctrl+C to exit)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Tail with color coding
tail -f "$LOG" 2>/dev/null | while IFS= read -r line; do
    if echo "$line" | grep -qE "BRACKET|ENTRY|EXECUTE|ADOPT"; then
        echo -e "\033[32m$line\033[0m"
    elif echo "$line" | grep -qE "EXIT|TRAIL STOP|TRAIL TARGET|Position closed"; then
        echo -e "\033[31m$line\033[0m"
    elif echo "$line" | grep -qE "HEARTBEAT|open="; then
        echo -e "\033[35m$line\033[0m"
    elif echo "$line" | grep -qE "THINK|SKIP|verdict"; then
        echo -e "\033[36m$line\033[0m"
    elif echo "$line" | grep -qE "ERROR|FATAL|Traceback|CRITICAL"; then
        echo -e "\033[1;31m$line\033[0m"
    elif echo "$line" | grep -qE "WARNING|warn|SAFETY NET"; then
        echo -e "\033[33m$line\033[0m"
    elif echo "$line" | grep -qE "connected|ready|Subscribed|streams active"; then
        echo -e "\033[32m$line\033[0m"
    elif echo "$line" | grep -qE "commissionReport|orderStatus|execDetails"; then
        echo -e "\033[90m$line\033[0m"
    else
        echo "$line"
    fi
done
