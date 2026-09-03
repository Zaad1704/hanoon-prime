#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  monitor.command — HANOON PRIME 3.0 — Live Log Viewer
#
#  One double-click tails the running JULI bot log, colorized.
#  READ-ONLY — never starts/stops the bot. Just follows the running bot.
#
#  Equivalent of: tail -f logs/hanoon_prime.log
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/logs/hanoon_prime.log"

echo "  📋 LIVE LOG — HANOON PRIME 3.0 (logs/hanoon_prime.log)"
echo "  ────────────────────────────────────────────────────────"
echo ""

tail -f "$LOG" | while IFS= read -r line; do
    # Color code based on content
    if echo "$line" | grep -qE "BRACKET|ENTRY|Entry payload approved|EXECUTE"; then
        echo -e "\033[32m$line\033[0m"          # green for entries
    elif echo "$line" | grep -qE "EXIT|TRAIL STOP|TRAIL TARGET|Position closed"; then
        echo -e "\033[31m$line\033[0m"          # red for exits
    elif echo "$line" | grep -qE "SPIKE|⚡"; then
        echo -e "\033[33m$line\033[0m"          # yellow for spikes
    elif echo "$line" | grep -qE "JULI|🧠|Cortex|Cerebellum|Hippocampus"; then
        echo -e "\033[36m$line\033[0m"          # cyan for brain
    elif echo "$line" | grep -qE "ERROR|FATAL|Traceback|CRITICAL"; then
        echo -e "\033[1;31m$line\033[0m"        # bold red for errors
    elif echo "$line" | grep -qE "WARNING|warn|SAFETY NET"; then
        echo -e "\033[33m$line\033[0m"          # yellow for warnings
    elif echo "$line" | grep -qE "✅|✓|connected|ready|healthy|Subscribed"; then
        echo -e "\033[32m$line\033[0m"          # green for success
    elif echo "$line" | grep -qE "EXEC|commissionReport|orderStatus"; then
        echo -e "\033[90m$line\033[0m"          # dim for ib_insync noise
    else
        echo "$line"
    fi
done
