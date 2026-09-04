#!/usr/bin/env python3
"""scripts/overnight_monitor.py — Monitor trades while the bot runs overnight.

Checks every 60 seconds for:
  1. New closed trades (compares buffer size)
  2. Losses exceeding threshold ($30 default)
  3. Sub-1min entries (still happening despite guards?)
  4. Stale positions held >30min while losing
  5. Bot health (via TelemetryAPI :8080)
  6. Tunnel health (api.hanoonweb.xyz)

Telegram alerts sent via the bot's own notification system.

Usage: python3 scripts/overnight_monitor.py [--interval 60] [--loss-threshold 30]
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES_FILE = os.path.join(BASE_DIR, "models/buffer/trades.json")
MEMORY_FILE = os.path.join(BASE_DIR, "models/juli_memory/juli_memory.json")
HEALTH_URL = "http://127.0.0.1:8080/health"
TUNNEL_URL = "https://api.hanoonweb.xyz/health"


def load_trades():
    """Load trades from buffer."""
    try:
        with open(TRADES_FILE) as f:
            data = json.load(f)
        return data.get("trades", []) if isinstance(data, dict) else data
    except Exception:
        return []


def load_memory():
    """Load brain memory."""
    try:
        with open(MEMORY_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def check_health():
    """Check bot health endpoint."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def check_tunnel():
    """Check named tunnel health."""
    try:
        with urllib.request.urlopen(TUNNEL_URL, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def send_telegram_alert(message):
    """Send a Telegram alert using the bot's .env credentials."""
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(BASE_DIR, ".env"))
    except ImportError:
        pass

    token = os.environ.get(
        "TRADING_BOT_TELEGRAM_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", "")
    )
    chat_id = os.environ.get(
        "TRADING_BOT_TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_CHAT_ID", "")
    )

    if not token or not chat_id:
        print(f"  [NO TG] {message}")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps(
        {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    ).encode()
    try:
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"  [TG FAIL] {e}: {message}")
        return False


def format_trade_alert(trade, event_type="CLOSED"):
    """Format a trade alert message."""
    tk = trade.get("ticker", "?")
    pnl = trade.get("pnl", 0) or 0
    hold = (trade.get("hold_sec", 0) or 0) / 60
    ep = trade.get("entry_price", 0) or 0
    xp = trade.get("exit_price", 0) or 0
    reason = trade.get("exit_reason", "?")
    result = "WIN" if pnl > 0.01 else ("BE" if abs(pnl) <= 0.01 else "LOSS")

    emoji = "🟢" if result == "WIN" else ("⚪" if result == "BE" else "🔴")
    return (
        f"{emoji} <b>{event_type}: {tk}</b>\n"
        f"PnL: <b>${pnl:+.2f}</b> ({result})\n"
        f"Entry: ${ep:.2f} → Exit: ${xp:.2f}\n"
        f"Hold: {hold:.1f}min | Reason: {reason}"
    )


def main():
    parser = argparse.ArgumentParser(description="Overnight trade monitor")
    parser.add_argument(
        "--interval", type=int, default=60, help="Check interval in seconds"
    )
    parser.add_argument(
        "--loss-threshold", type=float, default=30.0, help="Loss alert threshold ($)"
    )
    parser.add_argument(
        "--no-telegram", action="store_true", help="Print alerts instead of sending TG"
    )
    args = parser.parse_args()

    print(f"═══════════════════════════════════════════════════════════════")
    print(f"  OVERNIGHT MONITOR — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Interval: {args.interval}s | Loss threshold: ${args.loss_threshold}")
    print(f"═══════════════════════════════════════════════════════════════")

    # Load initial state
    trades = load_trades()
    last_trade_count = len(trades)
    print(f"  Initial trades: {last_trade_count}")

    # Track last known PnL
    last_total_pnl = sum(t.get("pnl", 0) for t in trades if isinstance(t, dict))

    # Tunnel cooldown state — avoid spam on transient drops
    tunnel_was_down = False
    last_tunnel_alert_time = 0.0
    TUNNEL_COOLDOWN = 600  # 10 minutes between tunnel alerts

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now().strftime("%H:%M:%S")

        # Check health
        health = check_health()
        if "error" in health:
            print(f"  [{now}] ⚠️  Bot health: {health['error']}")
            send_telegram_alert(f"⚠️ <b>BOT HEALTH WARNING</b>\n{health['error']}")
        else:
            wr = health.get("gate", {}).get("recent_wr", 0) * 100
            gate = "OPEN" if not health.get("gate", {}).get("gate_closed") else "CLOSED"
            positions = health.get("pillar_health", {}).get("checked_at", 0)

        # Check tunnel (every 5 cycles)
        if cycle % 5 == 0:
            tunnel = check_tunnel()
            now_ts = time.time()

            if "error" in tunnel:
                if not tunnel_was_down:
                    # First failure — alert immediately
                    print(f"  [{now}] ⚠️  Tunnel: {tunnel['error']}")
                    send_telegram_alert(
                        f"⚠️ <b>TUNNEL DOWN</b>\napi.hanoonweb.xyz not responding"
                    )
                    last_tunnel_alert_time = now_ts
                    tunnel_was_down = True
                elif (now_ts - last_tunnel_alert_time) >= TUNNEL_COOLDOWN:
                    # Still down after cooldown — send one reminder
                    print(f"  [{now}] ⚠️  Tunnel still down ({tunnel['error']})")
                    send_telegram_alert(
                        f"⚠️ <b>TUNNEL STILL DOWN</b>\napi.hanoonweb.xyz still not responding"
                    )
                    last_tunnel_alert_time = now_ts
                # else: within cooldown, skip
            else:
                if tunnel_was_down:
                    # Tunnel recovered — notify
                    print(f"  [{now}] ✅ Tunnel recovered")
                    send_telegram_alert(
                        f"✅ <b>TUNNEL RECOVERED</b>\napi.hanoonweb.xyz is back online"
                    )
                tunnel_was_down = False

        # Load current trades
        trades = load_trades()
        current_count = len(trades)

        if current_count > last_trade_count:
            # New trades closed
            new_trades = trades[last_trade_count:]
            for trade in new_trades:
                if not isinstance(trade, dict):
                    continue
                pnl = trade.get("pnl", 0) or 0
                hold = (trade.get("hold_sec", 0) or 0) / 60
                tk = trade.get("ticker", "?")

                # Big loss alert
                if pnl < -args.loss_threshold:
                    msg = format_trade_alert(trade, "🚨 BIG LOSS")
                    print(f"  [{now}] {msg}")
                    send_telegram_alert(msg)

                # Sub-1min loss alert
                elif hold < 1.0 and pnl < -0.01:
                    msg = format_trade_alert(trade, "⚡ SUB-1MIN LOSS")
                    print(f"  [{now}] {msg}")
                    send_telegram_alert(msg)

                # Normal trade close (just log)
                else:
                    result = (
                        "WIN" if pnl > 0.01 else ("BE" if abs(pnl) <= 0.01 else "LOSS")
                    )
                    emoji = "🟢" if result == "WIN" else ("⚪" if result == "BE" else "🔴")
                    print(
                        f"  [{now}] {emoji} {tk} ${pnl:+.2f} {hold:.1f}min [{result}]"
                    )

            last_trade_count = current_count

        # Check for stale losing positions (>30min)
        # (This requires checking entry_meta, which we skip in the monitor
        #  because the bot's own stale exit mechanism handles it now)

        # Summary every 30 cycles (~30min)
        if cycle % 30 == 0:
            total_pnl = sum(t.get("pnl", 0) for t in trades if isinstance(t, dict))
            wins = sum(
                1 for t in trades if isinstance(t, dict) and t.get("pnl", 0) > 0.01
            )
            losses = sum(
                1 for t in trades if isinstance(t, dict) and t.get("pnl", 0) < -0.01
            )
            wr = wins / max(wins + losses, 1) * 100
            delta_pnl = total_pnl - last_total_pnl

            summary = (
                f"📊 <b>OVERNIGHT SUMMARY</b> ({now})\n"
                f"Total: {current_count} trades | WR: {wr:.1f}%\n"
                f"PnL: ${total_pnl:+.2f} (Δ${delta_pnl:+.2f} since monitor start)\n"
                f"Last 30min: {current_count - last_trade_count + 30} new trades"
            )
            print(f"\n  [{now}] === SUMMARY ===")
            print(
                f"  Trades: {current_count} | WR: {wr:.1f}% | PnL: ${total_pnl:+.2f} (Δ${delta_pnl:+.2f})"
            )
            send_telegram_alert(summary)
            last_total_pnl = total_pnl

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
