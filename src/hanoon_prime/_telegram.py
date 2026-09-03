"""hanoon_prime._telegram — Telegram notifications for JULI.

Sends trade entries, exits, safety halts, and errors to Telegram.
Uses stdlib urllib — no external dependencies.
Rate-limited (max 10/min), 429-aware, chunked at 4096 chars.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Optional

log = logging.getLogger(__name__)

_MAX_PER_MIN: int = 10
_MIN_INTERVAL: float = 1.0
_429_COOLDOWN: float = 30.0
_bucket: dict[str, float] = {"count": 0.0, "window": 0.0}
_lock = threading.Lock()
_last_send: float = 0.0
_cooldown_until: float = 0.0


def _get_token() -> Optional[str]:
    """Read Telegram bot token from env."""
    for var in ("TRADING_BOT_TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN"):
        val = os.getenv(var)
        if val:
            return val
    return None


def _get_chat_id() -> Optional[str]:
    """Read Telegram chat ID from env."""
    for var in ("TRADING_BOT_TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID"):
        val = os.getenv(var)
        if val:
            return val
    return None


def _rate_ok() -> bool:
    """Check global token bucket. Returns True if send allowed."""
    global _bucket
    with _lock:
        now = time.time()
        if _bucket["count"] >= _MAX_PER_MIN:
            if now - _bucket["window"] < 60.0:
                return False
            _bucket = {"count": 0.0, "window": now}
        if _bucket["count"] == 0.0:
            _bucket["window"] = now
        _bucket["count"] += 1.0
        return True


def _send_chunk(token: str, chat_id: str, chunk: str) -> bool:
    """Send one chunk via Telegram API. Returns True on success."""
    global _last_send, _cooldown_until
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        body = json.dumps({"chat_id": chat_id, "text": chunk}).encode()
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15):
            _last_send = time.time()
        return True
    except Exception as exc:
        log.warning("Telegram send failed: %s", exc)
        if "429" in str(exc):
            _cooldown_until = time.time() + _429_COOLDOWN
        return False


def send(message: str) -> bool:
    """Send a message via Telegram. Returns True on success."""
    token, chat_id = _get_token(), _get_chat_id()
    if not token or not chat_id:
        log.debug("Telegram not configured")
        return False
    if not _rate_ok():
        log.debug("Telegram rate-limited, skipping")
        return False
    now = time.time()
    wait = _MIN_INTERVAL - (now - _last_send)
    if wait > 0:
        time.sleep(wait)
    if now < _cooldown_until:
        log.debug("Telegram 429 cooldown, skipping")
        return False
    for i in range(0, len(message), 4096):
        if not _send_chunk(token, chat_id, message[i : i + 4096]):
            return False
    return True


def trade_opened(
    ticker: str,
    side: str,
    qty: int,
    price: float,
    stop: float,
    target: float,
) -> None:
    """Notify trade entry."""
    msg = (
        f"🟢 TRADE OPENED\n"
        f"{side} {qty} {ticker} @ ${price:.2f}\n"
        f"Stop: ${stop:.2f} | Target: ${target:.2f}"
    )
    send(msg)
    log.info(msg)


def trade_closed(
    ticker: str,
    side: str,
    pnl: float,
    reason: str = "",
) -> None:
    """Notify trade exit with P&L."""
    result = "WIN" if pnl > 0.01 else ("LOSS" if pnl < -0.01 else "BREAKEVEN")
    emoji = "✅" if result == "WIN" else ("🔴" if result == "LOSS" else "➖")
    msg = f"👑 JULI {emoji} {result} {ticker}\n{side} | P&L: ${pnl:+.4f}"
    if reason:
        msg += f"\nReason: {reason}"
    send(msg)
    log.info(msg)


def safety_halt(reason: str) -> None:
    """Notify safety net halt."""
    send(f"🛑 TRADING HALTED\n{reason}")


def error_notify(context: str, detail: str) -> None:
    """Notify an error condition."""
    send(f"❗ ERROR in {context}\n{detail}")


def startup(tickers: list[str]) -> None:
    """Notify bot startup."""
    send(f"🚀 JULI Prime started\nTickers: {', '.join(tickers)}")


def shutdown(reason: str = "") -> None:
    """Notify bot shutdown."""
    send(f"🔴 JULI Prime stopped\n{reason}")


__all__ = [
    "send",
    "trade_opened",
    "trade_closed",
    "safety_halt",
    "error_notify",
    "startup",
    "shutdown",
]
