"""hanoon_prime._telegram_chat — read-only Telegram query interface.

Rebuild ops/telegram_chat.py port: long-polling listener answering
natural-language queries about JULI's state. READ-ONLY by design — the
chat answers questions from an injected state provider; it can never
place orders or change config (the webapp/API owns mutations). Auth =
the configured chat_id only. Stdlib-only, daemon thread, near-zero
idle CPU.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Any, Callable, Optional

from ._telegram import _get_chat_id, _get_token

log = logging.getLogger(__name__)

_POLL_INTERVAL: float = 3.0
_LONG_POLL_TIMEOUT: int = 2
_CHAT_MAX_PER_MIN: int = 20
_chat_bucket: dict[str, float] = {"count": 0.0, "window": 0.0}


class TelegramChat:
    """Long-polling listener: natural-language queries about JULI's state."""

    def __init__(self, state_provider: Callable[[], dict[str, Any]] | None = None):
        self._provider = state_provider
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._offset = 0

    def start(self) -> None:
        """Start the chat daemon (no-op when Telegram is unconfigured)."""
        if not _get_token() or not _get_chat_id():
            log.info("Chat disabled: Telegram not configured")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="telegram-chat"
        )
        self._thread.start()
        log.info("Telegram chat started")

    def stop(self) -> None:
        """Stop the chat listener thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while self._running:
            try:
                for update in self._poll():
                    self._handle(update)
            except Exception as exc:
                log.debug("Chat poll failed: %s", exc)
                time.sleep(_POLL_INTERVAL)

    def _poll(self) -> list[dict[str, Any]]:
        """Long-poll Telegram getUpdates (bounded, silent on failure)."""
        token = _get_token() or ""
        url = (
            f"https://api.telegram.org/bot{token}/getUpdates"
            f"?timeout={_LONG_POLL_TIMEOUT}&offset={self._offset}"
        )
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=_LONG_POLL_TIMEOUT + 10) as resp:
            raw = json.loads(resp.read(262144))
        result = raw.get("result", []) if isinstance(raw, dict) else []
        updates: list[dict[str, Any]] = (
            [u for u in result if isinstance(u, dict)]
            if isinstance(result, list)
            else []
        )
        if updates:
            self._offset = int(updates[-1].get("update_id", 0)) + 1
        return updates

    def _handle(self, update: dict[str, Any]) -> None:
        """Respond to one message from the AUTHORIZED chat only."""
        msg = update.get("message") or {}  # array-safe: dict-typed
        chat = msg.get("chat") if isinstance(msg, dict) else {}
        chat_id = str(chat.get("id", "")) if isinstance(chat, dict) else ""
        if not chat_id or chat_id != str(_get_chat_id() or ""):
            return
        text = str(msg.get("text", "")).strip()
        if not text:
            return
        now = time.time()
        if now - _chat_bucket["window"] >= 60.0:
            _chat_bucket["count"], _chat_bucket["window"] = 0.0, now
        _chat_bucket["count"] += 1
        if _chat_bucket["count"] > _CHAT_MAX_PER_MIN:
            return
        self._reply(self._answer(text))

    def _reply(self, text: str) -> None:
        """Send a chat reply (bypasses notification rate bucket)."""
        token, chat_id = _get_token(), _get_chat_id()
        if not token or not chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            body = json.dumps({"chat_id": chat_id, "text": text[:4096]}).encode()
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=15)
        except Exception as exc:
            log.debug("Chat reply failed: %s", exc)

    def _answer(self, text: str) -> str:
        """Route a query to a read-only answer built from brain state."""
        q = text.lower().lstrip("/")
        state: dict[str, Any] = self._provider() if self._provider else {}
        if q.startswith("status") or q == "s":
            return self._fmt_status(state)
        if q.startswith("positions"):
            return self._fmt_positions(state)
        if q.startswith("horizons"):
            return "Horizons: " + ", ".join(state.get("horizons", ["scalp"]))
        if q.startswith("help") or q == "start":
            return (
                "JULI Prime chat (read-only):\n"
                "/status — brain + P&L\n"
                "/positions — open trades\n"
                "/horizons — active horizons"
            )
        return "Unknown query. Try /help"

    @staticmethod
    def _fmt_status(state: dict[str, Any]) -> str:
        lines = [
            f"🧠 JULI | regime={state.get('regime', '?')}"
            f" threshold={state.get('threshold', 0):.2f}",
            f"Decisions: {state.get('decisions', 0)}"
            f" | Daily P&L: ${state.get('daily_pnl', 0.0):+.2f}",
            f"Open: {state.get('open_positions', 0)}"
            f" | Wins: {state.get('wins', 0)} Losses: {state.get('losses', 0)}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _fmt_positions(state: dict[str, Any]) -> str:
        positions = state.get("positions", {})
        if not positions:
            return "No open positions."
        lines = []
        for t, info in list(positions.items())[:10]:
            d = "LONG" if info.get("direction", 1) > 0 else "SHORT"
            lines.append(
                f"{t} {d} {info.get('shares', 0)}@${info.get('entry', 0):.2f}"
                f" [{info.get('horizon', 'scalp')}]"
            )
        return "\n".join(lines)


__all__ = ["TelegramChat"]
