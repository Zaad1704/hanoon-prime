"""hanoon_prime.telemetry — lightweight HTTP health endpoint for cloudflared.

Serves /health, /journal, /positions on :8080 so the Cloudflare
named tunnel (api.hanoonweb.xyz → :8080) has a real endpoint.
Runs in a background thread alongside the trading bot.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .immune import DAILY_LOSS_LIMIT, TELEMETRY_PORT
from .memory import Journal

log = __import__("logging").getLogger(__name__)


class _TelemetryHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks and journal read-out."""

    # Class-level references set by TelemetryAPI.start()
    bot: Any = None
    journal_path: Path | None = None

    def log_message(self, fmt: str, *args: Any) -> None:
        """Suppress default stderr logging — use structured logging instead."""

    def do_GET(self) -> None:
        """Handle GET requests for health/journal/positions/safety-net."""
        routes = {
            "/health": self._health,
            "/journal": self._journal,
            "/positions": self._positions,
            "/safety-net": self._safety_net_status,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._respond(404, {"error": "not found", "path": self.path})
        else:
            self._respond(200, handler())

    def do_POST(self) -> None:
        """Handle POST requests — toggle safety net on/off."""
        if self.path != "/safety-net":
            self._respond(404, {"error": "not found", "path": self.path})
            return
        action = self._read_body().get("action", "")
        if action in ("enable", "disable"):
            enabled = action == "enable"
            self._set_safety_net(enabled)
            self._respond(200, {"safety_net_enabled": enabled})
        else:
            self._respond(
                400,
                {"error": 'expected JSON {"action": "enable"|"disable"}'},
            )

    def _read_body(self) -> dict[str, Any]:
        """Parse JSON request body (empty dict if invalid)."""
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            raw = json.loads(self.rfile.read(length))
            if isinstance(raw, dict):
                return dict(raw)
            return {}
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("Invalid JSON in POST body: %s", e)
            return {}

    def _respond(self, code: int, payload: dict[str, Any]) -> None:
        """Write JSON response with the given status code."""
        body = json.dumps(payload).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as e:
            log.debug("client disconnected mid-response: %s", e)

    @staticmethod
    def _ib_positions(ib: Any) -> list[Any]:
        """Read positions directly from IB (source of truth)."""
        if not ib:
            return []
        try:
            return list(ib.positions())
        except Exception:
            return []

    def _health(self) -> dict[str, Any]:
        """Return bot connection and position status."""
        bot = self.bot
        if bot is None:
            return {"status": "starting", "bot": False}
        ib = getattr(bot, "ib", None)
        connected = ib.isConnected() if ib else False
        journal = getattr(bot, "journal", None)
        return {
            "status": "ok" if connected else "disconnected",
            "connected": connected,
            "tickers": [s for s in bot.streamer.ticker_subs] if bot else [],
            "positions": [p.contract.symbol for p in self._ib_positions(ib)],
            "journal_entries": journal.count() if journal else 0,
            "safety_net_enabled": self._safety_net_status().get("enabled", False),
        }

    def _safety_net_status(self) -> dict[str, Any]:
        """Return current safety net toggle status and limits."""
        bot = self.bot
        hp = getattr(bot, "hippocampus", None) if bot else None
        enabled = getattr(hp, "safety_enabled", False) if hp else False
        daily_pnl = getattr(hp, "_daily_pnl", 0.0) if hp else 0.0
        return {
            "enabled": enabled,
            "daily_pnl": daily_pnl,
            "limit": DAILY_LOSS_LIMIT,
            "consecutive_losses": (getattr(hp, "_consecutive_losses", 0) if hp else 0),
        }

    def _set_safety_net(self, enabled: bool) -> None:
        """Toggle safety net enforcement on the running bot."""
        bot = self.bot
        if bot is None:
            return
        hp = getattr(bot, "hippocampus", None)
        if hp is not None:
            hp.safety_enabled = enabled
            log.info("Safety net %s via webapp", "ENABLED" if enabled else "DISABLED")

    def _journal(self) -> dict[str, Any]:
        """Return the last 20 journal entries."""
        if not self.journal_path or not self.journal_path.exists():
            return {"entries": []}
        entries = Journal(self.journal_path).tail(20)[::-1]
        return {"entries": entries}

    def _positions(self) -> dict[str, Any]:
        """Return open positions from IB (source of truth)."""
        bot = self.bot
        if bot is None:
            return {"positions": []}
        ib = getattr(bot, "ib", None)
        positions = [
            {
                "ticker": p.contract.symbol,
                "entry_price": p.avgCost,
                "shares": abs(p.position),
                "direction": 1 if p.position > 0 else -1,
                "pnl_pct": 0,
            }
            for p in self._ib_positions(ib)
        ]
        return {"positions": positions}


class TelemetryAPI:
    """Background HTTP server on --health port for cloudflared tunnel."""

    def __init__(self, bot: Any, journal_path: Path) -> None:
        self._bot = bot
        self._journal_path = journal_path
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        _TelemetryHandler.bot = bot
        _TelemetryHandler.journal_path = journal_path

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        self._server = HTTPServer(("127.0.0.1", TELEMETRY_PORT), _TelemetryHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        log.info("TelemetryAPI live on http://127.0.0.1:%s", TELEMETRY_PORT)

    def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            log.info("TelemetryAPI stopped")


__all__ = ["TelemetryAPI"]
