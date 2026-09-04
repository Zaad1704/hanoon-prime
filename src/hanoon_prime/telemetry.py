"""hanoon_prime.telemetry — HTTP API for Juli webapp."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .config import TRADING_CONFIG
from .immune import DAILY_LOSS_LIMIT, TELEMETRY_PORT
from .memory import Journal

log = __import__("logging").getLogger(__name__)

ROUTES_GET = {
    "/health": "_health",
    "/journal": "_journal",
    "/positions": "_positions",
    "/safety-net": "_safety_net_status",
    "/brain": "_brain_state",
    "/trades": "_recent_trades",
    "/system2": "_system2_state",
    "/config": "_config",
}
POST_ROUTES = {"/safety-net", "/config"}


class _H(BaseHTTPRequestHandler):
    bot: Any = None
    journal_path: Path | None = None

    def log_message(self, *_a: Any) -> None:
        """Suppress default stderr logging."""

    def do_GET(self) -> None:
        """Route GET requests."""
        name = ROUTES_GET.get(self.path)
        if name is None:
            self._r(404, {"error": "not found", "path": self.path})
        else:
            self._r(200, getattr(self, name)())

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/safety-net":
            self._handle_safety_net()
        elif self.path == "/config":
            self._handle_config()
        elif self.path == "/flatten":
            self._handle_flatten()
        else:
            self._r(404, {"error": "not found", "path": self.path})

    def _handle_safety_net(self) -> None:
        """Toggle safety net on/off."""
        action = self._body().get("action", "")
        if action in ("enable", "disable"):
            en = action == "enable"
            hp = getattr(self.bot, "hippocampus", None) if self.bot else None
            if hp is not None:
                hp.safety_enabled = en
                log.info("Safety net %s via webapp", "ENABLED" if en else "DISABLED")
            self._r(200, {"safety_net_enabled": en})
        else:
            self._r(400, {"error": 'expected {"action": "enable"|"disable"}'})

    def _handle_config(self) -> None:
        """GET: return config. POST: update config fields."""
        body = self._body()
        if not body:
            self._r(200, TRADING_CONFIG.to_dict())
            return
        if "sessions" in body:
            for k, v in body["sessions"].items():
                attr = f"session_{k}"
                if hasattr(TRADING_CONFIG, attr) and isinstance(v, bool):
                    setattr(TRADING_CONFIG, attr, v)
                    log.info("Session %s -> %s", k, "ON" if v else "OFF")
        if "direction_mode" in body:
            dm = body["direction_mode"]
            if dm in ("both", "long_only", "short_only"):
                TRADING_CONFIG.direction_mode = dm
                log.info("Direction mode -> %s", dm)
        if "eod_flatten_enabled" in body:
            TRADING_CONFIG.eod_flatten_enabled = bool(body["eod_flatten_enabled"])
            log.info(
                "EOD flatten -> %s",
                "ON" if TRADING_CONFIG.eod_flatten_enabled else "OFF",
            )
        if "eod_flatten_minutes" in body:
            TRADING_CONFIG.eod_flatten_minutes = float(body["eod_flatten_minutes"])
            log.info("EOD window -> %.1f min", TRADING_CONFIG.eod_flatten_minutes)
        self._r(200, TRADING_CONFIG.to_dict())

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", 0))
        if n == 0:
            return {}
        try:
            raw = json.loads(self.rfile.read(n))
            return dict(raw) if isinstance(raw, dict) else {}
        except Exception as e:
            log.warning("Bad POST body: %s", e)
            return {}

    def _r(self, code: int, d: dict[str, Any]) -> None:
        body = json.dumps(d, default=str).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as exc:
            log.debug("Client disconnected: %s", exc.__class__.__name__)

    def _ib(self) -> Any:
        return getattr(self.bot, "ib", None) if self.bot else None

    def _ib_positions(self) -> list[Any]:
        ib = self._ib()
        return list(ib.positions()) if ib else []

    def _health(self) -> dict[str, Any]:
        bot, ib = self.bot, self._ib()
        if bot is None:
            return {"status": "starting", "bot": False}
        con = ib.isConnected() if ib else False
        j = getattr(bot, "journal", None)
        hp = getattr(bot, "hippocampus", None)
        ts_keys = (
            list(bot.streamer.ticker_subs.keys())
            if bot and hasattr(bot.streamer.ticker_subs, "keys")
            else []
        )
        return {
            "status": "ok" if con else "disconnected",
            "connected": con,
            "tickers": ts_keys,
            "positions": [p.contract.symbol for p in self._ib_positions()],
            "position_count": len(self._ib_positions()),
            "journal_entries": j.count() if j else 0,
            "safety_net_enabled": getattr(hp, "safety_enabled", False) if hp else False,
            "uptime": time.time(),
        }

    def _positions(self) -> dict[str, Any]:
        out = []
        for p in self._ib_positions():
            sym = p.contract.symbol
            mp = round(float(getattr(p, "marketPrice", p.avgCost)), 2)
            ep = round(p.avgCost, 2)
            up = round(float(getattr(p, "unrealizedPnl", 0)), 2)
            pp = round(((mp - ep) / ep) * 100, 2) if ep > 0 else 0.0
            out.append(
                {
                    "ticker": sym,
                    "entry_price": ep,
                    "shares": abs(int(p.position)),
                    "direction": "LONG" if p.position > 0 else "SHORT",
                    "market_price": mp,
                    "unrealized_pnl": up,
                    "pnl_pct": pp,
                }
            )
        return {
            "positions": out,
            "total_pnl": round(sum(p["unrealized_pnl"] for p in out), 2),
            "count": len(out),
        }

    def _recent_trades(self) -> dict[str, Any]:
        if not self.journal_path or not self.journal_path.exists():
            return {"trades": []}
        es = Journal(self.journal_path).tail(50)[::-1]
        return {
            "trades": [e for e in es if e.get("event") in ("position_closed", "exit")][
                :20
            ]
        }

    def _brain_state(self) -> dict[str, Any]:
        juli = getattr(self.bot, "juli", None) if self.bot else None
        brain = getattr(juli, "brain", None) if juli else None
        if brain is None:
            return {}
        s = brain.snapshot()
        m = s.get("memory", {})
        return {
            "threshold": round(s.get("threshold", 0.58), 4),
            "decision_count": s.get("decision_count", 0),
            "episodic_size": s.get("episodic_size", 0),
            "weights": m.get("weights", {}),
            "pred_error": m.get("pred_error", 0.0),
            "brain_state": s.get("brain_state", {}),
        }

    def _system2_state(self) -> dict[str, Any]:
        juli = getattr(self.bot, "juli", None) if self.bot else None
        brain = getattr(juli, "brain", None) if juli else None
        state = getattr(brain, "state", None) if brain else None
        if state is None:
            return {}
        s = state.snapshot()
        return {
            "regime_multiplier": s.get("regime_multiplier", 1.0),
            "regime_label": s.get("regime_label", "unknown"),
            "halim_modifier": s.get("halim_modifier", 0.0),
            "thinker_modifier": s.get("thinker_modifier", 0.0),
            "thinker_confidence": s.get("thinker_confidence_mod", 0.0),
            "refractory": s.get("refractory_until", 0) > time.time(),
        }

    def _safety_net_status(self) -> dict[str, Any]:
        hp = getattr(self.bot, "hippocampus", None) if self.bot else None
        if hp is None:
            return {
                "enabled": False,
                "daily_pnl": 0.0,
                "limit": DAILY_LOSS_LIMIT,
                "consecutive_losses": 0,
            }
        return {
            "enabled": getattr(hp, "safety_enabled", False),
            "daily_pnl": round(getattr(hp, "_daily_pnl", 0.0), 2),
            "limit": DAILY_LOSS_LIMIT,
            "consecutive_losses": getattr(hp, "_consecutive_losses", 0),
        }

    def _handle_flatten(self) -> None:
        """Flatten all positions via limit orders (post-market safe)."""
        bot = self.bot
        if bot is None:
            self._r(503, {"error": "bot not running"})
            return
        ib = self._ib()
        if ib is None or not ib.isConnected():
            self._r(503, {"error": "IB not connected"})
            return
        closed = 0
        for pos in self._ib_positions():
            sym = pos.contract.symbol
            qty = abs(int(pos.position))
            mp = float(getattr(pos, "marketPrice", pos.avgCost))
            action = "SELL" if pos.position > 0 else "BUY"
            try:
                from ib_insync import LimitOrder

                limit_price = round(mp, 2)
                order = LimitOrder(action, qty, limit_price, tif="DAY")
                ib.placeOrder(pos.contract, order)
                log.info("FLATTEN %s %s %d @ %.2f", action, sym, qty, limit_price)
                closed += 1
            except Exception as e:
                log.warning("FLATTEN failed %s: %s", sym, e)
        # Also cancel all pending orders
        try:
            getattr(ib, "cancelAllOrders", ib.reqGlobalCancel)()
        except Exception:
            pass
        self._r(200, {"flattened": closed, "action": "flatten_all"})

    def _config(self) -> dict[str, Any]:
        """Return current trading config + live EOD status."""
        from .monitor.sleep_manager import SleepManager

        sm = SleepManager()
        d = TRADING_CONFIG.to_dict()
        d["minutes_to_close"] = round(sm.minutes_to_close(), 1)
        d["eod_window_active"] = sm.is_eod_window(TRADING_CONFIG.eod_flatten_minutes)
        return d

    def _journal(self) -> dict[str, Any]:
        if not self.journal_path or not self.journal_path.exists():
            return {"entries": []}
        return {"entries": Journal(self.journal_path).tail(20)[::-1]}


class TelemetryAPI:
    """Background HTTP server for Juli webapp."""

    def __init__(self, bot: Any, journal_path: Path) -> None:
        self._bot, self._jp = bot, journal_path
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        _H.bot = bot
        _H.journal_path = journal_path

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        self._server = HTTPServer(("127.0.0.1", TELEMETRY_PORT), _H)
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
