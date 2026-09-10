"""hanoon_prime.telemetry — real-time HTTP + SSE API for Juli webapp.

Transport model (v2):
  * An always-on refresher thread rebuilds a full telemetry snapshot every
    SNAPSHOT_INTERVAL seconds — 24/7, whether or not any browser is watching.
  * ``GET /snapshot`` returns that cached snapshot instantly (one round-trip
    instead of the old 11 separate polls).
  * ``GET /stream`` is a Server-Sent Events endpoint: every new snapshot is
    pushed to all connected browsers within SNAPSHOT_INTERVAL. True real-time,
    works through Cloudflare tunnels with no extra configuration.
  * All legacy per-topic routes are kept and served from the cache when fresh
    (zero cost) or rebuilt directly when the refresher thread is down.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, cast

from .config import TRADING_CONFIG
from .immune import DAILY_LOSS_LIMIT, TELEMETRY_PORT
from .memory import Journal

log = __import__("logging").getLogger(__name__)

# Shared flag: main cycle checks this and flattens when non-empty
_FLATTEN_REQUESTED: list[int] = []  # [position_count] when pending

# Snapshot cadence. 1s = real-time feel without hammering IB; the refresher
# thread runs at this rate forever, independent of any browser.
SNAPSHOT_INTERVAL = 1.0

ROUTES_GET = {
    "/health": "_health",
    "/journal": "_journal",
    "/positions": "_positions",
    "/safety-net": "_safety_net_status",
    "/brain": "_brain_state",
    "/trades": "_recent_trades",
    "/system2": "_system2_state",
    "/pipeline": "_pipeline_state",
    "/risk": "_risk_state",
    "/config": "_config",
    "/halim": "_halim_state",
    "/verdicts": "_verdicts",
    "/session": "_session",
    "/account": "_account",
    "/resources": "_resources",
    "/inspection": "_inspection",
}
POST_ROUTES = {"/safety-net", "/config"}

# Routes computed outside the 1s snapshot (they are served on demand with a
# time-to-live cache). /account is cheap (a dict read) so it stays uncached.
EXTRA_TTL: dict[str, float] = {"/resources": 2.0, "/inspection": 30.0}

# Runtime daemon roles → pidfile name under <repo>/runtime/pids/.
_PROCESS_ROLES: tuple[tuple[str, str], ...] = (
    ("bot", "hanoon_prime"),
    ("halim", "halim_serve"),
    ("watchdog", "ib_gateway_watchdog"),
    ("overnight_monitor", "overnight_monitor"),
    ("production_monitor", "production_monitor"),
)

# Route → handler method for POST mutations (also flipped via /config UI).
POST_HANDLERS = {
    "/safety-net": "_handle_safety_net",
    "/config": "_handle_config",
    "/flatten": "_handle_flatten",
}

# Route → key inside the snapshot payload
_ROUTE_KEY = {
    "/health": "health",
    "/journal": "journal",
    "/positions": "positions",
    "/safety-net": "safety_net",
    "/brain": "brain",
    "/trades": "trades",
    "/system2": "system2",
    "/pipeline": "pipeline",
    "/risk": "risk",
    "/config": "config",
    "/halim": "halim",
    "/verdicts": "verdicts",
    "/session": "session",
}

_MISSING = object()


class _H(BaseHTTPRequestHandler):
    bot: Any = None
    journal_path: Path | None = None
    # Shared cache owned by TelemetryAPI; set before the server starts.
    cache: dict[str, Any] = {}
    cache_lock: threading.Lock | None = None
    builder: Callable[[], dict[str, Any]] | None = None
    sse_registry: "SseRegistry | None" = None
    # TTL cache for the on-demand routes (see EXTRA_TTL); set by TelemetryAPI.
    extra_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    extra_lock: threading.Lock | None = None
    # Overridable in tests so /inspection can be served without a subprocess.
    inspection_builder: Callable[[], dict[str, Any]] | None = None
    # Called after a successful POST so the cache reflects the mutation
    # immediately (no 1s staleness on user-initiated changes).
    on_mutation: Callable[[], None] | None = None

    protocol_version = "HTTP/1.1"  # keep-alive; required for smooth SSE

    def log_message(self, *_a: Any) -> None:
        """Suppress default stderr logging."""

    # ── Routing ─────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        """Serve telemetry routes from the cached snapshot."""
        path = self.path.split("?", 1)[0]
        if path == "/snapshot":
            self._r(200, self._current_snapshot())
            return
        if path == "/stream":
            self._handle_stream()
            return
        if path not in ROUTES_GET:
            self._r(404, {"error": "not found", "path": path})
            return
        self._r(200, self._serve_route(path))

    def _serve_route(self, path: str) -> dict[str, Any]:
        """Resolve a route payload: cached snapshot or on-demand builder."""
        if path in EXTRA_TTL:
            return self._extra_route(path)
        key = _ROUTE_KEY.get(path)
        if key is not None:
            payload = self._current_snapshot().get(key, _MISSING)
            if payload is not _MISSING:
                return cast(dict[str, Any], payload)
        return cast(dict[str, Any], getattr(self, ROUTES_GET[path])())

    def _extra_route(self, path: str) -> dict[str, Any]:
        """Serve a TTL-cached out-of-snapshot route (e.g. /inspection)."""
        ttl = EXTRA_TTL[path]
        lock = self.extra_lock or threading.Lock()
        with lock:
            hit = self.extra_cache.get(path)
            if hit is not None and time.time() - hit[0] < ttl:
                return hit[1]
        payload = cast(
            dict[str, Any], getattr(self, ROUTES_GET[path])()
        )
        with lock:
            self.extra_cache[path] = (time.time(), payload)
        return payload

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests, then refresh the snapshot cache at once."""
        handler = POST_HANDLERS.get(self.path)
        if handler is None:
            self._r(404, {"error": "not found", "path": self.path})
            return
        getattr(self, handler)()
        self._refresh_after_mutation()

    def _refresh_after_mutation(self) -> None:
        """Rebuild and broadcast the snapshot after a successful mutation."""
        if self.on_mutation is None:
            return
        try:
            self.on_mutation()
        except Exception as exc:
            log.debug("post-mutation refresh failed: %s", exc)

    # ── Snapshot / stream plumbing ──────────────────────────────────────

    def _current_snapshot(self) -> dict[str, Any]:
        try:
            with self.cache_lock or threading.Lock():
                cached = self.cache.get("data")
                if not isinstance(cached, dict):
                    return {}
                return cached
        except Exception:
            return {}

    def _handle_stream(self) -> None:
        """Server-Sent Events: push a fresh snapshot every interval."""
        reg = self.sse_registry
        if reg is None:
            self._r(503, {"error": "stream unavailable"})
            return
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            # Immediate first frame so the UI goes live instantly.
            first = self._current_snapshot()
            if first:
                self.wfile.write(
                    f"event: snapshot\ndata: {json.dumps(first, default=str)}\n\n".encode()
                )
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        reg.add(self)
        try:
            while True:
                # Sleep in short slices so shutdown() is honoured quickly.
                if getattr(self.server, "_shutdown_request", False):
                    break
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError, OSError):
            log.debug("sse client disconnected")
        finally:
            reg.discard(self)
            try:
                self.wfile.close()
            except Exception as exc:
                log.debug("sse client wfile close: %s", exc)

    # ── POST handlers (unchanged behaviour) ─────────────────────────────

    def _handle_safety_net(self) -> None:
        """Toggle safety net on/off, or resume from a halt (brain commands)."""
        action = self._body().get("action", "")
        if action in ("enable", "disable"):
            en = action == "enable"
            brain = self._brain()
            if brain is not None:
                brain.set_safety_enabled(en)
            else:
                hp = getattr(self.bot, "hippocampus", None) if self.bot else None
                if hp is not None:
                    hp.safety_enabled = en
            log.info("Safety net %s via webapp", "ENABLED" if en else "DISABLED")
            self._r(200, {"safety_net_enabled": en, "halting": False})
        elif action == "resume":
            brain = self._brain()
            if brain is not None:
                brain.resume()
                policy = self._policy_state()
                self._r(200, {"halted": policy.get("halted", False)})
                return
            if self.bot and hasattr(self.bot, "_halted"):
                self.bot._halted = False
                log.info("Halt CLEARED via webapp")
            self._r(200, {"halted": getattr(self.bot, "_halted", False)})
        else:
            self._r(400, {"error": 'expected {"action": "enable"|"disable"|"resume"}'})

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
        if "horizons" in body:
            # Webapp activates trading horizons (rebuild parity: the ladder
            # is fully implemented; scalp-only is the safe default).
            from .brain.horizons import get_horizon_manager

            enabled = get_horizon_manager().set_enabled(body["horizons"])
            TRADING_CONFIG.horizons = set(enabled)
            log.info("Horizons -> %s", enabled)
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

    def _brain(self) -> Any:
        """NeuromorphicBrain instance (decision state owner) or None."""
        juli = getattr(self.bot, "juli", None) if self.bot else None
        return getattr(juli, "brain", None) if juli else None

    def _policy_state(self) -> dict[str, Any]:
        """policy_state published by the slow cortex (or {} when absent)."""
        brain = self._brain()
        if brain is None:
            return {}
        policy = brain.state.get("policy_state")
        return policy if isinstance(policy, dict) else {}

    def _ib_positions(self) -> list[Any]:
        ib = self._ib()
        return list(ib.positions()) if ib else []

    # ── Payload builders (unchanged, now called by the refresher thread) ─

    def _health(self) -> dict[str, Any]:
        bot, ib = self.bot, self._ib()
        if bot is None:
            return {"status": "starting", "bot": False}
        con = ib.isConnected() if ib else False
        j = getattr(bot, "journal", None)
        hp = getattr(bot, "hippocampus", None)
        policy = self._policy_state()
        from .monitor.sleep_manager import SleepManager

        _st = SleepManager().effective_state(TRADING_CONFIG)
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
            "safety_net_enabled": policy.get(
                "enabled",
                getattr(hp, "safety_enabled", False) if hp else False,
            ),
            "halted": policy.get("halted", getattr(bot, "_halted", False)),
            "authorized": policy.get("authorized", True),
            "pause_reason": policy.get("pause_reason", ""),
            "last_beat": round(float(getattr(bot, "_last_beat", 0.0)), 1),
            "session": _st.session,
            "session_active": bool(_st.active),
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

    def _pipeline_state(self) -> dict[str, Any]:
        """Continuous pipeline health (monitor daemon view)."""
        mon = getattr(self.bot, "monitor", None) if self.bot else None
        if mon is None:
            return {"healthy": False, "error": "monitor not wired"}
        snap: dict[str, Any] = mon.snapshot()
        return snap

    def _risk_state(self) -> dict[str, Any]:
        """Portfolio risk snapshot (policy_state, observer-safe subset)."""
        policy = self._policy_state()
        keep = (
            "equity",
            "equity_synced",
            "drawdown",
            "exposure",
            "stress_mode",
            "risk_scalar",
            "position_count",
            "max_positions",
            "daily_pnl",
            "consecutive_losses",
        )
        return {k: policy.get(k) for k in keep if k in policy}

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

    def _account(self) -> dict[str, Any]:
        """Raw IB account feed published by ib_cycle (equity, summary, holdings)."""
        brain = self._brain()
        state = getattr(brain, "state", None) if brain else None
        feed = state.snapshot().get("account_feed", {}) if state is not None else {}
        return dict(feed) if isinstance(feed, dict) else {}

    def _resources(self) -> dict[str, Any]:
        """Per-process CPU/RAM plus host loadavg (cached, ps-based, no deps)."""
        pids_dir = (
            self.journal_path.resolve().parent.parent / "runtime" / "pids"
            if self.journal_path
            else None
        )
        bot_pid = os.getpid()
        procs: list[dict[str, Any]] = []
        seen: set[int] = set()
        for role, fname in _PROCESS_ROLES:
            pid = self._pidfile_pid(pids_dir, fname)
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            usage = self._process_usage(pid)
            procs.append(
                {"role": role, **usage} if usage else {"role": role, "alive": False}
            )
        if bot_pid not in seen:
            usage = self._process_usage(bot_pid)
            procs.append(
                {"role": "bot", **usage} if usage else {"role": "bot", "alive": False}
            )
        try:
            load1, load5, load15 = os.getloadavg()
            loadavg = [round(load1, 2), round(load5, 2), round(load15, 2)]
        except (OSError, AttributeError):
            loadavg = []
        return {"procs": procs, "loadavg": loadavg, "ts": time.time()}

    @staticmethod
    def _pidfile_pid(pids_dir: Path | None, fname: str) -> int | None:
        if not pids_dir:
            return None
        p = pids_dir / f"{fname}.pid"
        if not p.exists():
            return None
        try:
            pid = int(p.read_text().strip())
        except (OSError, ValueError):
            return None
        return pid if pid > 0 else None

    @staticmethod
    def _process_usage(pid: int) -> dict[str, Any] | None:
        """ps-based %CPU + RSS(kB) for one pid. None when not readable."""
        try:
            out = subprocess.run(
                ["ps", "-o", "pid=,pcpu=,rss=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        parts = out.stdout.split()
        if len(parts) < 3:
            return None
        try:
            return {
                "pid": int(parts[0]),
                "cpu_pct": float(parts[1]),
                "rss_kb": int(float(parts[2])),
            }
        except ValueError:
            return None

    def _inspection(self) -> dict[str, Any]:
        """Inside Man manifest, TTL-cached. Runs in an isolated subprocess so
        heavy log scans never stall the bot's telemetry thread."""
        # type() lookup keeps the overridable class attr an unbound callable.
        fn = type(self).inspection_builder or self._run_inspection
        try:
            return fn()
        except Exception as exc:
            return {"status": "UNVERIFIABLE", "error": str(exc)[:200]}

    def _run_inspection(self) -> dict[str, Any]:
        exe = sys.executable or "python3"
        cmd = [exe, "-m", "hanoon_prime.inspection", "manifest", "--json"]
        try:
            out = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=28,
                cwd=self._repo_root(),
            )
        except subprocess.TimeoutExpired:
            return {"status": "UNVERIFIABLE", "error": "inspection timed out"}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"status": "UNVERIFIABLE", "error": f"inspection failed: {exc}"}
        if out.returncode not in (0, 2, 3, 4):
            return {
                "status": "UNVERIFIABLE",
                "error": (out.stderr or "").strip()[:200] or f"exit {out.returncode}",
            }
        try:
            payload: dict[str, Any] = json.loads(out.stdout)
        except json.JSONDecodeError as exc:
            return {"status": "UNVERIFIABLE", "error": f"bad manifest json: {exc}"}
        payload["_runner"] = "subprocess"
        return payload

    def _repo_root(self) -> str:
        if self.journal_path:
            for p in self.journal_path.resolve().parents:
                if (p / "src" / "hanoon_prime").is_dir():
                    return str(p)
        return str(Path.cwd())

    def _safety_net_status(self) -> dict[str, Any]:
        policy = self._policy_state()
        hp = getattr(self.bot, "hippocampus", None) if self.bot else None
        return {
            "enabled": policy.get(
                "enabled", getattr(hp, "safety_enabled", False) if hp else False
            ),
            "halted": policy.get("halted", getattr(self.bot, "_halted", False)),
            "authorized": policy.get("authorized", True),
            "pause_reason": policy.get("pause_reason", ""),
            "daily_pnl": policy.get("daily_pnl", getattr(hp, "_daily_pnl", 0.0)),
            "limit": DAILY_LOSS_LIMIT,
            "consecutive_losses": policy.get(
                "consecutive_losses", getattr(hp, "_consecutive_losses", 0)
            ),
        }

    def _handle_flatten(self) -> None:
        """Request flatten — set flag for main cycle to execute."""
        bot = self.bot
        if bot is None:
            self._r(503, {"error": "bot not running"})
            return
        ib = self._ib()
        if ib is None or not ib.isConnected():
            self._r(503, {"error": "IB not connected"})
            return
        pos_count = len(self._ib_positions())
        # Set flag — main cycle thread will execute flatten
        _FLATTEN_REQUESTED.clear()
        _FLATTEN_REQUESTED.append(pos_count)
        log.warning("FLATTEN requested: %d positions", pos_count)
        self._r(
            200,
            {
                "flattened": 0,
                "pending": True,
                "action": "flatten_all",
                "positions": pos_count,
            },
        )

    def _config(self) -> dict[str, Any]:
        """Return current trading config + live EOD status."""
        from .monitor.sleep_manager import SleepManager

        sm = SleepManager()
        d = TRADING_CONFIG.to_dict()
        d["minutes_to_close"] = round(sm.minutes_to_close(), 1)
        d["eod_window_active"] = sm.is_eod_window(TRADING_CONFIG.eod_flatten_minutes)
        return d

    def _session(self) -> dict[str, Any]:
        """Current session-gate state (source of truth for HALIM + ops)."""
        from .monitor.sleep_manager import SleepManager

        st = SleepManager().effective_state(TRADING_CONFIG)
        return {
            "session": st.session,
            "active": bool(st.active),
            "enabled": TRADING_CONFIG.to_dict().get("sessions", {}),
            "reason": st.reason,
            "ts": time.time(),
        }

    def _halim_state(self) -> dict[str, Any]:
        """HALIM state: regime, modifier, postmortem, recommendations."""
        juli = getattr(self.bot, "juli", None) if self.bot else None
        brain = getattr(juli, "brain", None) if juli else None
        state = getattr(brain, "state", None) if brain else None
        if state is None:
            return {}
        s = state.snapshot()
        return {
            "regime_label": s.get("regime_label", "unknown"),
            "regime_multiplier": s.get("regime_multiplier", 1.0),
            "halim_modifier": s.get("halim_modifier", 0.0),
            "halim_last_insight": s.get("halim_last_insight", {}),
            "halim_recommendations": s.get("halim_recommendations", []),
        }

    def _verdicts(self) -> dict[str, Any]:
        """Recent brain Verdicts (observer view of decision state)."""
        juli = getattr(self.bot, "juli", None) if self.bot else None
        rv = getattr(juli, "_recent_verdicts", None) if juli else None
        recent: list[dict[str, Any]] = []
        if rv:
            recent = [v.to_dict() for v in list(rv)[-50:]]
        return {"count": len(recent), "verdicts": recent}

    def _journal(self) -> dict[str, Any]:
        if not self.journal_path or not self.journal_path.exists():
            return {"entries": []}
        return {"entries": Journal(self.journal_path).tail(20)[::-1]}


class SseRegistry:
    """Tracks connected SSE clients and pushes new snapshots to them."""

    def __init__(self) -> None:
        self._clients: list[_H] = []
        self._lock = threading.Lock()

    def add(self, client: _H) -> None:
        """Register a connected SSE client."""
        with self._lock:
            self._clients.append(client)

    def discard(self, client: _H) -> None:
        """Remove a client that disconnected or crashed."""
        with self._lock:
            try:
                self._clients.remove(client)
            except ValueError as exc:
                log.debug("sse discard missing client: %s", exc)

    def broadcast(self, snapshot: dict[str, Any]) -> int:
        """Push a snapshot to every connected client. Returns delivered count."""
        if not self._clients:
            return 0
        frame = f"event: snapshot\ndata: {json.dumps(snapshot, default=str)}\n\n".encode()
        dead: list[_H] = []
        with self._lock:
            clients = list(self._clients)
        for c in clients:
            try:
                c.wfile.write(frame)
                c.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead.append(c)
        for c in dead:
            self.discard(c)
        return len(clients) - len(dead)

    @property
    def count(self) -> int:
        """Number of currently connected SSE clients."""
        with self._lock:
            return len(self._clients)


def build_snapshot(handler: _H) -> dict[str, Any]:
    """Aggregate every telemetry topic into one payload."""
    return {
        "health": handler._health(),
        "positions": handler._positions(),
        "safety_net": handler._safety_net_status(),
        "brain": handler._brain_state(),
        "system2": handler._system2_state(),
        "pipeline": handler._pipeline_state(),
        "risk": handler._risk_state(),
        "config": handler._config(),
        "halim": handler._halim_state(),
        "verdicts": handler._verdicts(),
        "trades": handler._recent_trades(),
        "journal": handler._journal(),
        "session": handler._session(),
        "meta": {
            "built_ts": time.time(),
            "interval": SNAPSHOT_INTERVAL,
        },
    }


class TelemetryAPI:
    """Background HTTP + SSE server for Juli webapp.

    A refresher thread rebuilds the full snapshot every SNAPSHOT_INTERVAL
    seconds, forever — connected browsers or not. SSE clients get each new
    snapshot pushed instantly; GET /snapshot returns the cached copy; legacy
    per-topic routes are served from the cache.
    """

    def __init__(self, bot: Any, journal_path: Path) -> None:
        self._bot, self._jp = bot, journal_path
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._refresh_thread: threading.Thread | None = None
        self._cache: dict[str, Any] = {"data": {}, "ts": 0.0}
        self._cache_lock = threading.Lock()
        self._registry = SseRegistry()
        self._stop = threading.Event()
        # Legacy contract: binding at init lets bare HTTPServer(_H)
        # fixtures (and any external harness) work without start().
        _H.bot = bot
        _H.journal_path = journal_path
        _H.extra_cache = {}
        _H.extra_lock = threading.Lock()
        _H.inspection_builder = None

    # ── wiring ──────────────────────────────────────────────────────────

    def _make_handler(self) -> type[_H]:
        # Bind instance state onto the class once (handler classes are
        # instantiated per-request by http.server).
        _H.cache = self._cache
        _H.cache_lock = self._cache_lock
        _H.sse_registry = self._registry
        _H.on_mutation = self._build_and_cache
        return _H

    def _build_and_cache(self) -> None:
        h = _H.__new__(_H)
        h.bot = self._bot
        h.journal_path = self._jp
        try:
            snap = build_snapshot(h)
        except Exception as exc:  # never let the refresher die
            log.warning("snapshot build failed: %s", exc)
            return
        with self._cache_lock:
            self._cache["data"] = snap
            self._cache["ts"] = time.time()
        self._registry.broadcast(snap)

    def _refresh_loop(self) -> None:
        log.info("Telemetry snapshot refresher started (%.1fs cadence)", SNAPSHOT_INTERVAL)
        while not self._stop.is_set():
            t0 = time.time()
            self._build_and_cache()
            # Sleep the remainder of the interval in small slices.
            elapsed = time.time() - t0
            remaining = max(0.05, SNAPSHOT_INTERVAL - elapsed)
            if self._stop.wait(remaining):
                break

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self, port: int | None = None) -> None:
        """Start HTTP server + snapshot refresher in background threads.

        ``port`` overrides TELEMETRY_PORT (used by tests with port=0 for an
        ephemeral bind); production callers keep the default.
        """
        handler = self._make_handler()
        bind_port = TELEMETRY_PORT if port is None else port
        self._server = ThreadingHTTPServer(("127.0.0.1", bind_port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="telemetry-http"
        )
        self._thread.start()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True, name="telemetry-refresher"
        )
        self._refresh_thread.start()
        log.info(
            "TelemetryAPI live on http://127.0.0.1:%s (SSE /stream, /snapshot)",
            TELEMETRY_PORT,
        )

    def stop(self) -> None:
        """Shut down HTTP server + refresher."""
        self._stop.set()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        log.info("TelemetryAPI stopped")

    @property
    def stream_clients(self) -> int:
        """Number of currently connected SSE stream clients."""
        return self._registry.count


__all__ = ["TelemetryAPI", "SNAPSHOT_INTERVAL"]
