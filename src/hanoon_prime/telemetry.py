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
import hmac
import secrets
import subprocess
import sys
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, cast
from urllib.parse import urlparse

from ._ib_marks import mark_positions
from .inspection.notify import manifest_notify
from .config import TRADING_CONFIG
from .immune import DAILY_LOSS_LIMIT, TELEMETRY_AUTH_ENABLED, TELEMETRY_PORT
from .memory import Journal

log = __import__("logging").getLogger(__name__)

# Shared flag: main cycle checks this and flattens when non-empty
# Carries order type ("market" or "limit") + optional limit_price
# so the flatten executor can place the right order kind.
_FLATTEN_REQUESTED: dict[str, Any] = {}  # {"positions": N, "order_type": "market", "limit_price": None}

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
    "/logs": "_log_tail",
    "/ib": "_ib_raw",
    "/decisions": "_decisions",
    "/trade-quality": "_trade_quality",
    "/auth": "_auth",
}
POST_ROUTES = {"/safety-net", "/config"}

# Routes computed outside the 1s snapshot (they are served on demand with a
# time-to-live cache). /account is cheap (a dict read) so it stays uncached.
EXTRA_TTL: dict[str, float] = {"/resources": 2.0, "/inspection": 30.0, "/logs": 1.0}

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

# /ib is served on demand with a short TTL: it walks live IB objects
# (tickers, accounts, orders) — cheap, but not needed at 1s cadence.

# Browser origins allowed to read telemetry cross-origin. The tunnel URL
# (runtime/tunnel_url.txt) and any TELEMETRY_CORS_ORIGIN entries are added
# automatically, so a Vercel-hosted webapp works with zero manual config.
DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "https://www.hanoonweb.xyz",
    "https://hanoonweb.xyz",
    "https://hanoon-dash.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)
EXTRA_TTL["/ib"] = 1.0

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
    "/logs": "logs",
}

_MISSING = object()


def psutil_boot_time() -> float:
    """Boot time (epoch seconds) without psutil — sysctl/proc, 0 on miss."""
    try:
        if sys.platform == "darwin":
            return _sysctl_boot_time()
        return _proc_boot_time()
    except Exception as exc:
        log.debug("psutil_boot_time failed: %s", exc)
    return 0.0


def _sysctl_boot_time() -> float:
    """macOS boot time via ``sysctl kern.boottime``."""
    out = subprocess.run(
        ["sysctl", "-n", "kern.boottime"],
        capture_output=True,
        text=True,
        timeout=2,
    )
    for part in out.stdout.replace(",", " ").split():
        if part.startswith("sec="):
            return float(part.split("=")[1])
    return 0.0


def _proc_boot_time() -> float:
    """Linux boot time from ``/proc/stat`` btime field."""
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("btime"):
                return float(line.split()[1])
    return 0.0


def _tail_log(log_file: Path) -> list[str]:
    """Read the last 64KB of a log as up-to-200 tail lines."""
    with open(log_file, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        read_size = min(size, 64 * 1024)
        fh.seek(max(0, size - read_size))
        data = fh.read().decode("utf-8", errors="replace")
    return data.strip().splitlines()[-200:]


def _log_tail_lines(candidates: list[Path]) -> list[str]:
    """First readable candidate log tail, else empty."""
    for log_file in candidates:
        try:
            return _tail_log(log_file)
        except (OSError, ValueError):
            continue
    return []


def _proc_count() -> dict[str, Any]:
    """Process count via ``ps -A`` (macOS/Linux fallback)."""
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid="], capture_output=True, text=True, timeout=3
        )
        if out.returncode == 0:
            return {"proc_count": len(out.stdout.split())}
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("proc count failed: %s", exc)
    return {"proc_count": None}


def _uptime_s() -> int | None:
    """System uptime in seconds (epoch boot time → now)."""
    boot_s = psutil_boot_time()
    if boot_s <= 0:
        return None
    return int(time.time() - boot_s)


def _mem_stats() -> dict[str, Any]:
    """Host memory usage via ``sysctl`` (macOS) or ``/proc/meminfo``."""
    try:
        if sys.platform == "darwin":
            return _macos_mem_stats()
        return _linux_mem_stats()
    except Exception as exc:
        log.debug("memory stats failed: %s", exc)
    return {"mem_total_mb": None, "mem_used_mb": None, "mem_pct": None}


def _macos_mem_stats() -> dict[str, Any]:
    """macOS memory via ``sysctl hw.memsize`` + ``vm_stat``."""
    out = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        capture_output=True,
        text=True,
        timeout=2,
    )
    total_b = int(out.stdout.strip())
    total_mb = round(total_b / 1048576, 1)
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2)
    page = 16384
    used_pages = 0
    for line in vm.stdout.splitlines():
        if line.startswith(("Pages active", "Pages wired down", "Pages compressed")):
            used_pages += int(line.split(":")[1].strip().rstrip("."))
    used_mb = round(used_pages * page / 1048576, 1)
    return {
        "mem_total_mb": total_mb,
        "mem_used_mb": used_mb,
        "mem_pct": round(100.0 * used_mb / total_mb, 1) if total_mb else None,
    }


def _linux_mem_stats() -> dict[str, Any]:
    """Linux memory from ``/proc/meminfo`` (kB fields)."""
    mem: dict[str, int] = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, rest = line.partition(":")
            mem[k.strip()] = int(rest.strip().split()[0])
    total_kb = mem.get("MemTotal", 0)
    avail_kb = mem.get("MemAvailable", 0)
    total_mb = round(total_kb / 1024, 1)
    used_mb = round((total_kb - avail_kb) / 1024, 1)
    return {
        "mem_total_mb": total_mb,
        "mem_used_mb": used_mb,
        "mem_pct": round(100.0 * used_mb / total_mb, 1) if total_mb else None,
    }


def _ib_order_meta(order: Any, contract: Any) -> dict[str, Any]:
    """Static order/contract fields (symbol, qty, prices)."""
    return {
        "symbol": getattr(contract, "symbol", "") or "",
        "sec_type": getattr(contract, "secType", "") or "",
        "exchange": getattr(contract, "exchange", "") or "",
        "order_id": getattr(order, "orderId", None),
        "perm_id": getattr(order, "permId", None),
        "action": getattr(order, "action", "") or "",
        "total_qty": _H._num(getattr(order, "totalQuantity", None)),
        "order_type": getattr(order, "orderType", "") or "",
        "lmt_price": _H._num(getattr(order, "lmtPrice", None)),
        "aux_price": _H._num(getattr(order, "auxPrice", None)),
    }


def _format_exec_time(ex: Any) -> str:
    """Safely format an execution's time as ISO or string fallback."""
    raw = getattr(ex, "time", None)
    iso_fn = getattr(raw, "isoformat", None)
    if callable(iso_fn):
        return str(iso_fn())
    return str(raw or "")


def _epoch_seconds(raw: Any) -> int | None:
    """Convert a ticker/event timestamp (datetime, epoch, or None) to int
    epoch seconds. IB Ticker.time is a datetime — int(datetime) raises
    TypeError, which previously took down the whole /ib route."""
    if raw is None:
        return None
    ts_fn = getattr(raw, "timestamp", None)
    if callable(ts_fn):
        try:
            return int(ts_fn())
        except Exception:
            return None
    n = _H._num(raw)
    return int(n) if n is not None else None


def _is_async(v: Any) -> bool:
    """Detect unawaited coroutines from ib_insync async-mode calls."""
    return hasattr(v, "close") and hasattr(v, "__await__")


def _ib_fill_rows(trade: Any) -> list[dict[str, Any]]:
    """Fill executions for one IB Trade."""
    rows: list[dict[str, Any]] = []
    for f in getattr(trade, "fills", []) or []:
        ex = getattr(f, "execution", None)
        rows.append(
            {
                "exec_id": getattr(ex, "execId", ""),
                "time": _format_exec_time(ex),
                "side": getattr(ex, "side", "") or "",
                "shares": _H._num(getattr(ex, "shares", None)),
                "price": _H._num(getattr(ex, "price", None)),
                "commission": _H._num(
                    getattr(getattr(f, "commissionReport", None), "commission", None)
                ),
            }
        )
    return rows


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
    # Bearer auth for mutations + CORS posture (wired by TelemetryAPI.start()).
    auth_enabled: bool = False
    telemetry_token: str = ""
    cors_origin: str | None = None

    protocol_version = "HTTP/1.1"  # keep-alive; required for smooth SSE

    def log_message(self, *_a: Any) -> None:
        """Suppress default stderr logging."""

    def handle(self) -> None:
        """Guard against client disconnects during request-line reading.

        BaseHTTPRequestHandler.handle() calls self.rfile.readline() to read
        the raw request line; a client that closes the connection mid-handshake
        raises ConnectionResetError / BrokenPipeError, which propagates to
        ThreadingHTTPServer.process_request_thread → handle_error and produces
        a noisy traceback. Catching it here keeps the server silent on the
        common client-flush pattern (e.g. health checkers, browser preflights).
        """
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, ConnectionError):
            log.debug("client disconnected during request line read")

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
        payload = cast(dict[str, Any], getattr(self, ROUTES_GET[path])())
        with lock:
            self.extra_cache[path] = (time.time(), payload)
        return payload

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight (browser asks before sending the bearer)."""
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests, then refresh the snapshot cache at once."""
        if not self._authorized():
            self._unauthorized()
            return
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
            self._send_cors_headers()
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
        """Toggle safety net on/off, resume a halt, or re-arm the kill switch."""
        action = self._body().get("action", "")
        if action in ("enable", "disable"):
            self._toggle_safety_net(action == "enable")
        elif action == "kill_release":
            self._release_kill()
        elif action == "resume":
            self._resume_halt()
        else:
            self._r(400, {"error": 'expected {"action": "enable"|"disable"|"resume"|"kill_release"}'})

    def _toggle_safety_net(self, en: bool) -> None:
        """Enable/disable the safety net via brain or hippocampus fallback."""
        brain = self._brain()
        if brain is not None:
            brain.set_safety_enabled(en)
        else:
            hp = getattr(self.bot, "hippocampus", None) if self.bot else None
            if hp is not None:
                hp.safety_enabled = en
        log.info("Safety net %s via webapp", "ENABLED" if en else "DISABLED")
        self._r(200, {"safety_net_enabled": en, "halting": False})

    def _release_kill(self) -> None:
        """Latch is deliberate: only an explicit kill_release re-arms it."""
        brain = self._brain()
        if brain is None:
            self._r(400, {"error": "brain not ready"})
            return
        brain.release_kill()
        policy = self._policy_state()
        self._r(
            200,
            {
                "halted": policy.get("halted", False),
                "latched": self._latched_state(),
            },
        )

    def _resume_halt(self) -> None:
        """Clear a halt via brain resume (kill latch stays latched)."""
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

    def _r(
        self,
        code: int,
        d: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(d, default=str).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as exc:
            log.debug("Client disconnected: %s", exc.__class__.__name__)

    def _send_cors_headers(self) -> None:
        """Reflect an allowed origin — never the '*' wildcard.

        Accepts a single origin (str) or an allow-list (list). With a list,
        the request's own Origin is reflected back when it is allowed, so
        multiple first-party frontends (webapp + tunnel + localhost dev)
        work simultaneously. Requests without an Origin header (curl, SSE
        tooling, health checks) get the full list for informational use.
        """
        allowed = self._cors_origin()
        if allowed is None:
            return
        origin = self.headers.get("Origin")
        if isinstance(allowed, str):
            if origin is None or origin == allowed:
                self.send_header("Access-Control-Allow-Origin", allowed)
                self.send_header("Vary", "Origin")
            return
        if isinstance(allowed, list):
            if origin is None:
                self.send_header("Access-Control-Allow-Origin", ", ".join(allowed))
            elif origin in allowed:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

    def _tunnel_origin(self) -> str | None:
        """Origin derived from the cloudflared tunnel URL file, if readable."""
        try:
            tun = Path(self._repo_root()) / "runtime" / "tunnel_url.txt"
            if tun.is_file():
                p = urlparse(tun.read_text().strip())
                if p.scheme and p.netloc:
                    return f"{p.scheme}://{p.netloc}"
        except Exception:
            log.debug("cors origin probe failed", exc_info=True)
        return None

    def _cors_origin(self) -> str | list[str] | None:
        """Allowed browser origin(s).

        Precedence: handler attr (tests / explicit pin) > merged allow-list
        of DEFAULT_ALLOWED_ORIGINS + auto-derived cloudflared tunnel origin
        + any TELEMETRY_CORS_ORIGIN entries (single or comma-separated).
        Merging (not overriding) means the webapp works with zero manual
        config while the tunnel URL can change freely. None => deny.
        """
        if self.cors_origin:
            return self.cors_origin
        allowed: list[str] = list(DEFAULT_ALLOWED_ORIGINS)
        tun = self._tunnel_origin()
        if tun and tun not in allowed:
            allowed.append(tun)
        env = os.environ.get("TELEMETRY_CORS_ORIGIN")
        if env:
            for o in env.split(","):
                o = o.strip()
                if o and o not in allowed:
                    allowed.append(o)
        return allowed or None

    def _authorized(self) -> bool:
        """Bearer gate for mutations. Open when auth disabled or token unset."""
        if not self.auth_enabled or not self.telemetry_token:
            return True
        auth = self.headers.get("Authorization", "")
        return hmac.compare_digest(auth, f"Bearer {self.telemetry_token}")

    def _unauthorized(self) -> None:
        """401 for unauthenticated mutations."""
        self._r(401, {"error": "unauthorized"})

    def _auth(self) -> None:
        """GET /auth — hand the bearer token to first-party browser origins.

        Design: GETs are already open (the tunnel is the perimeter), so the
        token on its own grants nothing extra. Gating this route on the CORS
        allow-list keeps random websites from being able to read it via
        cross-origin fetch, while the first-party webapp auto-provisions
        with zero manual setup. Auth disabled => auth_disabled, so the UI
        knows POSTs need no header.
        """
        if not self.auth_enabled or not self.telemetry_token:
            self._r(200, {"auth": False, "auth_disabled": True})
            return
        allowed = self._cors_origin()
        origin = self.headers.get("Origin")
        ok = isinstance(allowed, str) and (origin is None or origin == allowed)
        ok = ok or (isinstance(allowed, list) and (origin is None or origin in allowed))
        if not ok:
            log.warning("/auth denied for origin %r", origin)
            self._r(403, {"error": "origin not allowed"})
            return
        self._r(
            200,
            {"auth": True, "token": self.telemetry_token, "scheme": "Bearer"},
            extra_headers={"Cache-Control": "no-store"},
        )

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

    def _latched_state(self) -> bool:
        """Kill-switch latch flag from the published policy state."""
        return bool(self._policy_state().get("latched", False))

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
        """Live-marked positions surface from the main-cycle cache.

        Read-only on the refresher thread: the cache is populated by the
        bot's main cycle so IB is only ever touched from its own thread.
        Falls back to a direct read when the cache is empty (single-shot
        requests on the request handler thread).
        """
        bot = self.bot
        cache = getattr(bot, "_position_marks", None)
        if cache is not None:
            lock = getattr(bot, "_positions_lock", None)
            if lock is not None:
                with lock:
                    return dict(cache)
            return dict(cache)
        get_snap = getattr(bot, "_snapshot", None)
        return mark_positions(self._ib(), get_snap)

    def _recent_trades(self) -> dict[str, Any]:
        if not self.journal_path or not self.journal_path.exists():
            return {"trades": []}
        es = Journal(self.journal_path).tail(50)[::-1]
        return {
            "trades": [e for e in es if e.get("event") in ("position_closed", "exit")][
                :20
            ]
        }

    def _decisions(self) -> dict[str, Any]:
        """Full decision chain: verdicts → entries → exits → fills → P&L."""
        if not self.journal_path or not self.journal_path.exists():
            return {"chain": [], "total_events": 0}
        entries = [
            r
            for r in Journal(self.journal_path).tail(500)
            if r.get("event")
            in ("verdict", "position_closed", "exit", "entry", "fill", "position_open")
        ]
        by_ticker: dict[str, list[dict[str, Any]]] = {}
        for e in sorted(entries, key=lambda r: r.get("ts", 0)):
            t = e.get("ticker", e.get("symbol", ""))
            if t:
                by_ticker.setdefault(t, []).append(e)
        chain = [{"ticker": t, "events": evs[-12:]} for t, evs in by_ticker.items()]
        return {"chain": chain, "total_events": len(entries)}

    def _trade_quality(self) -> dict[str, Any]:
        """Live trade-quality metrics: win rate, profit factor, P&L breakdown."""
        if not self.journal_path or not self.journal_path.exists():
            return {
                "trades": 0,
                "win_rate": 0,
                "pf": 0,
                "net_pnl": 0.0,
                "breakdown": [],
            }
        trades = [
            e
            for e in Journal(self.journal_path).tail(500)
            if e.get("event") in ("position_closed", "exit")
        ]
        if not trades:
            return {
                "trades": 0,
                "win_rate": 0,
                "pf": 0,
                "net_pnl": 0.0,
                "breakdown": [],
            }
        wins = [t for t in trades if (t.get("pnl") or 0) > 0]
        losses = [t for t in trades if (t.get("pnl") or 0) < 0]
        gw = sum(t.get("pnl") or 0 for t in wins)
        gl = abs(sum(t.get("pnl") or 0 for t in losses))
        pf = (gw / gl) if gl else float("inf")
        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades), 3),
            "pf": round(pf, 3) if pf != float("inf") else "inf",
            "net_pnl": round(sum(t.get("pnl") or 0 for t in trades), 2),
            "breakdown": trades[-20:],
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
        mem = s.get("memory", {})
        return {
            "threshold": round(s.get("threshold", 0.58), 4),
            "decision_count": s.get("decision_count", 0),
            "episodic_size": s.get("episodic_size", 0),
            "weights": mem.get("weights", {}),
            "pred_error": mem.get("pred_error", 0.0),
            "brain_state": s.get("brain_state", {}),
            "neuromorphic": s.get("neuromorphic", {}),
            "realized": s.get("realized", {}),
            "nash": s.get("nash", {}),
            "advisor": s.get("advisor", {}),
            "exits_adaptive": s.get("exits_adaptive", {}),
            "meta_label": s.get("meta_label", {}),
            "horizon_bandit": s.get("horizon_bandit", {}),
            "regime_weights": s.get("regime_weights", {}),
            "learned_exit": s.get("learned_exit", {}),
            "genome": s.get("genome", {}),
            "sleep_engine": s.get("sleep_engine", {}),
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
        """Full host + process telemetry (ps-based, no deps).

        Every daemon gets cpu/rss/threads/etime; the host surfaces total
        mem, uptime, load, cpu count, and process count so the webapp can
        render complete system-load panels without guessing.
        """
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
        return {
            "procs": procs,
            "loadavg": loadavg,
            **self._host_stats(),
            "ts": time.time(),
        }

    @staticmethod
    def _host_stats() -> dict[str, Any]:
        """Host-wide vitals via ps/sysctl — best-effort, nulls on miss."""
        stats: dict[str, Any] = {
            "mem_total_mb": None,
            "mem_used_mb": None,
            "mem_pct": None,
            "cpu_count": os.cpu_count(),
            "uptime_s": None,
            "proc_count": None,
        }
        stats.update(_proc_count())
        stats["uptime_s"] = _uptime_s()
        stats.update(_mem_stats())
        return stats

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
            result = fn()
        except Exception as exc:
            return {"status": "UNVERIFIABLE", "error": str(exc)[:200]}
        try:
            manifest_notify(result)
        except Exception as exc:
            log.debug("manifest_notify failed: %s", exc)
        return result

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

    # ── Raw IB Gateway surface ─────────────────────────────────────────

    @staticmethod
    def _safe(fn: Any, default: Any) -> Any:
        """Call an ib_insync accessor; return *default* on any failure."""
        try:
            result = fn()
        except Exception:
            return default
        if _is_async(result):
            result.close()
            return default
        return result

    @staticmethod
    def _num(v: Any) -> float | None:
        """Finite float or None (never NaN/inf into JSON)."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f == f and abs(f) != float("inf") else None

    @classmethod
    def _ib_ticker_row(cls, tk: Any) -> dict[str, Any]:
        """One live Ticker as a plain JSON-able dict (raw quote surface)."""
        bid = cls._num(tk.bid)
        ask = cls._num(tk.ask)
        last = cls._num(tk.last)
        close = cls._num(tk.close)
        spread = (
            round(ask - bid, 4)
            if bid is not None and ask is not None and ask >= bid >= 0
            else None
        )
        return {
            "symbol": getattr(getattr(tk, "contract", None), "symbol", "") or "",
            "bid": bid,
            "ask": ask,
            "last": last,
            "close": close,
            "open": cls._num(tk.open),
            "high": cls._num(tk.high),
            "low": cls._num(tk.low),
            "volume": cls._num(tk.volume),
            "bid_size": cls._num(tk.bidSize),
            "ask_size": cls._num(tk.askSize),
            "last_size": cls._num(tk.lastSize),
            "halted": bool(getattr(tk, "halted", 0)),
            "spread": spread,
            "market_price": cls._num(
                cls._safe(getattr(tk, "marketPrice", lambda: None), None)
            ),
            "time": _epoch_seconds(getattr(tk, "time", None)),
        }

    @classmethod
    def _ib_order_row(cls, t: Any) -> dict[str, Any]:
        """One live IB Trade (order + fills) as a plain dict."""
        order = getattr(t, "order", None)
        contract = getattr(t, "contract", None)
        status = getattr(t, "orderStatus", None)
        return {
            **_ib_order_meta(order, contract),
            "tif": getattr(order, "tif", "") or "",
            "status": getattr(status, "status", "") or "",
            "filled": cls._num(getattr(status, "filled", None)),
            "remaining": cls._num(getattr(status, "remaining", None)),
            "avg_fill_price": cls._num(getattr(status, "avgFillPrice", None)),
            "filled_fills": len(getattr(t, "fills", []) or []),
            "fills": _ib_fill_rows(t),
        }

    def _ib_raw(self) -> dict[str, Any]:
        """Everything IB Gateway currently exposes, raw.

        One dict per connection attribute: status, accounts, account
        summary values, positions, live tickers (quotes), open/managed
        orders with fills, executions, and recent error codes. Every
        accessor is guarded — a half-open socket degrades to nulls, not
        a broken /ib route.
        """
        ib = self._ib()
        out: dict[str, Any] = {
            "connected": False,
            "ts": round(time.time(), 3),
        }
        if ib is None:
            out["error"] = "no ib client (adapter not wired)"
            return out
        try:
            out["connected"] = bool(ib.isConnected())
        except Exception as exc:
            log.debug("ib connectivity check failed: %s", exc)
        if not out["connected"]:
            out["error"] = "disconnected"
            return out

        out.update(self._ib_conn_facts(ib))
        out["tickers"] = [
            type(self)._ib_ticker_row(tk)
            for tk in self._safe(lambda: list(ib.tickers()), [])
        ]
        out["orders"] = self._ib_order_rows(ib)
        out["executions"] = self._ib_exec_rows(ib)
        out["errors"] = self._safe(lambda: list(ib.client._logger.errors)[-20:], [])
        out["error"] = None
        return out

    def _ib_conn_facts(self, ib: Any) -> dict[str, Any]:
        """Connection-level facts: client id, version, accounts, summary."""
        accounts = self._safe(lambda: list(ib.managedAccounts()), [])
        return {
            "client_id": self._safe(lambda: ib.client.clientId, None),
            "server_version": self._safe(lambda: ib.client.serverVersion, None),
            "conn_time": self._safe(
                lambda: (
                    int(ib.client.connTime.timestamp()) if ib.client.connTime else None
                ),
                None,
            ),
            "accounts": accounts,
            "account_summary": self._ib_account_summary(ib, accounts),
            "positions": self._ib_position_rows(ib),
        }

    def _ib_account_summary(self, ib: Any, accounts: list[str]) -> dict[str, Any]:
        """Per-account raw summary tag values."""
        summary: dict[str, dict[str, Any]] = {}
        for acct in accounts[:2]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raw = self._safe(lambda a=acct: ib.accountSummary(a), None)
            items = list(raw) if raw is not None else []
            vals: dict[str, Any] = {}
            for it in items:
                v = self._num(getattr(it, "value", None))
                vals[getattr(it, "tag", "")] = {
                    "value": v if v is not None else str(getattr(it, "value", "")),
                    "currency": getattr(it, "currency", "") or "",
                }
            summary[acct] = vals
        return summary

    def _ib_position_rows(self, ib: Any) -> list[dict[str, Any]]:
        """Raw IB positions as JSON-able dicts."""
        rows: list[dict[str, Any]] = []
        positions = self._safe(lambda: list(ib.positions()), [])
        for p in positions:
            contract = getattr(p, "contract", None)
            rows.append(
                {
                    "account": getattr(p, "account", "") or "",
                    "symbol": getattr(contract, "symbol", "") or "",
                    "sec_type": getattr(contract, "secType", "") or "",
                    "position": self._num(p.position),
                    "avg_cost": self._num(p.avgCost),
                }
            )
        return rows

    def _ib_order_rows(self, ib: Any) -> list[dict[str, Any]]:
        """Open + managed trades, deduplicated by (orderId, permId)."""
        seen: set[tuple[Any, ...]] = set()
        rows: list[dict[str, Any]] = []
        trades = self._safe(lambda: list(ib.openTrades()), []) + self._safe(
            lambda: list(ib.trades()), []
        )
        for t in trades:
            order = getattr(t, "order", None)
            perm = getattr(order, "permId", None)
            key = (getattr(order, "orderId", None), perm)
            if key in seen:
                continue
            seen.add(key)
            rows.append(type(self)._ib_order_row(t))
        return rows

    def _ib_exec_rows(self, ib: Any) -> list[dict[str, Any]]:
        """Recent executions (capped to 100)."""
        rows: list[dict[str, Any]] = []
        for ex in self._safe(lambda: list(ib.executions()), []):
            rows.append(
                {
                    "exec_id": getattr(ex, "execId", ""),
                    "order_id": getattr(ex, "orderId", None),
                    "perm_id": getattr(ex, "permId", None),
                    "symbol": getattr(getattr(ex, "contract", None), "symbol", "")
                    or "",
                    "time": _format_exec_time(ex),
                    "side": getattr(ex, "side", "") or "",
                    "shares": self._num(getattr(ex, "shares", None)),
                    "price": self._num(getattr(ex, "price", None)),
                    "exchange": getattr(ex, "exchange", "") or "",
                }
            )
        return rows[-100:]

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
        """Request flatten with optional order_type (\"market\"|\"limit\") + limit_price."""
        bot = self.bot
        if bot is None:
            self._r(503, {"error": "bot not running"})
            return
        ib = self._ib()
        if ib is None or not ib.isConnected():
            self._r(503, {"error": "IB not connected"})
            return
        body = self._body()
        order_type = body.get("order_type") or TRADING_CONFIG.flatten_order_type
        if order_type not in ("market", "limit"):
            self._r(400, {"error": 'order_type must be "market" or "limit"'})
            return
        limit_price = float(body["limit_price"]) if body.get("limit_price") else None
        pos_count = len(self._ib_positions())
        # Set flag — main cycle thread will execute flatten
        _FLATTEN_REQUESTED.clear()
        _FLATTEN_REQUESTED.update(
            {
                "positions": pos_count,
                "order_type": order_type,
                "limit_price": limit_price,
            }
        )
        log.warning("FLATTEN requested: %d positions, order_type=%s", pos_count, order_type)
        self._r(
            200,
            {
                "flattened": 0,
                "pending": True,
                "action": "flatten_all",
                "positions": pos_count,
                "order_type": order_type,
                "limit_price": limit_price,
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

    def _log_tail(self) -> dict[str, Any]:
        """Return raw tail lines from the bot log file."""
        candidates: list[Path] = []
        if self.journal_path:
            jp = self.journal_path.resolve()
            for parent in jp.parents:
                if (parent / "src" / "hanoon_prime").is_dir():
                    candidates.append(parent / "logs" / "hanoon_prime.log")
                    candidates.append(parent / "runtime" / "hanoon_prime.log")
                    break
        candidates.append(Path("/tmp/hanoon_prime.log"))
        candidates.append(Path.home() / "Library" / "Logs" / "hanoon_prime.log")
        return {"lines": _log_tail_lines(candidates), "ts": time.time()}


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
        frame = (
            f"event: snapshot\ndata: {json.dumps(snapshot, default=str)}\n\n".encode()
        )
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
        "decisions": handler._decisions(),
        "trade_quality": handler._trade_quality(),
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
        log.info(
            "Telemetry snapshot refresher started (%.1fs cadence)", SNAPSHOT_INTERVAL
        )
        while not self._stop.is_set():
            t0 = time.time()
            self._build_and_cache()
            # Sleep the remainder of the interval in small slices.
            elapsed = time.time() - t0
            remaining = max(0.05, SNAPSHOT_INTERVAL - elapsed)
            if self._stop.wait(remaining):
                break

    # ── security: bearer auth for mutations + CORS origin ──────────────

    def _token_path(self) -> Path:
        """Where the runtime bearer token lives (0600, gitignored)."""
        return self._jp.resolve().parent / "telemetry.token"

    def _apply_security_posture(self) -> None:
        """Bind bearer-token auth + CORS origin onto the request handler."""
        if TELEMETRY_AUTH_ENABLED:
            self._ensure_token_file()
            _H.auth_enabled = True
            _H.telemetry_token = self._load_token()
        else:
            _H.auth_enabled = False
            _H.telemetry_token = ""
        # CORS allow-list is resolved per-request in _cors_origin(): the
        # tunnel origin can change on restart, so it is never pinned here.
        _H.cors_origin = None

    def _ensure_token_file(self) -> Path:
        tp = self._token_path()
        tp.parent.mkdir(parents=True, exist_ok=True)
        if not tp.exists():
            tp.write_text(secrets.token_urlsafe(32))
            try:
                os.chmod(tp, 0o600)
            except OSError as exc:
                log.debug("chmod token file 0600 failed: %s", exc)
            log.warning("telemetry bearer token generated at %s (keep SECRET)", tp)
        return tp

    def _load_token(self) -> str:
        try:
            return self._token_path().read_text().strip()
        except OSError:
            return ""

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self, port: int | None = None) -> None:
        """Start HTTP server + snapshot refresher in background threads.

        ``port`` overrides TELEMETRY_PORT (used by tests with port=0 for an
        ephemeral bind); production callers keep the default.
        """
        handler = self._make_handler()
        self._apply_security_posture()
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
        if TELEMETRY_AUTH_ENABLED:
            log.warning(
                "telemetry bearer auth ON; POST mutations require token at %s",
                self._token_path(),
            )
        else:
            log.warning("telemetry bearer auth OFF — POST mutations ungated")
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
