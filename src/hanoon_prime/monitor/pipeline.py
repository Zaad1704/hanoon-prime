"""monitor.pipeline — continuous pipeline health daemon for the live bot.

A daemon thread that evaluates read-only health probes every interval.
The cycle thread (the only thread allowed to touch ib_insync) publishes
vitals via record_cycle(); this thread only reads plain attributes,
evaluates probes, journals incidents, and sets heal flags that the
cycle consumes on its next iteration. Never places orders; alerts only.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

Probe = Callable[[], tuple[bool, str]]

BAR_STALE_SECS: float = 180.0
BRAIN_STALL_CYCLES: int = 10


class PipelineMonitor:
    """Continuous pipeline health monitoring (alert-only, self-heal flags)."""

    def __init__(self, bot: Any, journal: Any, interval: float = 15.0) -> None:
        """Bind to the live bot; probes run on a daemon thread."""
        self._bot = bot
        self._journal = journal
        self._interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._vitals: dict[str, Any] = {}
        self._failures: dict[str, tuple[bool, str]] = {}
        self._heal: dict[str, bool] = {"resub": False}
        self._incidents: list[dict[str, Any]] = []
        self._last_sizes: dict[str, int] = {}
        self._last_eval_fail: int = 0
        self._last_decisions: int = 0
        self._stall_cycles: int = 0
        self._last_bar_growth: float = time.time()

    # ── cycle-thread side ────────────────────────────────────────────
    def record_cycle(self, market_open: bool, session: str = "rth") -> None:
        """Publish vitals from the cycle thread (cheap, lock-light)."""
        streamer = self._bot.streamer
        sizes = {
            t: len(getattr(buf, "close", []) or [])
            for t, buf in list(getattr(streamer, "buffers", {}).items())
        }
        grew = any(sizes.get(t, 0) > self._last_sizes.get(t, 0) for t in sizes)
        if grew:
            self._last_bar_growth = time.time()
        self._last_sizes = sizes
        try:
            connected = bool(self._bot.ib.isConnected())
        except Exception:
            connected = False
        decisions = int(getattr(self._bot.juli.brain, "_decision_count", 0))
        with self._lock:
            self._vitals = {
                "ts": time.time(),
                "market_open": market_open,
                "session": session,
                "session_active": market_open,
                "ib_connected": connected,
                "bar_sizes": sizes,
                "decision_count": decisions,
                "journal_bytes": self._journal_path_size(),
            }
            if market_open and decisions == self._last_decisions:
                self._stall_cycles += 1
            elif market_open:
                self._stall_cycles = 0
            self._last_decisions = decisions

    def pop_heal(self) -> bool:
        """Consume the re-subscribe heal flag (cycle thread)."""
        with self._lock:
            if self._heal["resub"]:
                self._heal["resub"] = False
                return True
        return False

    # ── monitor-thread side ──────────────────────────────────────────
    def start(self) -> None:
        """Start the daemon loop."""
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pipeline")
        self._thread.start()
        log.info("PipelineMonitor started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        """Stop the daemon loop."""
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._check_once()
            except Exception as e:
                log.error("PipelineMonitor check error: %s", e)

    def _collect_checks(self, v: dict[str, Any]) -> list[tuple[str, bool, str]]:
        """Build probe results (pure reads; eval-fail counter updated)."""
        open_mkt = bool(v.get("market_open"))
        ef = int(getattr(self._bot.juli, "_eval_fail_count", 0) or 0)
        eval_burst = open_mkt and ef - self._last_eval_fail >= 3
        self._last_eval_fail = ef
        return [
            ("ib_connected", bool(v.get("ib_connected")), "Gateway link"),
            (
                "bars_fresh",
                (not open_mkt)
                or (time.time() - self._last_bar_growth < BAR_STALE_SECS),
                f"last bar growth {time.time() - self._last_bar_growth:.0f}s ago",
            ),
            (
                "brain_advancing",
                self._stall_cycles < BRAIN_STALL_CYCLES,
                f"{self._stall_cycles} cycles without a decision",
            ),
            (
                "entry_evals",
                not eval_burst,
                f"{ef - self._last_eval_fail} new eval failures",
            ),
            (
                "subs_present",
                (not open_mkt) or bool(v.get("bar_sizes")),
                f"{len(v.get('bar_sizes') or {})} tickers buffered",  # array-safe: dict-typed
            ),
        ]

    def _check_once(self) -> None:
        """Run all probes; alert on new failures, note recoveries."""
        v = dict(self._vitals)
        if not v.get("ts"):
            return
        open_mkt = bool(v.get("market_open"))
        checks = self._collect_checks(v)
        for name, ok, detail in checks:
            prev = self._failures.get(name)
            if not ok:
                self._failures[name] = (ok, detail)
                if prev is None or prev[1] != detail:
                    self._incident(name, detail, v)
            elif name in self._failures:
                del self._failures[name]
                log.info("PIPELINE RECOVERED: %s", name)
        if open_mkt and not v.get("ib_connected"):
            log.critical("PIPELINE: IB disconnected while market open")
        if open_mkt and self._stall_cycles >= BRAIN_STALL_CYCLES:
            with self._lock:
                self._heal["resub"] = True

    def _incident(self, name: str, detail: str, v: dict[str, Any]) -> None:
        """Log, journal, and remember one failure transition."""
        log.critical("PIPELINE FAIL: %s (%s)", name, detail)
        inc = {
            "ts": time.time(),
            "check": name,
            "detail": detail,
            "market_open": v.get("market_open"),
        }
        with self._lock:
            self._incidents.append(inc)
            self._incidents = self._incidents[-50:]
        try:
            self._journal.append({"event": "pipeline_incident", **inc})
        except Exception:
            log.warning("Incident journal write failed")

    def _journal_path_size(self) -> int:
        try:
            p: Path = self._journal._path
            return p.stat().st_size if p.exists() else 0
        except Exception:
            return -1

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view of pipeline health."""
        with self._lock:
            return {
                "healthy": not self._failures,
                "failing": {k: d for k, (_, d) in self._failures.items()},
                "vitals": dict(self._vitals),
                "stall_cycles": self._stall_cycles,
                "incidents_recent": self._incidents[-5:],
            }
