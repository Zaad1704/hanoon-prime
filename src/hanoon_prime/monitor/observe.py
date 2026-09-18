"""monitor.observe — observe-only runner for the dormant monitors.

Watchdog, DecisionHealthTracker, DeepEnforcement, PositionMonitor,
Reconciliation and ExitScorer are wired read-only so their state is visible
in telemetry. This suite NEVER calls Watchdog.check_panic(), NEVER starts
PositionMonitor's loop, and NEVER writes BrainState: the auto-flatten path
stays unreachable. It only reads live bot state on a daemon thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .decision_health import DecisionHealthTracker
from .enforcement import DeepEnforcement, HealthBudget, HealthDiagnosis
from .position_monitor import PositionMonitor
from .reconciliation import Reconciliation

log = logging.getLogger(__name__)

TICK_FRESH_SECS: float = 10.0


class MonitorSuite:
    """Read-only runner that surfaces the dormant monitors in telemetry."""

    def __init__(self, bot: Any, state: Any, pulse_sec: float = 5.0) -> None:
        self._bot = bot
        self._pulse = pulse_sec
        # PositionMonitor is instantiated WITHOUT start(): its panic/auto-flatten
        # loop is unreachable. Reuse its scorer + watchdog as the sole instances.
        self._pm = PositionMonitor(state, pulse_sec=pulse_sec)
        self.scorer = self._pm.scorer
        self.watchdog = self._pm.watchdog
        self.decisions = DecisionHealthTracker()
        self.enforcement = DeepEnforcement()
        self.budget = HealthBudget()
        self.diagnosis = HealthDiagnosis()
        self.reconciliation = Reconciliation()
        self._report: dict[str, Any] = {}
        self._seen_trades = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Seed history and start the read-only daemon thread."""
        self._seen_trades = len(self._trades())
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="monitor-observe"
        )
        self._thread.start()
        log.info("Observe-only monitor suite started (pulse=%.0fs)", self._pulse)

    def stop(self) -> None:
        """Stop the read-only daemon thread."""
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self._pulse):
            try:
                self.pulse()
            except Exception as exc:
                log.debug("monitor observe pulse failed: %s", exc)

    def _brain(self) -> Any:
        return getattr(getattr(self._bot, "juli", None), "brain", None)

    def _trades(self) -> list[Any]:
        cons = getattr(self._brain(), "_consolidation", None)
        buf = getattr(cons, "buffer", None) if cons else None
        return list(buf.get_trades()) if buf else []

    def _tick_map(self) -> dict[str, float]:
        return getattr(getattr(self._bot, "streamer", None), "last_data_ts", {}) or {}

    def _position_map(self) -> dict[str, Any]:
        marks = getattr(self._bot, "_position_marks", None) or {}
        positions = marks.get("positions")
        if positions is None:
            positions = []
        out: dict[str, Any] = {}
        for p in positions:
            sym = getattr(getattr(p, "contract", None), "symbol", None)
            if sym:
                out[sym] = p
        return out

    def _orders(self) -> dict[str, Any]:
        ex = getattr(self._bot, "executor", None)
        return dict(getattr(ex, "last_thoughts", {}) or {})

    def _feed_ticks(self) -> None:
        now = time.time()
        for ticker, ts in self._tick_map().items():
            if ts and now - float(ts) <= TICK_FRESH_SECS:
                self.watchdog.tick_received(ticker)

    def _feed_trades(self) -> None:
        trades = self._trades()
        for t in trades[self._seen_trades :]:
            won = bool(getattr(t, "win", False))
            ticker = str(getattr(t, "ticker", "?"))
            self.decisions.record_entry(ticker, 0.0, won)
            self.decisions.record_exit(ticker, "close", won)
        self._seen_trades = len(trades)

    def _enforce(self) -> None:
        rep = self.enforcement.run_pulse(
            self._position_map(), self._orders(), self._brain()
        )
        # fmt: off
        self._report = {
            "score": round(rep.score, 3),
            "has_critical": rep.has_critical,
            "checks": [
                {"id": c.check_id, "name": c.name, "ok": c.passed, "detail": c.detail}
                for c in rep.checks
            ],
        }
        # fmt: on
        self.budget.update("enforcement", rep.score)
        self.budget.update("decision_health", self.decisions.get_health().overall_score)
        panic = self.watchdog.snapshot().get("panic_mode", False)
        self.budget.update("watchdog", 0.0 if panic else 1.0)

    def pulse(self) -> None:
        """One read-only observation cycle (ticks, trades, enforcement)."""
        self._feed_ticks()
        self._feed_trades()
        self._enforce()

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view of every observe-only monitor."""
        h = self.decisions.get_health()
        # fmt: off
        return {
            "enabled": True,
            "mode": "observe-only",
            "watchdog": self.watchdog.snapshot(),
            "decision_health": {
                "overall_score": round(h.overall_score, 3),
                "entry_quality": round(h.entry_quality, 3),
                "exit_quality": round(h.exit_quality, 3),
                "recent_accuracy": round(h.recent_accuracy, 3),
                "n_decisions": h.n_decisions,
                "issues": list(h.issues or []),
            },
            "enforcement": self._report,
            "health_budget": {
                "overall": round(self.budget.get_overall(), 3),
                "scores": {k: round(v, 3) for k, v in self.budget.scores().items()},
                "critical": self.budget.get_critical_modules(),
                "diagnosis": self.diagnosis.diagnose(self.budget.scores()),
            },
            "exit_scorer": self.scorer.snapshot(),
            "reconciliation": self.reconciliation.snapshot(),
        }
        # fmt: on
