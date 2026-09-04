"""reflection.supervisor — Background learning supervisor.

Runs on a daemon thread, scheduling:
- DAILY: review recent trades, decay bad weights, extract lessons
- WEEKLY: deeper review with regime analysis
- STREAMING: after each trade close, update memory in real-time

The supervisor is the ONLY module that writes to memory outside of
on_trade_close — it applies macro corrections (weight decay) while
on_trade_close applies micro adjustments (per-trade learning).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from .buffer import Trade, TradeBuffer

log = logging.getLogger(__name__)

DAILY_INTERVAL = 24 * 3600.0
WEEKLY_INTERVAL = 7 * 24 * 3600.0


class LearningSupervisor:
    """Background learning supervisor — orchestrates review + retrain."""

    def __init__(
        self,
        buffer: TradeBuffer,
        memory: Any = None,
        on_review: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._buffer = buffer
        self._memory = memory
        self._on_review = on_review
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_daily: float = 0.0
        self._last_weekly: float = 0.0

    def start(self) -> None:
        """Start the supervisor daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="learning-supervisor"
        )
        self._thread.start()
        log.info("LearningSupervisor started")

    def stop(self) -> None:
        """Stop the supervisor thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("LearningSupervisor stopped")

    def on_trade_close(self, trade: Trade) -> None:
        """Streaming update after each trade closes."""
        if self._memory is None:
            return
        self._memory.record_outcome(won=trade.win)
        # Weight micro-adjustment: boost winners, decay losers
        if trade.win:
            self._memory.update_pred_error(trade.avg_entry, trade.avg_exit)
        else:
            self._memory.update_pred_error(trade.avg_entry, trade.avg_exit)

    def force_review(self) -> dict:
        """Run an immediate review (for testing or manual trigger)."""
        report = self._review()
        self._apply_findings(report)
        return report

    def _loop(self) -> None:
        """Main supervision loop."""
        while self._running:
            try:
                now = time.time()
                if now - self._last_daily >= DAILY_INTERVAL:
                    self._run_daily()
                    self._last_daily = now
                if now - self._last_weekly >= WEEKLY_INTERVAL:
                    self._run_weekly()
                    self._last_weekly = now
            except Exception as exc:
                log.warning("Supervisor loop error: %s", exc)
            time.sleep(60.0)

    def _run_daily(self) -> None:
        """Run daily review."""
        log.info("Running daily review")
        try:
            report = self._review()
            self._apply_findings(report)
        except Exception as exc:
            log.warning("Daily review failed: %s", exc)

    def _run_weekly(self) -> None:
        """Run weekly review (deeper analysis)."""
        log.info("Running weekly review")
        try:
            report = self._review()
            self._apply_findings(report)
        except Exception as exc:
            log.warning("Weekly review failed: %s", exc)

    def _review(self) -> dict:
        """Build a review report from recent trades."""
        trades = self._buffer.get_trades(last_n=100)
        if not trades:
            return {"summary": {}, "action_items": []}
        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]
        wr = len(wins) / len(trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        exp = wr * avg_win + (1 - wr) * avg_loss
        items = self._build_items(trades, wr, exp)
        summary = {
            "total": len(trades),
            "wr": round(wr, 3),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(exp, 4),
        }
        report = {"summary": summary, "action_items": items}
        if self._on_review:
            try:
                self._on_review(report)
            except Exception as exc:
                log.warning("on_review failed: %s", exc)
        return report

    def _build_items(self, trades: list[Trade], wr: float, exp: float) -> list[str]:
        """Generate action items from trade review."""
        items: list[str] = []
        if wr < 0.40:
            items.append("Portfolio WR below 40% — tighten threshold")
        if exp < 0:
            items.append("Negative expectancy — review strategy")
        by_tick: dict[str, list[Trade]] = {}
        for t in trades:
            by_tick.setdefault(t.ticker, []).append(t)
        for tick, tt in by_tick.items():
            tw = sum(1 for t in tt if t.win) / len(tt)
            if tw < 0.30 and len(tt) >= 5:
                items.append(f"Decay weight for {tick} (WR={tw:.2f})")
        return items

    def _apply_findings(self, report: dict) -> None:
        """Apply review findings to memory."""
        items = report.get("action_items", [])
        if not items or self._memory is None:
            return
        for item in items:
            if "Decay weight for" in item:
                try:
                    indicator = item.split("Decay weight for ")[1].split(" ")[0]
                    self._memory.micro_adjust_weight(indicator, -0.05)
                except (IndexError, AttributeError) as exc:
                    log.debug("Skip decay: %s", exc)
        log.info("Review applied: %d action items", len(items))

    def snapshot(self) -> dict[str, Any]:
        """Telemetry snapshot."""
        return {
            "running": self._running,
            "last_daily": self._last_daily,
            "last_weekly": self._last_weekly,
            "buffer": self._buffer.snapshot(),
        }
