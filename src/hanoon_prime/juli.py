"""hanoon_prime.juli — Thin scanner router for the Neuromorphic Brain.

NeuromorphicBrain is the LOCAL SOURCE OF TRUTH for all decisions.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any

from .brain.orchestrator import NeuromorphicBrain
from .brain.shared_state import BrainState
from .data.budget import DataBudget
from .data.scanner import IBScanner, ScanResult
from .juli_feed import JuliFeed, check_tick_latency, compute_alpha_from_snap, entry_bars

log = logging.getLogger(__name__)
MAX_CANDIDATES: int = 20
# Full-parallel-throttle: how many tickers get entry-scored per cycle.
# Rotating a small window keeps the flow continuous; the main loop never
# stalls on a fixed batch (no THINK burst, then silence).
EVAL_WINDOW: int = 4


class JuliBrain:
    """Scanner + router. NeuromorphicBrain makes all decisions."""

    def __init__(self, ib_client: Any) -> None:
        self.ib = ib_client
        self.scanner = IBScanner(ib_client)
        self.budget = DataBudget()
        self._state = BrainState()
        self.brain = NeuromorphicBrain(brain_state=self._state)
        self._candidates: list[ScanResult] = []
        self._last_alloc: float = 0.0
        self.feed = JuliFeed(self._state)
        self.brain.start()

    def tick(
        self,
        positions: set[str],
        get_snapshot: Any,
        streamer: Any,
        closing: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """One full brain cycle. Returns (entry_decisions, exit_signals)."""
        self.feed.ensure_refs(streamer)
        self._sync_and_scan()
        self._maybe_screen(get_snapshot)
        self.feed.fallback_regime()
        self._maybe_allocate(positions)
        exits = self._evaluate_exits(positions, get_snapshot, closing or set())
        entries = self._evaluate_entries(positions, get_snapshot)
        return exits, entries

    def _sync_and_scan(self) -> None:
        """Sync scanner results and start new scan if due."""
        if self.scanner.should_scan():
            try:
                self.scanner.scan("most_active")
            except Exception as e:
                log.warning("Scan start failed: %s", e)
        try:
            if results := self.scanner.collect():
                self._candidates = results
        except Exception as e:
            log.debug("Scan collect error: %s", e)

    def _maybe_screen(self, get_snapshot: Any) -> None:
        """Screen candidates + publish cross-asset ref prices."""
        if not self._candidates:
            return
        snaps = [get_snapshot(c.symbol) for c in self._candidates[:MAX_CANDIDATES]]
        n = sum(1 for s in snaps if s and s.get("last", 0) > 0)
        log.info("SCREEN: %d/%d passed", n, len(self._candidates))
        self.feed.publish_ref_prices(get_snapshot)

    def _maybe_allocate(self, positions: set[str]) -> None:
        """Allocate data budget periodically."""
        if time.time() - self._last_alloc < 5.0:
            return
        self._last_alloc = time.time()
        self.budget.allocate(
            positions, [c.symbol for c in self._candidates[:MAX_CANDIDATES]]
        )

    def _evaluate_exits(
        self, positions: set[str], get_snapshot: Any, closing: set[str]
    ) -> list[dict[str, Any]]:
        """Evaluate open positions for exit signals."""
        exits = []
        for t in positions:
            if t in closing:
                continue
            snap = get_snapshot(t)
            if snap is None or snap.get("last", 0) <= 0:
                continue
            sig = self.brain.check_exit(t, snap["last"], direction=1)
            if sig.should_exit:
                exits.append({"ticker": t, "reason": sig.reason, "type": sig.exit_type})
                log.info("EXIT SIGNAL %s: %s", t, sig.reason)
        return exits

    def _evaluate_entries(
        self, positions: set[str], get_snapshot: Any
    ) -> list[dict[str, Any]]:
        """Evaluate a rotating subset of tracked tickers for entry decisions.

        Full parallel throttle: instead of scoring every candidate every
        cycle (which produced a THINK burst, then silence), rotate through
        the tracked universe a few at a time so scoring is smooth and the
        main loop never stalls on a fixed batch.
        """
        universe = sorted(self.budget.get_all_tracked() | positions)
        if not universe:
            return []
        # Re-own the rotation cursor (persisted across cycles on self)
        off = getattr(self, "_eval_off", 0) % len(universe)
        window = EVAL_WINDOW
        slice_ = universe[off : off + window]
        self._eval_off = (off + window) % len(universe)
        decisions = []
        for ticker in slice_:
            snap = get_snapshot(ticker)
            if snap is None:
                continue
            try:
                dec = self._eval_one(ticker, snap, len(positions))
            except Exception as e:
                # Counted, never just logged away (FIXES.md Class D):
                # PipelineMonitor alerts when failures accumulate.
                self.brain.note_eval_failure(ticker, e)
                log.warning("Entry eval failed for %s: %s", ticker, e)
                continue
            if dec is not None:
                decisions.append(dec)
        return decisions

    def _build_decision(
        self, ticker: str, direction: int, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Build decision dict from brain result."""
        score, conf = result.get("score", 0), result.get("confidence", 0.5)
        verdict = result.get("verdict", "")
        log.info(
            "THINK %s %s score=%.3f regime=%s risk=%s hz=%s/%s",
            ticker,
            "BUY" if direction > 0 else "SELL",
            score,
            result.get("regime", "?"),
            result.get("risk", "normal"),
            result.get("horizon", "scalp"),
            result.get("horizon_reason", "classifier"),
        )
        return {
            "ticker": ticker,
            "direction": direction,
            "verdict": verdict,
            "score": score,
            "thought": SimpleNamespace(
                direction=direction, score=score, verdict=verdict, confidence=conf
            ),
            "sizing": result.get("sizing"),
            "horizon": result.get("horizon", "scalp"),
            "regime_canon": result.get("regime_canon", "unknown"),
        }

    def _eval_one(
        self, ticker: str, snap: dict[str, Any], open_count: int
    ) -> dict[str, Any] | None:
        """Evaluate one ticker through the full brain pipeline."""
        prices = snap.get("prices") or []  # array-safe: list-typed
        if len(prices) < 20:
            return None
        self._state.set_latest_prices(prices)
        t0 = time.perf_counter_ns()
        result = self.brain.tick(
            alpha=compute_alpha_from_snap(snap),
            ticker=ticker,
            entry_price=float(prices[-1]),
            atr=snap.get("atr", 1.0),
            open_positions=open_count,
            bars=entry_bars(snap, prices, self._state.get("regime_label", "unknown")),
        )
        check_tick_latency(t0, ticker)
        direction = result.get("direction", 0)
        return (
            self._build_decision(ticker, direction, result) if direction != 0 else None
        )

    def on_trade_close(
        self, ticker: str, won: bool, pnl_pct: float, direction: int = 1
    ) -> None:
        """Route trade close to NeuromorphicBrain — all learning here."""
        self.brain.on_trade_close(ticker, won, pnl_pct, direction)
