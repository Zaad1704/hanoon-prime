"""hanoon_prime.juli — Thin scanner router for the Neuromorphic Brain.

NeuromorphicBrain is the LOCAL SOURCE OF TRUTH for all decisions.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any

from .brain.indicators import compute_all_alpha
from .brain.orchestrator import NeuromorphicBrain
from .brain.shared_state import BrainState
from .cerebellum import compute_alpha
from .data.budget import DataBudget
from .data.scanner import IBScanner, ScanResult
from .types import BarSeries

log = logging.getLogger(__name__)
MAX_CANDIDATES: int = 20
# (brain key, snapshot key) — snapshot arrays feed alpha computation.
_KEYS = "close high low volume buy_volume bid_sizes ask_sizes"
_SRC = "close_arr high_arr low_arr vol_arr buy_vol_arr bid_sizes ask_sizes"
ATTRS = tuple(zip(_KEYS.split(), _SRC.split()))


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
        self.brain.start()

    def tick(
        self,
        positions: set[str],
        get_snapshot: Any,
        _streamer: Any,
        closing: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """One full brain cycle. Returns (entry_decisions, exit_signals)."""
        self._sync_and_scan()
        self._maybe_screen(get_snapshot)
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
        """Screen candidates through snapshots."""
        if not self._candidates:
            return
        snaps = [get_snapshot(c.symbol) for c in self._candidates[:MAX_CANDIDATES]]
        n = sum(1 for s in snaps if s and s.get("last", 0) > 0)
        log.info("SCREEN: %d/%d passed", n, len(self._candidates))

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
                exits.append({"ticker": t, "reason": sig.reason})
                log.info("EXIT SIGNAL %s: %s", t, sig.reason)
        return exits

    def _evaluate_entries(
        self, positions: set[str], get_snapshot: Any
    ) -> list[dict[str, Any]]:
        """Evaluate all tracked tickers for entry decisions."""
        decisions = []
        for ticker in self.budget.get_all_tracked() | positions:
            snap = get_snapshot(ticker)
            if snap is None:
                continue
            try:
                dec = self._eval_one(ticker, snap, len(positions))
            except Exception as e:
                log.warning("Entry eval failed for %s: %s", ticker, e)
                continue
            if dec is not None:
                decisions.append(dec)
        return decisions

    def _entry_bars(self, snap: dict[str, Any], prices: list[float]) -> dict[str, Any]:
        """Bar context INTO the brain (horizon classification happens there)."""
        return {
            "close": snap.get("close_arr") or prices,
            "high": snap.get("high_arr"),
            "low": snap.get("low_arr"),
            "regime": self._state.get("regime_label", "unknown"),
        }

    def _compute_alpha(self, snap: dict[str, Any]) -> dict[str, float]:
        """Compute all indicators from snapshot arrays."""
        kw: dict[str, Any] = {}
        for k, u in ATTRS:
            v = snap.get(u)
            kw[k] = [] if v is None else (v if isinstance(v, list) else list(v))
        if len(kw["close"]) < 20:
            return {}
        alpha = compute_all_alpha(BarSeries(**kw))
        return alpha if alpha else (compute_alpha(**kw) or {})

    def _build_decision(
        self, ticker: str, direction: int, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Build decision dict from brain result."""
        score, conf = result.get("score", 0), result.get("confidence", 0.5)
        verdict = result.get("verdict", "")
        log.info(
            "THINK %s %s score=%.3f regime=%s risk=%s hz=%s",
            ticker,
            "BUY" if direction > 0 else "SELL",
            score,
            result.get("regime", "?"),
            result.get("risk", "normal"),
            result.get("horizon", "scalp"),
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
        }

    def _eval_one(
        self, ticker: str, snap: dict[str, Any], open_count: int
    ) -> dict[str, Any] | None:
        """Evaluate one ticker through the full brain pipeline."""
        prices = snap.get("prices") or []
        if len(prices) < 20:
            return None
        self._state.set_latest_prices(prices)
        t0 = time.perf_counter_ns()
        result = self.brain.tick(
            alpha=self._compute_alpha(snap),
            ticker=ticker,
            entry_price=float(prices[-1]),
            atr=snap.get("atr", 1.0),
            open_positions=open_count,
            bars=self._entry_bars(snap, prices),
        )
        latency_us = (time.perf_counter_ns() - t0) / 1000.0
        if latency_us > 1000.0:
            log.warning("Tick latency spike: %.0f us for %s", latency_us, ticker)
        direction = result.get("direction", 0)
        return (
            self._build_decision(ticker, direction, result) if direction != 0 else None
        )

    def on_trade_close(
        self, ticker: str, won: bool, pnl_pct: float, direction: int = 1
    ) -> None:
        """Route trade close to NeuromorphicBrain — all learning here."""
        self.brain.on_trade_close(ticker, won, pnl_pct, direction)
