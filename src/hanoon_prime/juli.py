"""hanoon_prime.juli — Biological brain orchestrator.

JULI is the full brain: scanner → indicators → JuliBrain → decision.
Keeps IB scanner/budget for ticker discovery.
Delegates all cognition to brain/orchestrator.py (the 22-module brain).
Returns (decisions, exit_signals) for ib_adapter to execute.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .brain.indicators import compute_all_alpha
from .brain.orchestrator import JuliBrain as BrainPipeline
from .brain.shared_state import BrainState
from .brain.slow_cortex import SlowCortex
from .cerebellum import compute_alpha
from .data.budget import DataBudget
from .data.scanner import IBScanner, ScanResult

log = logging.getLogger(__name__)
MAX_CANDIDATES: int = 20


class JuliBrain:
    """Biological brain: scanner discovery + full cognitive pipeline."""

    def __init__(self, ib_client: Any) -> None:
        self.ib = ib_client
        self.scanner = IBScanner(ib_client)
        self.budget = DataBudget()
        self._state = BrainState()
        self.brain = BrainPipeline(brain_state=self._state)
        self.slow_cortex = SlowCortex(brain_state=self._state)
        self._candidates: list[ScanResult] = []
        self._last_scan: float = 0.0
        self._last_alloc: float = 0.0
        self._open_positions: dict[str, Any] = {}
        self.slow_cortex.start()

    def tick(
        self, positions: set[str], get_snapshot: Any, streamer: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """One full brain cycle. Returns (entry_decisions, exit_signals)."""
        self._open_positions = dict.fromkeys(positions, True)
        self._maybe_scan()
        self._maybe_screen(get_snapshot)
        self._maybe_allocate(positions)
        return self._evaluate_exits(positions, get_snapshot), self._evaluate_entries(
            positions, get_snapshot
        )

    def _maybe_scan(self) -> None:
        """Start scanner and collect results every 5 minutes."""
        if not self.scanner.should_scan():
            return
        try:
            self.scanner.scan("most_active")
            self._candidates = self.scanner.collect()
        except Exception as e:
            log.warning("Scan failed: %s", e)

    def _maybe_screen(self, get_snapshot: Any) -> None:
        """Screen candidates — subscribe only, budget handles allocation."""
        if not self._candidates:
            return
        passed = sum(
            1
            for c in self._candidates[:MAX_CANDIDATES]
            if (snap := get_snapshot(c.symbol)) is not None and snap.get("last", 0) > 0
        )
        log.info("SCREEN: %d/%d passed", passed, len(self._candidates))

    def _maybe_allocate(self, positions: set[str]) -> None:
        now = time.time()
        if now - self._last_alloc < 5.0:
            return
        self._last_alloc = now
        self.budget.allocate(
            positions, [c.symbol for c in self._candidates[:MAX_CANDIDATES]]
        )

    def _evaluate_exits(
        self, positions: set[str], get_snapshot: Any
    ) -> list[dict[str, Any]]:
        """Check brain exit signals for open positions."""
        exits: list[dict[str, Any]] = []
        for ticker in positions:
            snap = get_snapshot(ticker)
            if snap is None or snap.get("last", 0) <= 0:
                continue
            sig = self.brain.check_exit(ticker, snap["last"], direction=1)
            if sig.should_exit:
                exits.append(
                    {"ticker": ticker, "reason": sig.reason, "exit_type": sig.exit_type}
                )
                log.info("EXIT SIGNAL %s: %s", ticker, sig.reason)
        return exits

    def _evaluate_entries(
        self, positions: set[str], get_snapshot: Any
    ) -> list[dict[str, Any]]:
        """Evaluate all tracked tickers for entry decisions."""
        decisions: list[dict[str, Any]] = []
        for ticker in self.budget.get_all_tracked() | positions:
            snap = get_snapshot(ticker)
            if snap is None:
                continue
            dec = self._eval_one(ticker, snap, len(positions))
            if dec is not None:
                decisions.append(dec)
        return decisions

    def _compute_alpha(self, snap: dict[str, Any]) -> dict[str, float]:
        """Compute all indicators from snapshot arrays."""
        alpha = compute_all_alpha(
            close=snap.get("close_arr"),
            high=snap.get("high_arr"),
            low=snap.get("low_arr"),
            volume=snap.get("vol_arr"),
            buy_volume=snap.get("buy_vol_arr"),
            bid_sizes=snap.get("bid_sizes"),
            ask_sizes=snap.get("ask_sizes"),
        )
        if not alpha:
            alpha = (
                compute_alpha(
                    close=snap.get("close_arr"),
                    volume=snap.get("vol_arr"),
                    buy_volume=snap.get("buy_vol_arr"),
                    bid_sizes=snap.get("bid_sizes"),
                    ask_sizes=snap.get("ask_sizes"),
                )
                or {}
            )
        return alpha

    def _build_decision(
        self, ticker: str, direction: int, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Build decision dict from brain result."""
        score = result.get("score", 0)
        verdict = result.get("verdict", "")
        conf = result.get("confidence", 0.5)
        log.info(
            "THINK %s %s score=%.3f regime=%s",
            ticker,
            "BUY" if direction > 0 else "SELL",
            score,
            result.get("regime", "?"),
        )
        thought = type(
            "T",
            (),
            {
                "direction": direction,
                "score": score,
                "verdict": verdict,
                "confidence": conf,
            },
        )()
        return {
            "ticker": ticker,
            "direction": direction,
            "verdict": verdict,
            "score": score,
            "thought": thought,
            "sizing": result.get("sizing"),
        }

    def _eval_one(
        self, ticker: str, snap: dict[str, Any], open_count: int
    ) -> dict[str, Any] | None:
        """Evaluate one ticker through the full brain pipeline."""
        prices = snap.get("prices", [])
        if len(prices) < 20:
            return None
        result = self.brain.tick(
            alpha=self._compute_alpha(snap),
            ticker=ticker,
            entry_price=float(prices[-1]),
            atr=snap.get("atr", 1.0),
            open_positions=open_count,
        )
        direction = result.get("direction", 0)
        return (
            self._build_decision(ticker, direction, result) if direction != 0 else None
        )

    def on_trade_close(
        self, ticker: str, won: bool, pnl_pct: float, direction: int = 1
    ) -> None:
        """Route trade close to System 2 reflection."""
        self.slow_cortex.on_trade_close(ticker, won, pnl_pct, direction)
        self.brain.on_trade_close(ticker, won, pnl_pct, direction)
