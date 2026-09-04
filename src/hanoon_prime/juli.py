"""hanoon_prime.juli — Biological brain orchestrator.
Scanner discovery + cognitive pipeline; returns (decisions, exit_signals).
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
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
        log.info("System 1 (JuliBrain) fast-path initialized")
        self.slow_cortex.start()

    def tick(
        self,
        positions: set[str],
        get_snapshot: Any,
        streamer: Any,
        closing: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """One full brain cycle. Returns (entry_decisions, exit_signals)."""
        self._open_positions = dict.fromkeys(positions, True)
        self._maybe_scan()
        self._collect_scanner()
        self._maybe_screen(get_snapshot)
        self._maybe_allocate(positions)
        exits = self._evaluate_exits(positions, get_snapshot, closing or set())
        entries = self._evaluate_entries(positions, get_snapshot)
        return exits, entries

    def _maybe_scan(self) -> None:
        """Start a new scanner subscription every 5 minutes."""
        if not self.scanner.should_scan():
            return
        try:
            self.scanner.scan("most_active")
        except Exception as e:
            log.warning("Scan start failed: %s", e)

    def _collect_scanner(self) -> None:
        """Poll scanner results every cycle — IB returns them async."""
        try:
            results = self.scanner.collect()
            if results:
                self._candidates = results
        except Exception as e:
            log.debug("Scan collect error: %s", e)

    def _maybe_screen(self, get_snapshot: Any) -> None:
        """Screen candidates through snapshots."""
        if not self._candidates:
            return
        n = sum(1 for c in self._candidates[:MAX_CANDIDATES]
                if (snap := get_snapshot(c.symbol)) and snap.get("last", 0) > 0)
        log.info("SCREEN: %d/%d passed", n, len(self._candidates))

    def _maybe_allocate(self, positions: set[str]) -> None:
        """Allocate data budget periodically."""
        if time.time() - self._last_alloc < 5.0:
            return
        self._last_alloc = time.time()
        syms = [c.symbol for c in self._candidates[:MAX_CANDIDATES]]
        self.budget.allocate(positions, syms)

    def _evaluate_exits(self, positions: set[str], get_snapshot: Any,
                        closing: set[str]) -> list[dict[str, Any]]:
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

    def _evaluate_entries(self, positions: set[str],
                         get_snapshot: Any) -> list[dict[str, Any]]:
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

    def _compute_alpha(self, snap: dict[str, Any]) -> dict[str, float]:
        """Compute all indicators from snapshot arrays."""
        kw = {
            k: snap.get(v)
            for k, v in [
                ("close", "close_arr"),
                ("high", "high_arr"),
                ("low", "low_arr"),
                ("volume", "vol_arr"),
                ("buy_volume", "buy_vol_arr"),
                ("bid_sizes", "bid_sizes"),
                ("ask_sizes", "ask_sizes"),
            ]
        }
        alpha = compute_all_alpha(**kw)
        return alpha if alpha else (compute_alpha(**kw) or {})

    def _build_decision(self, ticker: str, direction: int,
                        result: dict[str, Any]) -> dict[str, Any]:
        """Build decision dict from brain result."""
        score = result.get("score", 0)
        verdict = result.get("verdict", "")
        conf = result.get("confidence", 0.5)
        log.info("THINK %s %s score=%.3f regime=%s risk=%s",
                 ticker, "BUY" if direction > 0 else "SELL", score,
                 result.get("regime", "?"), result.get("risk", "normal"))
        thought = SimpleNamespace(direction=direction, score=score,
                                 verdict=verdict, confidence=conf)
        return {"ticker": ticker, "direction": direction, "verdict": verdict,
                "score": score, "thought": thought,
                "sizing": result.get("sizing")}

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
        """Route trade close to System 2 reflection."""
        self.slow_cortex.on_trade_close(ticker, won, pnl_pct, direction)
        self.brain.on_trade_close(ticker, won, pnl_pct, direction)
