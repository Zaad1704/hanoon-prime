"""hanoon_prime.brain.exits — exit intelligence for open positions.

JULI's exit decision layer: profit-lock tiers, consolidation exit,
alpha delta tracking. Works alongside the mechanical ATR trailing
in ib_executor.py.

Merge of rebuild's learned_exit.py + profit_lock + consolidation exit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import (
    CONSOLIDATION_PULSES,
    GIVEBACK_KEEP_RATIO,
    PROFIT_LOCK_TIERS,
    STALE_EXIT_MINUTES,
)


@dataclass
class ExitSignal:
    """Exit decision for a position."""

    should_exit: bool = False
    reason: str = ""
    exit_type: str = "hold"


class ExitPolicy:
    """JulI's exit intelligence — decides when to close positions."""

    def __init__(self) -> None:
        self._entry_price: dict[str, float] = {}
        self._peak_price: dict[str, float] = {}
        self._peak_pnl: dict[str, float] = {}
        self._entry_ts: dict[str, float] = {}
        self._flat_pulses: dict[str, int] = {}
        self._prev_price: dict[str, float] = {}
        self._entry_alpha: dict[str, dict[str, float]] = {}

    def register(
        self,
        ticker: str,
        entry_price: float,
        entry_alpha: dict[str, float] | None = None,
    ) -> None:
        """Register a new position for exit monitoring."""
        self._entry_price[ticker] = entry_price
        self._peak_price[ticker] = entry_price
        self._peak_pnl[ticker] = 0.0
        self._entry_ts[ticker] = time.time()
        self._flat_pulses[ticker] = 0
        self._prev_price[ticker] = entry_price
        if entry_alpha:
            self._entry_alpha[ticker] = dict(entry_alpha)

    def evaluate(
        self,
        ticker: str,
        current_price: float,
        ib_unrealized_pnl: float = 0.0,
        direction: int = 1,
    ) -> ExitSignal:
        """Evaluate exit conditions for one position."""
        if ticker not in self._entry_ts:
            return ExitSignal()
        self._update_peaks(ticker, current_price, ib_unrealized_pnl)
        for check in (
            self._check_profit_lock,
            self._check_giveback,
            self._check_stale,
            self._check_consolidation,
        ):
            sig = check(ticker, ib_unrealized_pnl, direction, current_price)
            if sig.should_exit:
                return sig
        return ExitSignal()

    def deregister(self, ticker: str) -> None:
        """Remove a position from all exit-monitoring state."""
        for d in (
            self._entry_price,
            self._peak_price,
            self._peak_pnl,
            self._entry_ts,
            self._flat_pulses,
            self._prev_price,
            self._entry_alpha,
        ):
            d.pop(ticker, None)

    def _update_peaks(self, ticker: str, price: float, pnl: float) -> None:
        if price > self._peak_price.get(ticker, 0):
            self._peak_price[ticker] = price
        if pnl > self._peak_pnl.get(ticker, 0):
            self._peak_pnl[ticker] = pnl

    def _check_profit_lock(
        self,
        ticker: str,
        pnl: float,
        direction: int,
        price: float,
    ) -> ExitSignal:
        """If peak gain reached a tier, lock in minimum profit."""
        entry = self._entry_price.get(ticker, 0)
        peak = self._peak_pnl.get(ticker, 0)
        if entry <= 0 or peak <= 0:
            return ExitSignal()
        peak_gain_pct = abs(peak) / entry
        current_gain_pct = abs(pnl) / entry if pnl > 0 else 0.0
        for min_peak, min_lock in PROFIT_LOCK_TIERS:
            if peak_gain_pct >= min_peak and current_gain_pct < min_lock:
                return ExitSignal(
                    True,
                    f"profit_lock peak={peak_gain_pct:.1%} cur={current_gain_pct:.1%}",
                    "profit_lock",
                )
        return ExitSignal()

    def _check_giveback(
        self,
        ticker: str,
        pnl: float,
        direction: int = 1,
        price: float = 0.0,
    ) -> ExitSignal:
        """Exit if unrealized P&L drops more than 45% from its peak."""
        peak = self._peak_pnl.get(ticker, 0.0)
        if peak <= 0:
            return ExitSignal()
        giveback = 1.0 - (pnl / peak if peak != 0 else 0)
        if giveback > GIVEBACK_KEEP_RATIO:
            return ExitSignal(True, f"giveback {giveback:.0%}", "giveback")
        return ExitSignal()

    def _check_stale(
        self,
        ticker: str,
        pnl: float = 0.0,
        direction: int = 1,
        price: float = 0.0,
    ) -> ExitSignal:
        """Exit if held too long without significant progress."""
        entry = self._entry_ts.get(ticker, time.time())
        hold_min = (time.time() - entry) / 60.0
        if hold_min > STALE_EXIT_MINUTES:
            return ExitSignal(True, f"stale {hold_min:.0f}min", "stale")
        return ExitSignal()

    def _check_consolidation(
        self,
        ticker: str,
        pnl: float = 0.0,
        direction: int = 1,
        price: float = 0.0,
    ) -> ExitSignal:
        """Exit if price moves less than 0.1% for N consecutive pulses."""
        prev = self._prev_price.get(ticker, price)
        threshold = abs(prev) * 0.001 if prev else 0.01
        if abs(price - prev) < threshold:
            self._flat_pulses[ticker] = self._flat_pulses.get(ticker, 0) + 1
        else:
            self._flat_pulses[ticker] = 0
        self._prev_price[ticker] = price
        if self._flat_pulses.get(ticker, 0) >= CONSOLIDATION_PULSES:
            return ExitSignal(
                True, f"consolidation {self._flat_pulses[ticker]}p", "consolidation"
            )
        return ExitSignal()
