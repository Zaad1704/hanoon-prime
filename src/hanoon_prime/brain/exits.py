"""brain.exits — exit intelligence for open positions.

JULI's exit decision layer: profit-lock tiers, consolidation exit,
alpha delta tracking. Works alongside the mechanical ATR trailing
in ib_executor.py. Trigger math lives in ``exit_checks.py``; this
module owns per-position state and the learned thresholds.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .config import (
    EXIT_ADAPT_GIVEBACK_MAX,
    EXIT_ADAPT_MIN_TRADES,
    EXIT_ADAPT_STALE_MAX,
    EXIT_ADAPT_STEP_SCALE,
    GIVEBACK_KEEP_RATIO,
    STALE_EXIT_MINUTES,
)
from .exit_checks import ExitSignal, check_consolidation
from .exit_checks import check_giveback as _giveback
from .exit_checks import check_profit_lock as _profit_lock
from .exit_checks import check_stale as _stale

if TYPE_CHECKING:
    from .realized_ev import RealizedStats


class ExitPolicy:
    """JULI's exit intelligence — decides when to close positions.

    v2.1: thresholds are LEARNED. ``adapt_from_realized`` shifts the
    giveback keep-ratio and the stale window from realized exit outcomes
    (many giveback exits that would have hit target → widen keep-ratio;
    many stale exits → shorten the stale window). Bounded by
    EXIT_ADAPT_* so the policy adapts without reinventing itself.
    """

    def __init__(self) -> None:
        """Initialize empty per-position state and base thresholds."""
        self._entry_price: dict[str, float] = {}
        self._peak_price: dict[str, float] = {}
        self._peak_pnl: dict[str, float] = {}
        self._entry_ts: dict[str, float] = {}
        self._flat_pulses: dict[str, int] = {}
        self._prev_price: dict[str, float] = {}
        self._entry_alpha: dict[str, dict[str, float]] = {}
        self._keep_ratio: float = GIVEBACK_KEEP_RATIO
        self._stale_minutes: float = STALE_EXIT_MINUTES
        self._adapt_count: int = 0

    def adapt_from_realized(self, realized: "RealizedStats") -> None:
        """Retune giveback/stale thresholds from realized exit outcomes.

        Signal extraction (bounded, no veto semantics — exits only):
        • When realized R:R is weak relative to the 3:1 target while
          giveback exits fire, winners are being cut — WIDEN keep-ratio
          so trailing protects more of the peak.
        • When realized R:R is strong, tighten slightly toward base.
        • Stale exits shrink when R:R underperforms (capital recycling).
        """
        rr, rel, n = realized.realized_rr()
        if n < EXIT_ADAPT_MIN_TRADES or rel <= 0.0:
            return
        self._adapt_count += 1
        shift = (3.0 - min(rr, 3.0)) * EXIT_ADAPT_STEP_SCALE
        self._keep_ratio = max(
            GIVEBACK_KEEP_RATIO - EXIT_ADAPT_GIVEBACK_MAX,
            min(
                GIVEBACK_KEEP_RATIO + EXIT_ADAPT_GIVEBACK_MAX, self._keep_ratio + shift
            ),
        )
        stale_shift = (3.0 - min(rr, 3.0)) * 10.0
        self._stale_minutes = max(
            STALE_EXIT_MINUTES - EXIT_ADAPT_STALE_MAX,
            min(
                STALE_EXIT_MINUTES + EXIT_ADAPT_STALE_MAX,
                self._stale_minutes - stale_shift,
            ),
        )

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

    def telemetry(self) -> dict[str, float | int]:
        """Learned-exit telemetry (advisor view of the current policy)."""
        return {
            "keep_ratio": round(self._keep_ratio, 3),
            "stale_minutes": round(self._stale_minutes, 1),
            "adapt_count": self._adapt_count,
        }

    def _update_peaks(self, ticker: str, price: float, pnl: float) -> None:
        """Track the running price and P&L peaks for one position."""
        if price > self._peak_price.get(ticker, 0):
            self._peak_price[ticker] = price
        if pnl > self._peak_pnl.get(ticker, 0):
            self._peak_pnl[ticker] = pnl

    def _check_profit_lock(
        self,
        ticker: str,
        pnl: float,
        _direction: int,
        _price: float,
    ) -> ExitSignal:
        """If peak gain reached a tier, lock in minimum profit."""
        return _profit_lock(
            self._entry_price.get(ticker, 0),
            self._peak_pnl.get(ticker, 0),
            pnl,
        )

    def _check_giveback(
        self,
        ticker: str,
        pnl: float,
        _direction: int = 1,
        _price: float = 0.0,
    ) -> ExitSignal:
        """Exit when P&L drops more than the learned keep-ratio from its peak."""
        return _giveback(pnl, self._peak_pnl.get(ticker, 0.0), self._keep_ratio)

    def _check_stale(
        self,
        ticker: str,
        _pnl: float = 0.0,
        _direction: int = 1,
        _price: float = 0.0,
    ) -> ExitSignal:
        """Exit if held longer than the learned stale window."""
        return _stale(self._entry_ts.get(ticker, time.time()), self._stale_minutes)

    def _check_consolidation(
        self,
        ticker: str,
        _pnl: float = 0.0,
        _direction: int = 1,
        price: float = 0.0,
    ) -> ExitSignal:
        """Exit if price moves less than 0.1% for N consecutive pulses."""
        prev = self._prev_price.get(ticker, price)
        sig, pulses = check_consolidation(price, prev, self._flat_pulses.get(ticker, 0))
        self._flat_pulses[ticker] = pulses
        self._prev_price[ticker] = price
        return sig
