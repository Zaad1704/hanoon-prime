"""brain.exit_ladder — 3-tier exit coordination (JULI is primary).

A thin, non-breaking wrapper around the existing :class:`ExitPolicy`
(mechanical TIER3). Faithful Prime translation of rebuild's
``brackets.should_exit`` hierarchy, but disciplined to R1/R13: JULI's
verdict is a *numeric* exit-likelihood compared against learned
thresholds — never a string verdict comparison. BUILD_NAME "Aegis".

Hierarchy (first decisive tier wins):
  1. TIER1 HARD STOP  — ``force_exit`` or a stop-price breach (absolute,
     never overridden).
  2. TIER2 JULI VERDICT — ``exit_likelihood`` vs ``AdaptiveThresholds``
     threshold (exit / watch-ride-winners / hold), driven by the learned
     win-rate band. Off-by-default **hysteresis**: a soft TIER2 exit must
     persist HYSTERESIS_BARS consecutive evaluations before confirming, so
     a one-bar indicator spike can't kill a fresh (esp. probe) entry.
  3. TIER3 MECHANICAL — :class:`ExitPolicy` (profit_lock, giveback, stale,
     consolidation). Only consulted when TIER2 yields hold.

Dormant by default: with ``stop_price=None`` / ``force_exit=False``, TIER1
is a no-op; when ``exit_likelihood`` is not passed (0.0) it is derived from
the policy's ``exit_likelihood`` pillar signal, and ``win_rate`` defaults to
0.5 unless a ``win_rate_provider`` supplies the realized win rate. Behavior
is otherwise identical to the current mechanical ExitPolicy — the live
caller in ``juli.py`` needs no change.

Per R1, BUY/SELL/HOLD verdicts live only in cortex; here we use lowercase
exit_type labels (exit/watch/hold + the mechanical types) and never compare
against verdict tokens (R13).
"""

from __future__ import annotations

from typing import Callable, Optional

from ..immune import HYSTERESIS_BARS, HYSTERESIS_EXIT_ENABLED
from .adaptive_thresholds import AdaptiveThresholds, get_adaptive_thresholds
from .exit_checks import ExitSignal
from .exits import ExitPolicy


class ExitLadder:
    """Coordinates the TIER1/TIER2/TIER3 exit decision (non-breaking)."""

    def __init__(
        self,
        policy: ExitPolicy,
        thresholds: Optional[AdaptiveThresholds] = None,
        win_rate_provider: Optional[Callable[[], tuple[float, int]]] = None,
    ) -> None:
        self._policy = policy
        self._thresholds = thresholds or get_adaptive_thresholds()
        self._win_rate_provider = win_rate_provider
        # Per-ticker soft-exit confirmation streak (hysteresis gate).
        self._exit_streak: dict[str, int] = {}

    def evaluate(
        self,
        ticker: str,
        current_price: float,
        ib_pnl: float = 0.0,
        direction: int = 1,
        *,
        exit_likelihood: float = 0.0,
        win_rate: float = 0.5,
        stop_price: Optional[float] = None,
        force_exit: bool = False,
    ) -> ExitSignal:
        """Run TIER1→TIER2→TIER3; the first decisive tier wins. Unset
        exit_likelihood derives from the policy pillar signal and the 0.5
        default win_rate is overridden by the win-rate provider.
        """
        exit_likelihood, win_rate = self._resolve_defaults(
            ticker, current_price, direction, exit_likelihood, win_rate
        )
        tier1 = self._tier1(current_price, stop_price, force_exit, direction)
        if tier1.should_exit:
            return tier1
        tier2 = self._tier2(exit_likelihood, win_rate, ib_pnl)
        confirmed = self._hysteresis_confirm(ticker, tier2.should_exit)
        if tier2.should_exit:
            # TIER1 already returned above; TIER2 is a SOFT exit — confirm it
            # persisted HYSTERESIS_BARS bars (hard override never applies
            # here), else suppress to hold so a one-bar spike can't kill the
            # position (esp. a death-sprial PROBE entry).
            if confirmed:
                return tier2
            return ExitSignal(False, "hysteresis (pending)", "hold")
        if tier2.exit_type == "watch":
            return tier2
        return self._tier3(ticker, current_price, ib_pnl, direction)

    def _resolve_defaults(
        self,
        ticker: str,
        current_price: float,
        direction: int,
        exit_likelihood: float,
        win_rate: float,
    ) -> tuple[float, float]:
        """Derive TIER2 inputs not supplied by the caller."""
        if exit_likelihood <= 0.0:
            derive = getattr(self._policy, "exit_likelihood", None)
            exit_likelihood = (
                float(derive(ticker, current_price, direction)) if derive else 0.0
            )
        if win_rate == 0.5 and self._win_rate_provider is not None:
            derived, n = self._win_rate_provider()
            if n > 0:
                win_rate = derived
        return exit_likelihood, win_rate

    def _hysteresis_confirm(self, ticker: str, fired: bool) -> bool:
        """Soft-exit persistence gate (rebuild HYSTERESIS_BARS; off by default).

        TIER1 hard stops bypass this entirely. When disabled, returns ``fired``
        so soft-exit behavior is byte-identical to prior (immediate). When
        enabled, a soft exit must fire HYSTERESIS_BARS consecutive evaluations;
        any non-firing bar resets the streak.
        """
        if not HYSTERESIS_EXIT_ENABLED:
            return fired
        if fired:
            self._exit_streak[ticker] = self._exit_streak.get(ticker, 0) + 1
            return self._exit_streak[ticker] >= HYSTERESIS_BARS
        self._exit_streak[ticker] = 0
        return False

    def _tier1(
        self,
        current_price: float,
        stop_price: Optional[float],
        force_exit: bool,
        direction: int,
    ) -> ExitSignal:
        """Emergency hard stop — absolute floor, never overridden."""
        if force_exit:
            return ExitSignal(True, "hard_stop (force)", "exit")
        if stop_price is not None and self._breached(
            current_price, stop_price, direction
        ):
            return ExitSignal(True, "hard_stop (breached)", "exit")
        return ExitSignal()

    @staticmethod
    def _breached(price: float, stop_price: float, direction: int) -> bool:
        """Did price cross the stop? Long exits at ``<=``, short at ``>=``."""
        return price <= stop_price if direction > 0 else price >= stop_price

    def _tier2(
        self, exit_likelihood: float, win_rate: float, ib_pnl: float
    ) -> ExitSignal:
        """JULI verdict: exit / watch-ride-winners / hold (numeric, not string)."""
        thr = self._thresholds.get_exit_threshold(win_rate)
        watch = self._thresholds.get_watch_threshold(win_rate)
        ride_pnl, ride_lik = self._thresholds.get_ride_winners_params()
        if ib_pnl > ride_pnl and exit_likelihood < ride_lik:
            return ExitSignal(False, "ride_winners (watch)", "watch")
        if exit_likelihood >= thr:
            return ExitSignal(True, "juli_verdict (exit)", "exit")
        if exit_likelihood >= watch:
            return ExitSignal(False, "juli_watch", "watch")
        return ExitSignal(False, "", "hold")

    def _tier3(
        self, ticker: str, current_price: float, ib_pnl: float, direction: int
    ) -> ExitSignal:
        """Mechanical safety nets (the existing ExitPolicy)."""
        return self._policy.evaluate(ticker, current_price, ib_pnl, direction)


__all__ = ["ExitLadder"]
