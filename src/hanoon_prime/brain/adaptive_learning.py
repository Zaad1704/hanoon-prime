"""brain.adaptive_learning — the nine exit-threshold update rules.

Pure functions. They receive an ``adjust`` callable (bound to
``AdaptiveThresholds._adjust``) and the live ``tiers`` list, so the
learning logic is decoupled from the state container and the two can
evolve independently. Imports only from ``threshold_limits`` (constants),
never from ``adaptive_thresholds`` → no circular import.

Mirrors rebuild ``AdaptiveThresholds.update_from_outcome`` rules 1-8,
one rule per tiny function (R3: functions ≤ 40 lines, nesting ≤ 3).
"""

from __future__ import annotations

import logging
from typing import Callable, List

from .threshold_limits import LR_DOWN, LR_UP

log = logging.getLogger(__name__)

Adjust = Callable[[str, float], None]


def learn_stale(adj: Adjust, won: bool, reason: str) -> None:
    """Stale exits: tighten on winners, loosen on losers."""
    if "stale" not in reason:
        return
    adj("stale_minutes_force", -LR_DOWN * 2 if won else LR_UP)
    if won:
        adj("stale_loss_pct_force", -LR_DOWN)


def learn_consolidation(adj: Adjust, won: bool, reason: str) -> None:
    """Consolidation exits: tighten if good, loosen if bad."""
    if "consolidation" in reason:
        adj("consolidation_pulses", -LR_DOWN if won else LR_UP)


def learn_trailing(adj: Adjust, won: bool, reason: str) -> None:
    """Trailing stops: ride more on winners, tighten on losers."""
    if "trailing_stop" in reason:
        adj("trail_keep_ratio", LR_UP if won else -LR_DOWN)


def learn_profit_lock(
    tiers: List[List[float]], won: bool, peak_pct: float, pnl_pct: float, reason: str
) -> None:
    """Profit-lock: reinforce the tier that fired (grow the locked floor)."""
    if "profit_lock" not in reason:
        return
    for tier in tiers:
        if peak_pct >= tier[0] and pnl_pct < tier[1] and won:
            tier[1] = min(tier[1] * 1.05, tier[0] * 0.8)
            return


def learn_giveback(adj: Adjust, won: bool, reason: str) -> None:
    """Giveback on a loss → trailing keep-ratio was too loose."""
    if "giveback" in reason and not won:
        adj("trail_keep_ratio", -LR_DOWN)


def learn_flat_timeout(adj: Adjust, won: bool, reason: str) -> None:
    """Flat-timeout: too aggressive on winners, too slow on losers."""
    if "flat_timeout" in reason:
        adj("flat_timeout_minutes", LR_UP * 2 if won else -LR_DOWN)


def learn_atr_stop(adj: Adjust, won: bool, pnl_pct: float, reason: str) -> None:
    """ATR stop: too wide (big loss) tighten; too tight (win stopped) widen."""
    if "stop" not in reason:
        return
    if not won and abs(pnl_pct) > 0.03:
        adj("atr_stop_mult", -LR_DOWN)
    elif won:
        adj("atr_stop_mult", LR_UP)


def learn_max_risk(adj: Adjust, won: bool, pnl_pct: float, reason: str) -> None:
    """Max-risk: a clobbered trade tightens the per-trade risk cap."""
    if not won and abs(pnl_pct) > 0.05:
        adj("max_risk_per_trade", -LR_DOWN)
    elif won and abs(pnl_pct) > 0.03:
        adj("max_risk_per_trade", LR_UP * 0.5)


def apply_learning(
    adj: Adjust,
    tiers: List[List[float]],
    *,
    won: bool,
    hold_minutes: float,
    pnl_pct: float,
    peak_pct: float,
    exit_reason: str,
    exit_likelihood: float,
) -> None:
    """Run all nine learning rules for one closed trade."""
    learn_stale(adj, won, exit_reason)
    learn_consolidation(adj, won, exit_reason)
    learn_trailing(adj, won, exit_reason)
    learn_profit_lock(tiers, won, peak_pct, pnl_pct, exit_reason)
    learn_giveback(adj, won, exit_reason)
    learn_flat_timeout(adj, won, exit_reason)
    learn_atr_stop(adj, won, pnl_pct, exit_reason)
    learn_max_risk(adj, won, pnl_pct, exit_reason)
    log.debug("adaptive learning applied: won=%s reason=%s", won, exit_reason)


__all__ = [
    "LR_UP",
    "LR_DOWN",
    "apply_learning",
    "learn_stale",
    "learn_consolidation",
    "learn_trailing",
    "learn_profit_lock",
    "learn_giveback",
    "learn_flat_timeout",
    "learn_atr_stop",
    "learn_max_risk",
]
