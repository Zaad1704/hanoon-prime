"""brain.exit_checks — Stateless exit trigger computations.

Pure per-tick exit checks extracted from ``ExitPolicy`` so the policy
coordinator and the trigger math each stay under the 200-line R3b limit.
All functions are side-effect free: per-ticker state is passed in and,
for consolidation, the updated pulse count is returned alongside the
signal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import CONSOLIDATION_PULSES, PROFIT_LOCK_TIERS


@dataclass
class ExitSignal:
    """Exit decision for a position."""

    should_exit: bool = False
    reason: str = ""
    exit_type: str = "hold"
    exit_score: float = 0.0  # Pillar combination score (0-1) when should_exit=True


def check_profit_lock(entry: float, peak_pnl: float, pnl: float) -> ExitSignal:
    """If peak gain reached a tier, lock in minimum profit."""
    if entry <= 0 or peak_pnl <= 0:
        return ExitSignal()
    peak_gain_pct = abs(peak_pnl) / entry
    current_gain_pct = abs(pnl) / entry if pnl > 0 else 0.0
    for min_peak, min_lock in PROFIT_LOCK_TIERS:
        if peak_gain_pct >= min_peak and current_gain_pct < min_lock:
            return ExitSignal(
                True,
                f"profit_lock peak={peak_gain_pct:.1%} cur={current_gain_pct:.1%}",
                "profit_lock",
            )
    return ExitSignal()


def check_giveback(pnl: float, peak_pnl: float, keep_ratio: float) -> ExitSignal:
    """Exit when P&L drops more than the learned keep-ratio from its peak."""
    if peak_pnl <= 0:
        return ExitSignal()
    giveback = 1.0 - (pnl / peak_pnl if peak_pnl != 0 else 0)
    if giveback > keep_ratio:
        return ExitSignal(True, f"giveback {giveback:.0%}", "giveback")
    return ExitSignal()


def check_stale(entry_ts: float, stale_minutes: float) -> ExitSignal:
    """Exit if held longer than the learned stale window."""
    hold_min = (time.time() - entry_ts) / 60.0
    if hold_min > stale_minutes:
        return ExitSignal(True, f"stale {hold_min:.0f}min", "stale")
    return ExitSignal()


def check_consolidation(
    price: float, prev_price: float, pulses: int
) -> tuple[ExitSignal, int]:
    """Exit if price moves less than 0.1% for N consecutive pulses.

    Returns the signal and the updated flat-pulse count (the caller
    persists it per ticker).
    """
    threshold = abs(prev_price) * 0.001 if prev_price else 0.01
    if abs(price - prev_price) < threshold:
        pulses += 1
    else:
        pulses = 0
    if pulses >= CONSOLIDATION_PULSES:
        return ExitSignal(True, f"consolidation {pulses}p", "consolidation"), pulses
    return ExitSignal(), pulses
