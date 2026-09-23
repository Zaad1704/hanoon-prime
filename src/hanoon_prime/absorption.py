"""hanoon_prime.absorption — pure MM absorption detection (Phase 2).

When heavy one-sided flow fails to move the level, a market maker is
absorbing. Pure functions: tape metrics in, signed alpha out. Never
emits verdict strings (R1) — cortex remains the sole verdict source.
"""

from __future__ import annotations

from .brain.learning_config import (
    ABSORPTION_DELTA_THETA,
    ABSORPTION_SIGNAL_MIN,
    ABSORPTION_VOL_RATIO,
)

ABSORPTION_KEY: str = "absorption"


def _side_score(
    cvd: float,
    level_held: bool,
    vol_dom: float,
    vol_sub: float,
    sell_side: bool,
) -> float:
    """Score one absorption side in [0, 1] (0 = no setup)."""
    if not level_held:
        return 0.0
    directed = -cvd if sell_side else cvd
    if directed < ABSORPTION_DELTA_THETA:
        return 0.0
    if vol_dom < ABSORPTION_VOL_RATIO * max(vol_sub, 1.0):
        return 0.0
    delta_part = min(1.0, directed / (2.0 * ABSORPTION_DELTA_THETA))
    ratio = vol_dom / max(vol_sub, 1.0)
    vol_part = min(1.0, ratio / (2.0 * ABSORPTION_VOL_RATIO))
    return 0.5 * delta_part + 0.5 * vol_part


def detect_absorption(metrics: dict[str, float]) -> dict[str, float]:
    """Score tape metrics into the signed ``absorption`` alpha key.

    ``+1`` = sell-side absorption (MM buying the bid → bullish pressure);
    ``-1`` = buy-side absorption (MM selling the ask → bearish pressure);
    ``0`` = no active absorption. Empty/missing metrics → ``{}`` so the
    key is absent from alpha (backtest path stays byte-identical).
    """
    if not metrics:
        return {}
    cvd = float(metrics.get("cvd_fast", 0.0))
    vol_buy = float(metrics.get("vol_buy", 0.0))
    vol_sell = float(metrics.get("vol_sell", 0.0))
    bid_held = float(metrics.get("bid_held", 0.0)) > 0.5
    ask_held = float(metrics.get("ask_held", 0.0)) > 0.5
    sell_score = _side_score(cvd, bid_held, vol_sell, vol_buy, sell_side=True)
    buy_score = _side_score(cvd, ask_held, vol_buy, vol_sell, sell_side=False)
    signal = sell_score - buy_score
    if abs(signal) < ABSORPTION_SIGNAL_MIN:
        signal = 0.0
    return {ABSORPTION_KEY: max(-1.0, min(1.0, signal))}


def is_absorption_active(alpha: dict[str, float], floor: float) -> bool:
    """True when ``alpha`` carries an absorption signal at/above ``floor``."""
    return abs(float(alpha.get(ABSORPTION_KEY, 0.0))) >= floor


__all__ = ["ABSORPTION_KEY", "detect_absorption", "is_absorption_active"]
