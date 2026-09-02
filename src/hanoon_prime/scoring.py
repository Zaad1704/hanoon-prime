"""hanoon_prime.scoring — score the alpha dict into [THRESHOLD_MIN, THRESHOLD_MAX].

NO inversion. NO neuromorphic blend. NO modifiers. NO percentile normalization.
Just: normalize each indicator to [0, 1] (higher = more bullish), take the
weighted average. Simple enough to reason about, correct enough to trust.
"""
from __future__ import annotations

from .constants import (
    INDICATOR_WEIGHTS,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
)


def _normalize_signed(x: float) -> float:
    """Map [-1, 1] → [0, 1] (higher = bullish)."""
    return float(max(0.0, min(1.0, (x + 1.0) / 2.0)))


def _normalize_positive(x: float) -> float:
    """Clamp [0, ∞) → [0, 1] (higher = more bullish)."""
    return float(max(0.0, min(1.0, x)))


_INDICATOR_NORMALIZERS = {
    "vpin": lambda x: _normalize_positive(x),   # uses vpin_magnitude (unsigned [0,1])
    "orderbook_imbalance": _normalize_signed,  # [-1, 1] → [0, 1]
    "institutional_flow": _normalize_positive,  # [0, 3] → clamp to [0, 1]
    "momentum": _normalize_signed,      # [-1, 1] → [0, 1]
    "vwap_deviation": _normalize_signed,  # [-1, 1] → [0, 1]
}


def compute_score(alpha: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Compute a normalized signal score in [THRESHOLD_MIN, THRESHOLD_MAX].

    Steps:
      1. Normalize each of the 5 indicators to [0, 1] (higher = bullish).
      2. Weighted average using INDICATOR_WEIGHTS (or custom learned weights).
      3. Clamp to [THRESHOLD_MIN, THRESHOLD_MAX].

    No inversion. No neuromorphic. No modifiers. Just the raw signal.
    """
    if weights is None:
        weights = INDICATOR_WEIGHTS
    total = 0.0
    weighted = 0.0
    for name, weight in weights.items():
        # VPIN is stored signed [-1,1]; scorer uses the unsigned magnitude
        raw = alpha.get(name + "_magnitude" if name == "vpin" else name, 0.0)
        normalizer = _INDICATOR_NORMALIZERS[name]
        normed = normalizer(raw)
        weighted += normed * weight
        total += weight

    if total == 0.0:
        return 0.50

    raw_score = weighted / total
    return float(max(THRESHOLD_MIN, min(THRESHOLD_MAX, raw_score)))
