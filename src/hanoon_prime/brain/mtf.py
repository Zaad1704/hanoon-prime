"""brain.mtf — Multi-Timeframe DNN feature math (single source of truth).

Offline trainer (``scripts/train_meta_dnn.py``) and the live snapshot
generator (``ib_cycle.py``) MUST compute OBI / VPIN / entropy features
through THESE functions — any divergence silently corrupts P(Win) at
inference.  Low-level resampling and per-bar signals live in
``brain.mtf_resample``; this module adds the composite features and
re-exports the full public API.

Cold / insufficient-history degradation (identical offline + live):
  * obi / vpin          -> 0.0 (neutral)
  * price_entropy       -> 1.0 (max entropy = random)
  * volume_entropy      -> 1.0 (max entropy = random)
Warm-up (1-min bars): 50 bars minimum for entropy window.
"""

from __future__ import annotations

import math

import numpy as np

from .mtf_resample import (
    _bucket_atr,
    _bucket_atr_series,
    _bucket_true_ranges,
    _ema,
    _window,
    bucket_bars,
    compute_obi,
    compute_vpin,
)

# ── Constants ───────────────────────────────────────────────────────────────

MTF5_BUCKET: int = 5
MTF15_BUCKET: int = 15
ENTROPY_WINDOW: int = 50
ENTROPY_MIN_BARS: int = 60


# ── Shannon Entropy features ───────────────────────────────────────────────


def compute_price_entropy(
    close: list[float] | np.ndarray,
    window: int = ENTROPY_WINDOW,
    end: int | None = None,
) -> float:
    """Shannon entropy of price direction over trailing window.

    Classifies each bar as +1 (up), -1 (down), or 0 (flat), then computes
    H = -sum(p * log2(p)) over the window.  Returns 1.0 (max entropy)
    when fewer than 10 bars are available (neutral degradation).
    """
    arr = np.asarray(close, dtype=float).ravel()
    n = end if end is not None else len(arr)
    if n < 10:
        return 1.0
    start = max(0, n - window)
    diffs = np.diff(arr[start:n])
    if len(diffs) == 0:
        return 1.0
    up = float(np.sum(diffs > 0))
    down = float(np.sum(diffs < 0))
    flat = float(np.sum(diffs == 0))
    total = up + down + flat
    if total <= 0:
        return 1.0
    probs = []
    for count in (up, down, flat):
        if count > 0:
            p = count / total
            probs.append(p * math.log2(p))
    h = -sum(probs)
    max_h = math.log2(3)
    return float(h / max_h) if max_h > 0 else 1.0


def compute_volume_entropy(
    volume: list[float] | np.ndarray,
    window: int = ENTROPY_WINDOW,
    end: int | None = None,
) -> float:
    """Shannon entropy of volume change direction over trailing window.

    Classifies each bar's volume change as +1 (above median), -1 (below),
    then computes normalized entropy.  Returns 1.0 when insufficient data.
    """
    arr = np.asarray(volume, dtype=float).ravel()
    n = end if end is not None else len(arr)
    if n < 10:
        return 1.0
    start = max(0, n - window)
    vols = arr[start:n]
    if len(vols) < 3:
        return 1.0
    median = float(np.median(vols))
    above = float(np.sum(vols > median))
    below = float(np.sum(vols <= median))
    total = above + below
    if total <= 0:
        return 1.0
    probs = []
    for count in (above, below):
        if count > 0:
            p = count / total
            probs.append(p * math.log2(p))
    h = -sum(probs)
    max_h = math.log2(2)
    return float(h / max_h) if max_h > 0 else 1.0


# ── Public API (re-exports everything) ─────────────────────────────────────

__all__ = [
    "MTF5_BUCKET",
    "MTF15_BUCKET",
    "ENTROPY_WINDOW",
    "ENTROPY_MIN_BARS",
    "compute_obi",
    "compute_vpin",
    "compute_price_entropy",
    "compute_volume_entropy",
]
