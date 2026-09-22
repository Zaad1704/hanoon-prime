"""brain.mtf — Multi-Timeframe DNN feature math (single source of truth).

Offline trainer (``scripts/train_meta_dnn.py``) and the live snapshot
generator (``ib_cycle.py``) MUST compute OBI / VPIN / tf5 / tf15 features
through THESE functions — any divergence silently corrupts P(Win) at
inference.  Low-level resampling and per-bar signals live in
``brain.mtf_resample``; this module adds the composite trend/volatility
features and re-exports the full public API.

Cold / insufficient-history degradation (identical offline + live):
  * obi / vpin -> 0.0 (neutral)   * tf5_align -> 0.0 (neutral)
  * tf15_vol   -> 1.0 (neutral)
Warm-up (1-min bars): tf5 = (EMA21 + 2 buckets) x 5 = 115;
tf15 = (ATR14 + SMA20 = 34 buckets) x 15 = 510.
"""

from __future__ import annotations

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
TF5_EMA_FAST: int = 9
TF5_EMA_SLOW: int = 21
TF5_ATR_PERIOD: int = 14
TF15_ATR_PERIOD: int = 14
TF15_SMA_PERIOD: int = 20
MTF_MIN_BARS: int = (TF15_ATR_PERIOD + TF15_SMA_PERIOD) * MTF15_BUCKET + 30


# ── Trend alignment (5-min) ────────────────────────────────────────────────


def compute_tf5_trend_alignment(
    close: list[float] | np.ndarray,
    high: list[float] | np.ndarray,
    low: list[float] | np.ndarray,
    end: int | None = None,
) -> float:
    """5-min EMA-fan alignment in [-1, 1]: (EMA21 - EMA9) / ATR14.

    0.0 (neutral) when fewer than (TF5_EMA_SLOW + 2) buckets are available.
    """
    bc, bh, bl, _ = bucket_bars(close, high, low, close, bucket=MTF5_BUCKET, end=end)
    if len(bc) < TF5_EMA_SLOW + 2:
        return 0.0
    fast = _ema(bc, TF5_EMA_FAST)
    slow = _ema(bc, TF5_EMA_SLOW)
    atr = _bucket_atr(bc, bh, bl, TF5_ATR_PERIOD)
    if atr <= 0.0:
        atr = 1e-12
    return float(np.clip((fast[-1] - slow[-1]) / atr, -1.0, 1.0))


# ── Volatility expansion (15-min) ──────────────────────────────────────────


def compute_tf15_vol_expansion(
    close: list[float] | np.ndarray,
    high: list[float] | np.ndarray,
    low: list[float] | np.ndarray,
    end: int | None = None,
) -> float:
    """15-min volatility expansion: ATR15 / SMA20(ATR15), clipped [0.5, 3.0].

    1.0 (neutral) when fewer than (ATR_period + SMA_period) buckets exist.
    """
    bc, bh, bl, _ = bucket_bars(close, high, low, close, bucket=MTF15_BUCKET, end=end)
    if len(bc) < TF15_ATR_PERIOD + TF15_SMA_PERIOD:
        return 1.0
    atrs = _bucket_atr_series(bc, bh, bl, TF15_ATR_PERIOD)
    last = atrs[-1]
    base = float(np.mean(atrs[-TF15_SMA_PERIOD:])) + 1e-12
    ratio = last / base
    return float(np.clip(ratio, 0.5, 3.0))


# ── Public API (re-exports everything) ─────────────────────────────────────

__all__ = [
    "MTF5_BUCKET",
    "MTF15_BUCKET",
    "TF5_EMA_FAST",
    "TF5_EMA_SLOW",
    "TF5_ATR_PERIOD",
    "TF15_ATR_PERIOD",
    "TF15_SMA_PERIOD",
    "MTF_MIN_BARS",
    "compute_obi",
    "compute_vpin",
    "compute_tf5_trend_alignment",
    "compute_tf15_vol_expansion",
]
