"""brain.mtf — Multi-Timeframe DNN feature math (single source of truth).

Offline trainer (``scripts/train_meta_dnn.py``) and the live snapshot
generator (``ib_cycle.py``) MUST compute OBI / VPIN / tf5 / tf15 features
through THESE functions — any divergence silently corrupts P(Win) at
inference. All functions take the full 1-min array plus an optional ``end``
bar; live snapshots pass ``end=None`` (whole buffered history), the trainer
passes ``end=i`` at each entry bar. Row-count bucket resampling keeps
offline fixtures and the live rolling buffer byte-for-byte identical.

Cold / insufficient-history degradation (identical offline + live):
  * obi / vpin → 0.0 (neutral)   * tf5_align → 0.0 (neutral)
  * tf15_vol   → 1.0 (neutral)
Warm-up (1-min bars): tf5 = (EMA21 + 2 buckets) x 5 = 115;
tf15 = (ATR14 + SMA20 = 34 buckets) x 15 = 510.
"""

from __future__ import annotations

import numpy as np

MTF5_BUCKET: int = 5
MTF15_BUCKET: int = 15
TF5_EMA_FAST: int = 9
TF5_EMA_SLOW: int = 21
TF5_ATR_PERIOD: int = 14
TF15_ATR_PERIOD: int = 14
TF15_SMA_PERIOD: int = 20
MTF_MIN_BARS: int = (TF15_ATR_PERIOD + TF15_SMA_PERIOD) * MTF15_BUCKET + 30


def _window(
    arr: list[float] | np.ndarray,
    end: int | None,
    size: int,
) -> np.ndarray | None:
    """Trailing ``size`` elements ending at ``end`` (default: last bar)."""
    a = np.asarray(arr, dtype=np.float64)
    if end is not None:
        a = a[: min(end + 1, len(a))]
    if len(a) < size:
        return None
    return a[-size:]


def bucket_bars(
    close: list[float] | np.ndarray,
    high: list[float] | np.ndarray,
    low: list[float] | np.ndarray,
    volume: list[float] | np.ndarray,
    bucket: int = 5,
    end: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-count resample into contiguous ``bucket``-bar buckets (last partial
    bucket included — the live forming bucket, so offline/live stay aligned)."""
    c = np.asarray(close, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    if end is not None:
        c, h, l, v = c[: end + 1], h[: end + 1], l[: end + 1], v[: end + 1]
    b_close, b_high, b_low, b_vol = [], [], [], []
    n = len(c)
    for s in range(0, n, bucket):
        e = min(s + bucket, n)
        if e <= s:
            continue
        seg = slice(s, e)
        b_close.append(float(c[seg][-1]))
        b_high.append(float(h[seg].max()))
        b_low.append(float(l[seg].min()))
        b_vol.append(float(v[seg].sum()))
    return (
        np.asarray(b_close),
        np.asarray(b_high),
        np.asarray(b_low),
        np.asarray(b_vol),
    )


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Recursive EMA over a 1-D series."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for t in range(1, len(values)):
        out[t] = alpha * values[t] + (1.0 - alpha) * out[t - 1]
    return out


def _bucket_atr(
    b_close: np.ndarray, b_high: np.ndarray, b_low: np.ndarray, period: int
) -> float:
    """ATR on bucket bars (mean true range over trailing ``period`` buckets)."""
    trs = _bucket_true_ranges(b_close, b_high, b_low)
    if len(trs) < period:
        return 0.0
    return float(np.mean(trs[-period:]))


def _bucket_true_ranges(
    b_close: np.ndarray,
    b_high: np.ndarray,
    b_low: np.ndarray,
) -> np.ndarray:
    """Per-bucket true ranges (needs at least 2 buckets)."""
    n = len(b_close)
    if n < 2:
        return np.zeros(n)
    trs = np.empty(n, dtype=np.float64)
    trs[0] = b_high[0] - b_low[0]
    for j in range(1, n):
        prev_c = b_close[j - 1]
        trs[j] = max(
            b_high[j] - b_low[j],
            abs(b_high[j] - prev_c),
            abs(b_low[j] - prev_c),
        )
    return trs


def compute_tf5_trend_alignment(
    close: list[float] | np.ndarray,
    high: list[float] | np.ndarray,
    low: list[float] | np.ndarray,
    end: int | None = None,
) -> float:
    """5-min EMA-fan alignment in [-1, 1]: (EMA21 − EMA9) / ATR14.

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


def _bucket_atr_series(
    b_close: np.ndarray, b_high: np.ndarray, b_low: np.ndarray, period: int
) -> np.ndarray:
    """ATR at each of the trailing buckets (rolling mean over ``period``)."""
    trs = _bucket_true_ranges(b_close, b_high, b_low)
    out = np.empty(len(trs), dtype=np.float64)
    for j in range(len(trs)):
        lo = max(0, j - period + 1)
        out[j] = float(np.mean(trs[lo : j + 1]))
    return out


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
