"""brain.mtf_resample — Low-level resampling, OBI, VPIN, bucket helpers.

Row-count bucket resampling (5-min / 15-min), window buffer management,
and per-bar signal computation (OBI, VPIN) used by the higher-level
``brain.mtf`` feature functions.  Both offline trainer and live snapshot
call these through the public ``brain.mtf`` API — never directly.
"""

from __future__ import annotations

import numpy as np

# ── Window buffer ───────────────────────────────────────────────────────────

_OBI_WINDOW = 20  # trailing bars for OBI averaging
_VPIN_WINDOW = 20  # trailing bars for VPIN averaging


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


# ── OBI / VPIN ──────────────────────────────────────────────────────────────


def compute_obi(
    high: list[float] | np.ndarray,
    low: list[float] | np.ndarray,
    close: list[float] | np.ndarray,
    end: int | None = None,
) -> float:
    """Order-book imbalance proxy from OHLC close position.

    Maps each bar's close position within its [low, high] range to [-1, +1]:
        obi_bar = (2*close - low - high) / (high - low)
    Averaged over the trailing ``_OBI_WINDOW`` 1-min bars.
    Range: [-1.0, +1.0].  Returns 0.0 on insufficient data.
    """
    h = _window(np.asarray(high, dtype=np.float64), end, _OBI_WINDOW)
    l = _window(np.asarray(low, dtype=np.float64), end, _OBI_WINDOW)
    c = _window(np.asarray(close, dtype=np.float64), end, _OBI_WINDOW)
    if h is None or l is None or c is None:
        return 0.0
    rng = h - l
    safe_rng = np.where(rng > 1e-12, rng, 1e-12)
    bar_obi = (2.0 * c - l - h) / safe_rng
    return float(np.clip(np.mean(bar_obi), -1.0, 1.0))


def compute_vpin(
    volume: list[float] | np.ndarray,
    close: list[float] | np.ndarray,
    end: int | None = None,
) -> float:
    """Volume-Synchronized Probability of Informed Trading proxy.

    Estimates buy/sell volume from consecutive close direction (tick-rule
    approximation): uptick -> buy, downtick -> sell, unchanged -> split.
        VPIN = mean((buy_vol - sell_vol) / total_vol)
    Averaged over the trailing ``_VPIN_WINDOW`` 1-min bars.
    Range: [-1.0, +1.0].  Returns 0.0 on insufficient data.
    """
    v = _window(np.asarray(volume, dtype=np.float64), end, _VPIN_WINDOW)
    c = _window(np.asarray(close, dtype=np.float64), end, _VPIN_WINDOW)
    if v is None or c is None or len(c) < 2:
        return 0.0
    direction = np.sign(np.diff(c, prepend=c[0]))
    buy_vol = v * np.where(direction > 0, 1.0, np.where(direction == 0, 0.5, 0.0))
    sell_vol = v - buy_vol
    total = buy_vol + sell_vol
    safe_total = np.maximum(total, 1e-12)
    vpin = (buy_vol - sell_vol) / safe_total
    return float(np.clip(np.mean(vpin), -1.0, 1.0))


# ── Bucketing ───────────────────────────────────────────────────────────────


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
