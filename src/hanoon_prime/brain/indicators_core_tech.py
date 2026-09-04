"""hanoon_prime.brain.indicators_core_tech — flow/structure indicators."""

from __future__ import annotations

from typing import Any

import numpy as np

from .indicators_core import _cl


def compute_ad_signal(close: Any, high: Any, low: Any, volume: Any) -> float:
    """A/D line slope [-1, 1]."""
    c, h, l, v = (np.asarray(x, dtype=float) for x in (close, high, low, volume))
    n = min(len(c), len(h), len(l), len(v))
    if n < 10:
        return 0.0
    c, h, l, v = c[-n:], h[-n:], l[-n:], v[-n:]
    rng = np.where(h - l == 0, 1e-10, h - l)
    ad = np.cumsum(((c - l) - (h - c)) / rng * v)
    return _cl((ad[-1] - ad[-10]) / (float(np.sum(v[-10:])) + 1e-12))


def compute_obv_divergence(close: Any, volume: Any) -> float:
    """OBV vs price correlation [-1, 1]."""
    c, v = np.asarray(close, dtype=float), np.asarray(volume, dtype=float)
    n = min(len(c), len(v))
    if n < 10:
        return 0.0
    c, v = c[-n:], v[-n:]
    obv = np.concatenate([[0.0], np.cumsum(np.sign(np.diff(c)) * v[1:])])
    return (
        _cl(float(np.corrcoef(obv, c)[0, 1]))
        if np.std(obv) > 1e-12 and np.std(c) > 1e-12
        else 0.0
    )


def compute_spread_tightness(high: Any, low: Any) -> float:
    """Intrabar range tightness [0, 1]."""
    h, l = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    n = min(len(h), len(l))
    if n < 5:
        return 0.5
    spreads = h[-n:] - l[-n:]
    avg = float(np.mean(spreads))
    return (
        _cl(1.0 - float(np.std(spreads)) / (avg + 1e-12), 0.0, 1.0) if avg > 0 else 0.5
    )


def compute_volume_profile(close: Any, volume: Any) -> float:
    """POC proximity [0, 1]."""
    c, v = np.asarray(close, dtype=float), np.asarray(volume, dtype=float)
    n = min(len(c), len(v))
    if n < 10 or np.sum(v[-n:]) <= 0:
        return 0.5
    c, v = c[-n:], v[-n:]
    return _cl(
        1.0
        - abs(c[-1] - float(np.sum(c * v) / np.sum(v)))
        / (float(np.std(c)) + 1e-12)
        / 3.0,
        0.0,
        1.0,
    )


def compute_trade_intensity(volume: Any, window: int = 20) -> float:
    """Volume burst detection [0, 1]."""
    v = np.asarray(volume, dtype=float)
    return (
        0.5
        if len(v) < window
        else _cl(v[-1] / (np.mean(v[-window:]) + 1e-12) / 3.0, 0.0, 1.0)
    )


def compute_mean_reversion(close: Any, window: int = 20) -> float:
    """Probability of mean reversion [0, 1]."""
    c = np.asarray(close, dtype=float)
    if len(c) < window:
        return 0.5
    std = float(np.std(c[-window:]))
    exp_val = 1.0 - float(
        np.exp(-abs(c[-1] - float(np.mean(c[-window:]))) / max(std, 1e-12) / 2.0)
    )
    return _cl(exp_val, 0.0, 1.0) if std > 0 else 0.5


def compute_trend_strength(close: Any, window: int = 20) -> float:
    """Linear regression slope magnitude [0, 1]."""
    c = np.asarray(close, dtype=float)
    if len(c) < window:
        return 0.0
    w = c[-window:]
    slope = float(np.polyfit(np.arange(window, dtype=float), w, 1)[0])
    return _cl(abs(slope) / (float(np.std(w)) + 1e-12) * 5.0, 0.0, 1.0)


def compute_sr_proximity(close: Any, high: Any, low: Any, window: int = 20) -> float:
    """Proximity to nearest S/R level [0, 1]."""
    c, h, l = (np.asarray(x, dtype=float) for x in (close, high, low))
    n = min(len(c), len(h), len(l))
    if n < window:
        return 0.5
    c, h, l = c[-n:], h[-n:], l[-n:]
    hi, lo = float(np.max(h[-window:])), float(np.min(l[-window:]))
    rng = hi - lo
    return (
        _cl(1.0 - min((hi - c[-1]) / rng, (c[-1] - lo) / rng), 0.0, 1.0)
        if rng > 0
        else 0.5
    )


def compute_elliott_wave(close: Any, window: int = 30) -> float:
    """Impulse vs corrective balance [-1, 1]."""
    c = np.asarray(close, dtype=float)
    if len(c) < window:
        return 0.0
    r = np.diff(c[-window:])
    up = float(np.mean(r[r > 0])) if len(r[r > 0]) > 0 else 0.0
    dn = float(np.mean(-r[r < 0])) if len(r[r < 0]) > 0 else 0.0
    return _cl((up - dn) / (up + dn + 1e-12))


def compute_institutional_wave(
    close: Any, high: Any, low: Any, volume: Any, window: int = 20
) -> float:
    """Volume-confirmed directional flow [-1, 1]."""
    c, v = (np.asarray(x, dtype=float) for x in (close, volume))
    n = min(len(c), len(v))
    if n < window:
        return 0.0
    c, v = c[-n:], v[-n:]
    return _cl(
        float(np.sum(np.sign(np.diff(c)) * v[1:]))
        / (float(np.sum(v[1:])) + 1e-12)
        * 3.0
    )


def compute_keltner_position(
    close: Any, high: Any, low: Any, window: int = 20
) -> float:
    """Position within Keltner channels [0, 1]."""
    c, h, l = (np.asarray(x, dtype=float) for x in (close, high, low))
    n = min(len(c), len(h), len(l))
    if n < window:
        return 0.5
    c, h, l = c[-n:], h[-n:], l[-n:]
    atr = float(np.mean(h[-window:] - l[-window:])) + 1e-12
    return _cl(
        (c[-1] - (float(np.mean(c[-window:])) - 2 * atr)) / (4 * atr + 1e-12), 0.0, 1.0
    )


def compute_vw_macd_hist(
    close: Any, volume: Any, fast: int = 12, slow: int = 26
) -> float:
    """Volume-weighted MACD histogram [-1, 1]."""
    c, v = np.asarray(close, dtype=float), np.asarray(volume, dtype=float)
    n = min(len(c), len(v))
    if n < slow:
        return 0.0
    c, v = c[-n:], v[-n:]
    vw = c * v
    af, as_ = 2.0 / (fast + 1), 2.0 / (slow + 1)
    ef, es = float(vw[0]), float(vw[0])
    for x in vw[1:]:
        ef = af * float(x) + (1 - af) * ef
        es = as_ * float(x) + (1 - as_) * es
    return _cl((ef - es) / (float(np.std(c)) * float(np.mean(v)) + 1e-12))


def compute_flow_signals(
    close: Any, high: Any, low: Any, volume: Any
) -> dict[str, float]:
    """Flow/structure indicators into named dict."""
    return {
        "ad_signal": compute_ad_signal(close, high, low, volume),
        "obv_divergence": compute_obv_divergence(close, volume),
        "spread_tightness": compute_spread_tightness(high, low),
        "volume_profile_proximity": compute_volume_profile(close, volume),
        "trade_intensity": compute_trade_intensity(volume),
        "mean_reversion": compute_mean_reversion(close),
        "trend_strength": compute_trend_strength(close),
        "sr_proximity": compute_sr_proximity(close, high, low),
        "elliott_wave": compute_elliott_wave(close),
        "institutional_wave": compute_institutional_wave(close, high, low, volume),
        "keltner_position": compute_keltner_position(close, high, low),
        "vw_macd_hist": compute_vw_macd_hist(close, volume),
        "microstructure": 0.5,
        "fib_proximity": 0.5,
    }
