"""hanoon_prime.brain.indicators_core — oscillator/momentum indicators."""

from __future__ import annotations

from typing import Any

import numpy as np


def _cl(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp value, NaN-safe."""
    return lo if v != v else float(max(lo, min(hi, v)))


def _ema_last(arr: np.ndarray, span: int) -> float:
    """Last EMA value."""
    a = 2.0 / (span + 1)
    o = arr[0]
    for x in arr[1:]:
        o = a * x + (1 - a) * o
    return float(o)


def _ema_series(arr: np.ndarray, span: int) -> list[float]:
    """Full EMA series as list."""
    a = 2.0 / (span + 1)
    out = [float(arr[0])]
    for x in arr[1:]:
        out.append(a * float(x) + (1 - a) * out[-1])
    return out


def compute_rsi(close: Any, window: int = 14) -> float:
    """Relative Strength Index [0, 1]."""
    c = np.asarray(close, dtype=float)
    if len(c) < window + 1:
        return 0.5
    d = np.diff(c[-(window + 1) :])
    g, l = float(np.mean(np.where(d > 0, d, 0.0))), float(
        np.mean(np.where(d < 0, -d, 0.0))
    )
    return _cl(100.0 - 100.0 / (1.0 + g / max(l, 1e-12)), 0.0, 1.0) / 100.0


def compute_macd_hist(
    close: Any, fast: int = 12, slow: int = 26, signal: int = 9
) -> float:
    """MACD histogram [-1, 1]."""
    c = np.asarray(close, dtype=float)
    if len(c) < slow + signal:
        return 0.0
    ml = [f - s for f, s in zip(_ema_series(c, fast), _ema_series(c, slow))]
    sig = _ema_series(np.array(ml), signal)
    return _cl((ml[-1] - sig[-1]) / (float(np.std(c)) + 1e-12))


def compute_bollinger_position(close: Any, window: int = 20) -> float:
    """Bollinger %B [0, 1]."""
    c = np.asarray(close, dtype=float)
    if len(c) < window:
        return 0.5
    w = c[-window:]
    sma, std = float(np.mean(w)), float(np.std(w))
    return (
        _cl((c[-1] - (sma - 2 * std)) / (4 * std + 1e-12), 0.0, 1.0) if std > 0 else 0.5
    )


def compute_stochastic_rsi(close: Any, window: int = 14) -> tuple[float, float]:
    """Stochastic RSI K and D [0, 1]."""
    c = np.asarray(close, dtype=float)
    if len(c) < window * 2:
        return 0.5, 0.5
    d = np.diff(c)
    g = float(np.mean(np.where(d > 0, d, 0.0)[-window:]))
    l = float(np.mean(np.where(d < 0, -d, 0.0)[-window:]))
    if l < 1e-12:
        return 0.9, 0.9
    rsi = _cl((100.0 - 100.0 / (1.0 + g / l)) / 100.0, 0.0, 1.0)
    return rsi, rsi


def compute_mfi(
    close: Any, high: Any, low: Any, volume: Any, window: int = 14
) -> float:
    """Money Flow Index [0, 1]."""
    c, h, l, v = (np.asarray(x, dtype=float) for x in (close, high, low, volume))
    n = min(len(c), len(h), len(l), len(v))
    if n < window + 1:
        return 0.5
    c, h, l, v = c[-n:], h[-n:], l[-n:], v[-n:]
    delta = np.diff((h + l + c) / 3.0 * v)
    p, ne = float(np.sum(delta[delta > 0])), float(np.abs(np.sum(delta[delta < 0])))
    return (
        _cl(100.0 - 100.0 / (1.0 + p / max(ne, 1e-12)), 0.0, 1.0) / 100.0
        if p + ne > 0
        else 0.5
    )


def compute_adx(high: Any, low: Any, close: Any, window: int = 14) -> float:
    """Average Directional Index [0, 1]."""
    h, l, c = (np.asarray(x, dtype=float) for x in (high, low, close))
    n = min(len(h), len(l), len(c))
    if n < window + 1:
        return 0.0
    h, l, c = h[-n:], l[-n:], c[-n:]
    up, dn = h[1:] - h[:-1], l[:-1] - l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    atr = float(np.sum(tr[-window:])) + 1e-12
    return _cl(
        abs(float(np.sum(pdm[-window:])) / atr - float(np.sum(mdm[-window:])) / atr)
        / (
            float(np.sum(pdm[-window:])) / atr
            + float(np.sum(mdm[-window:])) / atr
            + 1e-12
        ),
        0.0,
        1.0,
    )


def compute_hurst_exponent(close: Any, max_lag: int = 20) -> float:
    """Hurst exponent [0, 1], 0.5 = random."""
    c = np.asarray(close, dtype=float)
    if len(c) < max_lag + 1:
        return 0.5
    lags = list(range(2, min(max_lag, len(c) - 1)))
    tau = [
        float(np.std(c[lg:] - c[:-lg]))
        for lg in lags
        if float(np.std(c[lg:] - c[:-lg])) > 0
    ]
    if len(tau) < 2:
        return 0.5
    return _cl(
        float(
            np.polyfit(np.log(np.array(lags[: len(tau)])), np.log(np.array(tau)), 1)[0]
        ),
        0.0,
        1.0,
    )


def compute_kelly_fraction(close: Any, window: int = 30) -> float:
    """Kelly fraction from realized returns [0, 1]."""
    c = np.asarray(close, dtype=float)
    if len(c) < window:
        return 0.0
    rets = np.diff(c[-window:])
    w, lo = rets[rets > 0], -rets[rets < 0]
    wr = float(len(w)) / max(len(rets), 1)
    aw = float(np.mean(w)) if len(w) else 0.0
    al = float(np.mean(lo)) if len(lo) else 0.0
    return (
        _cl(
            (wr * (aw / max(al, 1e-12) + 1) - 1) / max(aw / max(al, 1e-12), 1e-12),
            0.0,
            1.0,
        )
        if wr < 1 and al > 0
        else 0.0
    )


def compute_osc_signals(
    close: Any, high: Any, low: Any, volume: Any
) -> dict[str, float]:
    """Oscillator indicators into named dict."""
    k, d = compute_stochastic_rsi(close)
    return {
        "rsi": compute_rsi(close),
        "macd_hist": compute_macd_hist(close),
        "bollinger_position": compute_bollinger_position(close),
        "stoch_k": k,
        "stoch_d": d,
        "mfi": compute_mfi(close, high, low, volume),
        "adx": compute_adx(high, low, close),
        "hurst_exponent": compute_hurst_exponent(close),
        "kelly_fraction": compute_kelly_fraction(close),
    }
