"""hanoon_prime.alpha — exactly 5 indicators from IB market data.

No more, no fewer. Each indicator is a pure function of OHLCV + depth data.
Every indicator is validated by tests/test_indicator_edge.py — if it can't
show positive correlation with next-bar return, it doesn't belong here.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

INDICATOR_NAMES: tuple[str, ...] = (
    "vpin",
    "orderbook_imbalance",
    "institutional_flow",
    "momentum",
    "vwap_deviation",
)


def _safe_float(x: float | int | None, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
        return v if v == v else default  # NaN check
    except (TypeError, ValueError):
        return default


def compute_vpin(volume: NDArray, buy_volume: NDArray) -> float:
    """Volume-Synchronized Probability of Informed Trading.

    VPIN = signed (buy_volume - sell_volume) / total_volume, averaged
    over window. Positive = buying pressure (informed buying), negative
    = selling pressure.

    Returns: -1.0 (maximum selling pressure) → +1.0 (maximum buying pressure)
    """
    v = np.asarray(volume, dtype=float)
    bv = np.asarray(buy_volume, dtype=float)
    if len(v) < 2 or np.sum(v) <= 0:
        return 0.0
    sv = np.maximum(v - bv, 0)
    signed = bv - sv
    vpin = float(np.mean(signed / np.maximum(v, 1e-12)))
    return float(np.clip(vpin, -1.0, 1.0))


def compute_orderbook_imbalance(bid_sizes: NDArray, ask_sizes: NDArray) -> float:
    """Orderbook imbalance from IB depth-of-market.

    imbalance = (total_bid - total_ask) / (total_bid + total_ask)
    Positive = buying pressure, negative = selling pressure.
    Normalized to [-1, 1].

    Signal: extreme imbalance that persists tends to predict near-term
    price movement in the direction of the imbalance.
    """
    bids = np.asarray(bid_sizes, dtype=float)
    asks = np.asarray(ask_sizes, dtype=float)
    total_bid = float(np.sum(bids)) if len(bids) else 0.0
    total_ask = float(np.sum(asks)) if len(asks) else 0.0
    denom = total_bid + total_ask
    if denom <= 0:
        return 0.0
    return float((total_bid - total_ask) / denom)


def compute_institutional_flow(volume: NDArray, avg_volume: float) -> float:
    """Volume-spike proxy for institutional activity.

    inst_flow = min(volume / avg_volume, 3.0) — normalized so that
    1.0 = normal, 3.0 = 3x normal volume (institutional spike).

    Signal: volume spikes without proportional price movement often
    indicate accumulation/distribution by large players.
    """
    if avg_volume <= 0:
        return 1.0
    v = np.asarray(volume, dtype=float)
    recent = float(np.mean(v[-5:])) if len(v) >= 5 else float(np.mean(v))
    ratio = recent / avg_volume
    return float(np.clip(ratio, 0.0, 3.0))


def compute_momentum(close: NDArray, lookback: int = 5) -> float:
    """Normalized momentum over lookback periods.

    momentum = (close[-1] - close[-lookback]) / close[-lookback]
    Returns signed value. Positive = bullish continuation, negative = bearish.

    Signal: momentum that is strong and in the same direction as order
    flow tends to persist for 1-2 bars.
    """
    c = np.asarray(close, dtype=float)
    if len(c) < lookback + 1 or c[-lookback] == 0:
        return 0.0
    mom = (c[-1] - c[-lookback]) / c[-lookback]
    return float(np.clip(mom, -1.0, 1.0))


def compute_vwap_deviation(
    close: NDArray, volume: NDArray
) -> float:
    """Deviation from volume-weighted average price.

    vwap_dev = (close[-1] - vwap) / vwap
    Positive = price above VWAP (bullish), negative = below (bearish).

    Signal: price deviating from VWAP and converging back tends to
    produce short-term reversals; extended deviation without convergence
    indicates sustained momentum.
    """
    c = np.asarray(close, dtype=float)
    v = np.asarray(volume, dtype=float)
    if len(c) < 2 or np.sum(v) <= 0:
        return 0.0
    vwap = float(np.sum(c * v) / np.sum(v))
    if vwap == 0:
        return 0.0
    dev = (c[-1] - vwap) / vwap
    return float(np.clip(dev, -1.0, 1.0))


def compute_alpha(
    close: NDArray,
    volume: NDArray,
    buy_volume: NDArray | None = None,
    bid_sizes: NDArray | None = None,
    ask_sizes: NDArray | None = None,
) -> dict[str, float]:
    """Compute all 5 indicators and return as a named dict.

    Args:
        close: close prices (1-min bars).
        volume: total volume per bar.
        buy_volume: buy-side volume per bar (volume * buy_fraction).
        bid_sizes: bid sizes from depth of market.
        ask_sizes: ask sizes from depth of market.

    Returns:
        Dict with 5 indicator keys + 1 risk key (volatility).
        Indicators: vpin, orderbook_imbalance, institutional_flow,
        momentum, vwap_deviation. volatility is a risk metric, not an
        indicator.
    """
    avg_vol = float(np.mean(volume)) if len(volume) > 0 else 1.0

    bv = buy_volume if buy_volume is not None else volume * 0.5
    bids = bid_sizes if bid_sizes is not None else np.array([1.0])
    asks = ask_sizes if ask_sizes is not None else np.array([1.0])

    alpha = {
        "vpin": compute_vpin(volume, bv),
        "orderbook_imbalance": compute_orderbook_imbalance(bids, asks),
        "institutional_flow": compute_institutional_flow(volume, avg_vol),
        "momentum": compute_momentum(close),
        "vwap_deviation": compute_vwap_deviation(close, volume),
        "volatility": _compute_volatility(close),
        "vpin_magnitude": _compute_vpin_magnitude(volume, bv),
    }
    return alpha


def _compute_vpin_magnitude(volume: NDArray, buy_volume: NDArray) -> float:
    """Unsigned VPIN magnitude — |buy_vol - sell_vol| / total_volume.

    This is the magnitude-only version of VPIN (always [0, 1]). The signed
    VPIN (compute_vpin) is used for directional edge detection; this
    magnitude version is used for non-directional scoring.
    """
    v = np.asarray(volume, dtype=float)
    bv = np.asarray(buy_volume, dtype=float)
    if len(v) < 2 or np.sum(v) <= 0:
        return 0.0
    sv = np.maximum(v - bv, 0)
    signed = np.abs(bv - sv)
    vpin = float(np.mean(signed / np.maximum(v, 1e-12)))
    return float(np.clip(vpin, 0.0, 1.0))


def _compute_volatility(close: NDArray) -> float:
    """Rolling price range normalized by current price.

    This is a RISK metric, not a 5th indicator. It tells the thinker
    whether the ticker is tradable (enough movement to hit stop/target).
    """
    c = np.asarray(close, dtype=float)
    if len(c) < 2:
        return 0.0
    price_range = float(np.max(c) - np.min(c))
    if c[-1] == 0:
        return 0.0
    return float(np.clip(price_range / abs(c[-1]), 0.0, 1.0))
