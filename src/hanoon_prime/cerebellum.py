"""hanoon_prime.cerebellum — exactly 5 indicators from IB market data.

No more, no fewer. Each indicator is a pure function of OHLCV + depth
data. Every indicator is validated by tests/test_indicator_edge.py —
if it can't show significant edge (p < 0.05 via permutation test), it
doesn't belong here.

All 5 indicators preserve the original semantics:
  - vpin: signed buying pressure [-1, 1]
  - orderbook_imbalance: bid vs ask [-1, 1]
  - institutional_flow: price-volume confirmation [0, 1]
  - momentum: normalized price change [-1, 1]
  - vwap_deviation: price vs VWAP [-1, 1]
"""

from __future__ import annotations

from typing import Any

import numpy as np

INDICATOR_NAMES: tuple[str, ...] = (
    "vpin",
    "orderbook_imbalance",
    "institutional_flow",
    "momentum",
    "vwap_deviation",
)


def _safe_float(x: float | int | None, default: float = 0.0) -> float:
    """Coerce to float, returning *default* on None or NaN."""
    if x is None:
        return default
    try:
        v = float(x)
        return v if v == v else default  # NaN check
    except (TypeError, ValueError):
        return default


def compute_vpin(volume: Any, buy_volume: Any) -> float:
    """Volume-Synchronized Probability of Informed Trading.

    VPIN = signed (buy_volume - sell_volume) / total_volume, averaged
    over the window. Positive = buying pressure, negative = selling.
    Range: [-1.0, +1.0].
    """
    v = np.asarray(volume, dtype=float)
    bv = np.asarray(buy_volume, dtype=float)
    if len(v) < 2 or np.sum(v) <= 0:
        return 0.0
    sv = np.maximum(v - bv, 0)
    signed = bv - sv
    vpin = float(np.mean(signed / np.maximum(v, 1e-12)))
    return float(np.clip(vpin, -1.0, 1.0))


def compute_orderbook_imbalance(bid_sizes: Any, ask_sizes: Any) -> float:
    """Orderbook imbalance from IB depth-of-market.

    imbalance = (total_bid - total_ask) / (total_bid + total_ask)
    Positive = buying pressure, negative = selling pressure.
    Range: [-1.0, +1.0].
    """
    bids = np.asarray(bid_sizes, dtype=float)
    asks = np.asarray(ask_sizes, dtype=float)
    total_bid = float(np.sum(bids)) if len(bids) else 0.0
    total_ask = float(np.sum(asks)) if len(asks) else 0.0
    denom = total_bid + total_ask
    if denom <= 0:
        return 0.0
    return float((total_bid - total_ask) / denom)


def compute_institutional_flow(close: Any, volume: Any, avg_volume: float) -> float:
    """Price-volume confirmation signal for institutional activity.

    Institutions trade large blocks — when they buy, both price AND
    volume increase; when they sell, both decrease. The pooled edge
    test shows a POSITIVE correlation (corr≈+0.021, p<0.001): price+vol
    spikes tend to bounce on 1-min bars.

    Returns [0, 1]:
      1.0 = price ↓ + volume ↓ (selling climax → reversion up → bullish)
      0.0 = price ↑ + volume ↑ (buying climax → reversion down → bearish)
      0.5 = mixed/neutral
    """
    c = np.asarray(close, dtype=float)
    v = np.asarray(volume, dtype=float)
    if len(c) < 5 or len(v) < 5 or avg_volume <= 0:
        return 0.5
    price_up = bool(c[-1] > c[-5])
    vol_up = bool(v[-1] > float(np.mean(v[-5:])))
    if price_up and vol_up:
        return 0.0  # spike → reverts down
    if not price_up and not vol_up:
        return 1.0  # dip → reverts up
    return 0.5


def compute_momentum(close: Any, lookback: int = 5) -> float:
    """Normalized momentum over *lookback* periods.

    momentum = (close[-1] - close[-lookback]) / close[-lookback]
    Positive = bullish continuation, negative = bearish.
    Range: [-1.0, +1.0].
    """
    c = np.asarray(close, dtype=float)
    if len(c) < lookback + 1 or c[-lookback] == 0:
        return 0.0
    mom = (c[-1] - c[-lookback]) / c[-lookback]
    return float(np.clip(mom, -1.0, 1.0))


def compute_vwap_deviation(close: Any, volume: Any) -> float:
    """Deviation from volume-weighted average price.

    vwap_dev = (close[-1] - vwap) / vwap
    Positive = price above VWAP (bullish), negative = below (bearish).
    Range: [-1.0, +1.0].
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
    close: Any,
    volume: Any,
    buy_volume: Any | None = None,
    bid_sizes: Any | None = None,
    ask_sizes: Any | None = None,
) -> dict[str, float]:
    """Compute all 5 indicators + risk metrics. Returns a named dict.

    Args:
        close: 1-min close prices.
        volume: total volume per bar.
        buy_volume: buy-side volume (defaults to 50% of total).
        bid_sizes: bid sizes from depth of market.
        ask_sizes: ask sizes from depth of market.

    Returns:
        Dict with 5 indicator keys + volatility + vpin_magnitude.
    """
    avg_vol = float(np.mean(volume)) if len(volume) > 0 else 1.0
    bv = buy_volume if buy_volume is not None else volume * 0.5
    bids = bid_sizes if bid_sizes is not None else np.array([1.0])
    asks = ask_sizes if ask_sizes is not None else np.array([1.0])

    return {
        "vpin": compute_vpin(volume, bv),
        "orderbook_imbalance": compute_orderbook_imbalance(bids, asks),
        "institutional_flow": compute_institutional_flow(close, volume, avg_vol),
        "momentum": compute_momentum(close),
        "vwap_deviation": compute_vwap_deviation(close, volume),
        "volatility": _compute_volatility(close),
        "vpin_magnitude": _compute_vpin_magnitude(volume, bv),
    }


def _compute_vpin_magnitude(volume: Any, buy_volume: Any) -> float:
    """Unsigned VPIN magnitude — |buy_vol - sell_vol| / total_volume. [0, 1]."""
    v = np.asarray(volume, dtype=float)
    bv = np.asarray(buy_volume, dtype=float)
    if len(v) < 2 or np.sum(v) <= 0:
        return 0.0
    sv = np.maximum(v - bv, 0)
    signed = np.abs(bv - sv)
    vpin = float(np.mean(signed / np.maximum(v, 1e-12)))
    return float(np.clip(vpin, 0.0, 1.0))


def _compute_volatility(close: Any) -> float:
    """Rolling price range normalized by current price (risk metric)."""
    c = np.asarray(close, dtype=float)
    if len(c) < 2:
        return 0.0
    price_range = float(np.max(c) - np.min(c))
    if c[-1] == 0:
        return 0.0
    return float(np.clip(price_range / abs(c[-1]), 0.0, 1.0))
