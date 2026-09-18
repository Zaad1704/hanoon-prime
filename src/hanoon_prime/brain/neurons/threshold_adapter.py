"""hanoon_prime.brain.neurons.threshold_adapter — Dynamic spike threshold.

Adjusts neuron firing thresholds based on market conditions:
- High volatility → higher thresholds (prevent overshoot)
- Low volatility → lower thresholds (allow subtler signals)

The multiplier is measured against a nominal 1%-per-bar return volatility, so
a doubling of realized vol doubles the threshold (clamped to [0.5x, 5.0x]).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

NOMINAL_REL_VOL: float = 0.01  # 1% per-bar returns == 1.0x scale factor


class DynamicThresholdAdapter:
    """Adaptive threshold engine for volatility-conditioned spiking."""

    def __init__(
        self,
        num_assets: int = 7,
        base_threshold: float = 0.02,
    ) -> None:
        self.base_threshold = base_threshold
        self.vix = 15.0
        self.volatility_window = 30
        self.price_history: Dict[int, List[float]] = {i: [] for i in range(num_assets)}
        self.current_asset: str = ""

    def update_vix(self, vix: float) -> None:
        """Update implied volatility."""
        self.vix = max(10.0, min(50.0, vix))

    def record_price(self, asset_idx: int, price: float) -> None:
        """Record a price tick for volatility calculation."""
        history = self.price_history.get(asset_idx, [])
        history.append(price)
        if len(history) > self.volatility_window:
            self.price_history[asset_idx] = history[-self.volatility_window :]

    def create_ticker_key(self, ticker: str) -> str:
        """Generate stable key for ticker-specific settings."""
        return hashlib.md5(ticker.encode()).hexdigest()[:8]

    def compute_dynamic_threshold(self, asset_idx: int) -> float:
        """Compute a base-scaled threshold from local volatility.

        Returns ``base_threshold * multiplier`` where the multiplier is the
        ratio of realized per-bar volatility to NOMINAL_REL_VOL, further scaled
        by VIX (clamped to [0.5x, 5.0x]). ``base_threshold`` stays the caller's
        unit (e.g. a neuron firing threshold), so output is directly usable.
        """
        history = self.price_history.get(asset_idx, [])

        if len(history) < 10:
            return self.base_threshold

        prices = history[-min(len(history), self.volatility_window) :]
        returns = [prices[i + 1] / prices[i] - 1 for i in range(len(prices) - 1)]

        if not returns:
            return self.base_threshold

        vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5
        vix_scale = 1.0 + 0.02 * max(0.0, self.vix - 15.0)
        multiplier = (vol / NOMINAL_REL_VOL) * vix_scale

        result: float = self.base_threshold * max(0.5, min(5.0, multiplier))
        return result

    def adapt_for_market(
        self,
        _ticker: str,
        regime: str,
        volatility: float,
    ) -> float:
        """Get threshold adapted for current market conditions."""
        if regime == "trending":
            return self.base_threshold * 1.5
        if regime == "mean_reverting":
            return self.base_threshold * 0.7
        return self.base_threshold * volatility


__all__ = ["DynamicThresholdAdapter"]
