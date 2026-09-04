"""hanoon_prime.brain.neurons.threshold_adapter — Dynamic spike threshold.

Adjusts neuron firing thresholds based on market conditions:
- High volatility → higher thresholds (prevent overshoot)
- Low volatility → lower thresholds (allow subtler signals)
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional


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
        """Compute threshold scaled by local volatility."""
        history = self.price_history.get(asset_idx, [])

        if len(history) < 10:
            return self.base_threshold

        prices = history[-min(len(history), self.volatility_window) :]
        returns = [prices[i + 1] / prices[i] - 1 for i in range(len(prices) - 1)]

        if not returns:
            return self.base_threshold

        vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5
        vix_scale = 1.0 + 0.02 * max(0.0, self.vix - 15.0)
        dynamic_theta = self.base_threshold * vol * vix_scale

        return max(
            self.base_threshold * 0.5, min(self.base_threshold * 5.0, dynamic_theta)
        )

    def adapt_for_market(
        self,
        ticker: str,
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
