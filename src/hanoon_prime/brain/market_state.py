"""brain.market_state — Rich market state object.

Maintains a structured market state object that all modules can read.
Replaces flat dicts with typed, documented state.

Source: rebuild's market_state.py (simplified).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MarketState:
    """Rich market state shared across all modules."""

    # Price data
    last_prices: dict[str, float] = field(default_factory=dict)
    daily_volumes: dict[str, float] = field(default_factory=dict)
    spreads: dict[str, float] = field(default_factory=dict)
    # Derived signals
    regime: str = "unknown"
    regime_confidence: float = 0.5
    vix_level: float = 0.0
    # Session state
    session: str = "RTH"
    is_market_open: bool = True
    time_of_day: float = 0.5  # 0=open, 1=close
    # Portfolio state
    open_positions: dict[str, float] = field(default_factory=dict)
    daily_pnl: float = 0.0
    equity: float = 0.0
    # Timing
    last_update: float = 0.0

    def update_price(
        self, ticker: str, price: float, volume: float = 0, spread: float = 0
    ) -> None:
        """Update price data for a ticker."""
        self.last_prices[ticker] = price
        if volume > 0:
            self.daily_volumes[ticker] = volume
        if spread > 0:
            self.spreads[ticker] = spread
        self.last_update = time.time()

    def get_snapshot(self) -> dict[str, Any]:
        """Auto-generated docstring."""
        return {
            "regime": self.regime,
            "session": self.session,
            "n_positions": len(self.open_positions),
            "daily_pnl": self.daily_pnl,
            "n_tickers": len(self.last_prices),
        }
