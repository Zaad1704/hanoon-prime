"""brain.volatility — Volatility predictor for risk management.

Forecast future volatility using ATR-based methods.
Provides volatility regime detection and expected range prediction.

Source: rebuild's volatility_predictor.py (simplified).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class VolatilityState:
    current_vol: float = 0.0
    forecast_vol: float = 0.0
    regime: str = "normal"
    expected_range: float = 0.0
    modifier: float = 0.0


class VolatilityPredictor:
    """Forecast future volatility for better risk management."""

    def __init__(self, history_len: int = 50) -> None:
        """Auto-generated docstring."""
        self._atr_history: deque[float] = deque(maxlen=history_len)
        self._returns: deque[float] = deque(maxlen=history_len)

    def update(self, atr: float, price: float) -> VolatilityState:
        """Update with new ATR and price data."""
        self._atr_history.append(atr)
        if len(self._returns) > 0 and self._returns[-1] != 0:
            ret = (price - self._returns[-1]) / self._returns[-1]
            self._returns.append(ret)
        else:
            self._returns.append(0.0)

        current = self._current_vol()
        forecast = self._forecast_vol()
        regime = self._detect_regime(current, forecast)
        exp_range = forecast * price * 0.01
        mod = self._compute_modifier(regime)
        return VolatilityState(
            current_vol=current,
            forecast_vol=forecast,
            regime=regime,
            expected_range=exp_range,
            modifier=mod,
        )

    def _current_vol(self) -> float:
        """Auto-generated docstring."""
        if not self._atr_history:
            return 0.0
        return float(np.mean(list(self._atr_history)[-20:]))

    def _forecast_vol(self) -> float:
        """Auto-generated docstring."""
        if len(self._atr_history) < 10:
            return self._current_vol()
        atrs = list(self._atr_history)
        # Simple exponential smoothing
        alpha = 0.3
        forecast = atrs[0]
        for a in atrs[1:]:
            forecast = alpha * a + (1 - alpha) * forecast
        return forecast

    def _detect_regime(self, current: float, forecast: float) -> str:
        """Auto-generated docstring."""
        if len(self._atr_history) < 20:
            return "normal"
        arr = np.array(list(self._atr_history))
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std == 0:
            return "normal"
        z = (current - mean) / std
        if z > 1.5:
            return "high"
        if z < -1.0:
            return "low"
        return "normal"

    def _compute_modifier(self, regime: str) -> float:
        """Auto-generated docstring."""
        if regime == "high":
            return -0.02  # reduce size in high vol
        if regime == "low":
            return 0.01  # slight boost in low vol
        return 0.0
