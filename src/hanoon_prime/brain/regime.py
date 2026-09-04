"""hanoon_prime.brain.regime — market regime detection.

Classifies market context into trending/ranging/volatile using
volatility percentile and trend strength. Returns a regime multiplier
that scales downstream signals.

Pure numpy, no ML.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import REGIME_TREND_WINDOW, REGIME_VOL_WINDOW


@dataclass
class RegimeState:
    """Current market regime classification."""

    regime: str = "unknown"
    vol_percentile: float = 0.5
    trend_strength: float = 0.0
    multiplier: float = 1.0


class RegimeDetector:
    """Detects market regime from price and volume history."""

    def __init__(self) -> None:
        self._vol_history: list[float] = []

    def detect(
        self,
        close: list[float] | np.ndarray,
        high: list[float] | np.ndarray | None = None,
        low: list[float] | np.ndarray | None = None,
        volume: list[float] | np.ndarray | None = None,
    ) -> RegimeState:
        """Classify current regime from OHLCV arrays."""
        c = np.asarray(close, dtype=float)
        if len(c) < max(REGIME_VOL_WINDOW, REGIME_TREND_WINDOW):
            return RegimeState()
        vol = self._compute_volatility(c)
        self._vol_history.append(vol)
        if len(self._vol_history) > 200:
            self._vol_history = self._vol_history[-200:]
        vol_pct = self._percentile(vol)
        trend = self._compute_trend(c)
        regime, mult = self._classify(vol_pct, trend)
        return RegimeState(
            regime=regime,
            vol_percentile=vol_pct,
            trend_strength=trend,
            multiplier=mult,
        )

    @staticmethod
    def _compute_volatility(c: np.ndarray) -> float:
        """Recent returns standard deviation."""
        if len(c) < 2:
            return 0.0
        returns = np.diff(c[-REGIME_VOL_WINDOW:]) / np.maximum(
            c[-REGIME_VOL_WINDOW:-1], 1e-12
        )
        return float(np.std(returns))

    @staticmethod
    def _compute_trend(c: np.ndarray) -> float:
        """Normalized slope of recent prices."""
        n = min(REGIME_TREND_WINDOW, len(c))
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, c[-n:], 1)[0])
        std = float(np.std(c[-n:]))
        if std == 0:
            return 0.0
        return float(max(-1.0, min(1.0, slope / (std + 1e-12) * 5)))

    def _percentile(self, value: float) -> float:
        """Where does value sit in recent history?"""
        if len(self._vol_history) < 5:
            return 0.5
        arr = np.array(self._vol_history, dtype=float)
        return float(np.mean(arr <= value))

    @staticmethod
    def _classify(vol_pct: float, trend: float) -> tuple[str, float]:
        """Map vol percentile + trend to regime + multiplier."""
        abs_trend = abs(trend)
        if vol_pct > 0.80:
            return "volatile", 0.7
        if abs_trend > 0.3 and vol_pct < 0.60:
            direction = "bullish" if trend > 0 else "bearish"
            return f"trending_{direction}", 1.3
        if abs_trend < 0.1 and vol_pct < 0.40:
            return "ranging", 0.9
        return "normal", 1.0
