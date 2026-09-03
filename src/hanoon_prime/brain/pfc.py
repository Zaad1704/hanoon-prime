"""hanoon_prime.brain.pfc — Prefrontal Cortex: regime and trend analysis.

Detects market regime (trending, ranging, volatile) and generates
multi-timeframe directional intent. Orchestrates the decision pipeline.

Biological analogy: Executive function — long-term planning,
context-dependent behavior, regime awareness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

REGIME_TREND_UP: str = "trend_up"
REGIME_TREND_DOWN: str = "trend_down"
REGIME_RANGING: str = "ranging"
REGIME_VOLATILE: str = "volatile"
SLOPE_THRESHOLD: float = 0.001
RANGE_THRESHOLD: float = 0.005


@dataclass
class PFCVerdict:
    """PFC's analysis output."""

    ticker: str
    regime: str
    intent: float  # -1.0 (short) to +1.0 (long)
    confidence: float  # 0.0 to 1.0
    multi_tf_score: float = 0.0
    reasons: list[str] | None = None


class PrefrontalCortex:
    """Regime detection and multi-timeframe trend analysis."""

    def __init__(self) -> None:
        self._regimes: dict[str, str] = {}

    def evaluate(
        self,
        ticker: str,
        prices: list[float],
        volumes: list[float],
        atr: float,
    ) -> PFCVerdict:
        """Full PFC evaluation: regime + trend + multi-timeframe."""
        if len(prices) < 20:
            return PFCVerdict(ticker, REGIME_RANGING, 0.0, 0.0)

        arr = np.array(prices, dtype=float)
        regime = self._detect_regime(arr, atr)
        intent = self._compute_intent(arr, regime)
        confidence = self._compute_confidence(arr, regime)
        multi_tf = self._multi_timeframe_score(arr)

        self._regimes[ticker] = regime
        return PFCVerdict(
            ticker=ticker,
            regime=regime,
            intent=intent,
            confidence=confidence,
            multi_tf_score=multi_tf,
        )

    def _detect_regime(self, prices: np.ndarray, atr: float) -> str:
        """Detect current market regime from price action."""
        c = [float(x) for x in prices]
        rets = [(c[i] - c[i - 1]) / max(c[i - 1], 0.01) for i in range(1, len(c))]
        vol = float(np.std(rets)) if rets else 0.0
        n = len(c)
        x_mean = (n - 1) / 2.0
        y_mean = sum(c) / n
        num = sum((i - x_mean) * (c[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        trend = num / den if den > 0 else 0.0
        normalized_trend = trend / max(float(np.mean(c)), 0.01)

        if vol > RANGE_THRESHOLD * 2:
            return REGIME_VOLATILE
        if normalized_trend > SLOPE_THRESHOLD:
            return REGIME_TREND_UP
        if normalized_trend < -SLOPE_THRESHOLD:
            return REGIME_TREND_DOWN
        return REGIME_RANGING

    def _compute_intent(self, prices: np.ndarray, regime: str) -> float:
        """Compute directional intent from regime and price position."""
        recent = prices[-10:]
        sma = float(np.mean(recent))
        current = float(recent[-1])

        if regime == REGIME_TREND_UP:
            return min((current - sma) / sma * 100, 1.0)
        if regime == REGIME_TREND_DOWN:
            return max((current - sma) / sma * 100, -1.0)
        return 0.0

    def _compute_confidence(self, prices: np.ndarray, regime: str) -> float:
        """How confident are we in the regime detection?"""
        if regime == REGIME_VOLATILE:
            return 0.3
        if regime == REGIME_RANGING:
            return 0.5
        returns = np.diff(prices) / prices[:-1]
        directional = sum(1 for r in returns[-10:] if r > 0) / max(
            len(returns[-10:]), 1
        )
        return abs(directional - 0.5) * 2

    def _multi_timeframe_score(self, prices: np.ndarray) -> float:
        """Score alignment across multiple timeframes."""
        scores = []
        for window in [5, 10, 20]:
            if len(prices) >= window:
                segment = prices[-window:]
                trend = (segment[-1] - segment[0]) / max(segment[0], 0.01)
                scores.append(np.clip(trend * 10, -1.0, 1.0))
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def get_regime(self, ticker: str) -> str:
        """Get cached regime for a ticker."""
        return self._regimes.get(ticker, REGIME_RANGING)
