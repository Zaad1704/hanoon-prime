"""hanoon_prime.brain.amygdala — Threat evaluation and risk detection.

Monitors for danger signals: volatility spikes, sudden drops,
bid/ask imbalances, unusual volume. Can trigger immediate exits.

Biological analogy: Fight-or-flight center — detects threats
before the conscious mind (cortex) processes them.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

VOLATILITY_SPIKE_THRESHOLD: float = 2.0
DROP_THRESHOLD: float = -0.02  # -2% in 5 minutes
VOLUME_SPIKE_MULTIPLIER: float = 3.0


@dataclass
class ThreatLevel:
    """Amygdala's assessment of danger."""

    ticker: str
    score: float  # -1.0 (extreme danger) to +1.0 (safe)
    fear: float  # 0.0 (no fear) to 1.0 (panic)
    greed: float  # 0.0 (no greed) to 1.0 (euphoria)
    trigger_exit: bool = False
    reason: str = ""


class Amygdala:
    """Threat detection engine — microsecond risk evaluation."""

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {}
        self._volume_history: dict[str, list[float]] = {}

    def evaluate(
        self,
        ticker: str,
        bid: float,
        ask: float,
        last: float,
        volume: float,
        atr: float,
        prices: list[float] | None = None,
    ) -> ThreatLevel:
        """Evaluate threat level for a ticker."""
        fear = 0.0
        greed = 0.0
        reasons: list[str] = []

        if prices and len(prices) >= 5:
            fear += self._check_price_drop(prices, ticker, reasons)
            fear += self._check_volatility(prices, ticker, atr, reasons)

        fear += self._check_volume_spike(volume, ticker, reasons)
        fear += self._check_spread_widen(bid, ask, reasons)

        greed = self._check_greed(prices, reasons)

        score = max(-1.0, min(1.0, 1.0 - fear + greed))
        trigger_exit = fear > 0.8

        return ThreatLevel(
            ticker=ticker,
            score=score,
            fear=min(fear, 1.0),
            greed=min(greed, 1.0),
            trigger_exit=trigger_exit,
            reason="; ".join(reasons) if reasons else "normal",
        )

    def _check_price_drop(
        self,
        prices: list[float],
        ticker: str,
        reasons: list[str],
    ) -> float:
        """Check for sudden price drops."""
        if len(prices) < 2:
            return 0.0
        recent = prices[-5:] if len(prices) >= 5 else prices
        change = (recent[-1] - recent[0]) / max(recent[0], 0.01)
        if change < DROP_THRESHOLD:
            reasons.append(f"drop_{change:.3f}")
            return min(abs(change) * 10, 1.0)
        return 0.0

    def _check_volatility(
        self,
        prices: list[float],
        ticker: str,
        atr: float,
        reasons: list[str],
    ) -> float:
        """Check for volatility spikes."""
        if len(prices) < 10 or atr <= 0:
            return 0.0
        arr = np.array(prices[-10:], dtype=float)
        prev = arr[:-1]  # type: ignore[index]
        returns = np.diff(arr) / np.where(prev != 0, prev, 1.0)
        current_vol = float(np.std(returns))
        hist_vol = (
            float(np.std(np.diff(arr) / np.where(prev != 0, prev, 1.0)))
            if len(arr) > 2
            else current_vol
        )
        if hist_vol > 0 and current_vol > hist_vol * VOLATILITY_SPIKE_THRESHOLD:
            reasons.append("vol_spike")
            return 0.5
        return 0.0

    def _check_volume_spike(
        self,
        volume: float,
        ticker: str,
        reasons: list[str],
    ) -> float:
        """Check for unusual volume."""
        history = self._volume_history.get(ticker, [])
        history.append(volume)
        if len(history) > 100:
            history = history[-100:]
        self._volume_history[ticker] = history
        if len(history) < 10:
            return 0.0
        avg_vol = float(np.mean(history[:-1]))
        if avg_vol > 0 and volume > avg_vol * VOLUME_SPIKE_MULTIPLIER:
            reasons.append("vol_spike")
            return 0.3
        return 0.0

    def _check_spread_widen(self, bid: float, ask: float, reasons: list[str]) -> float:
        """Check for widening bid-ask spread."""
        if bid <= 0:
            return 0.5
        spread_pct = (ask - bid) / bid
        if spread_pct > 0.01:
            reasons.append(f"spread_{spread_pct:.4f}")
            return min(spread_pct * 20, 1.0)
        return 0.0

    def _check_greed(self, prices: list[float] | None, reasons: list[str]) -> float:
        """Check for euphoria/greed signals."""
        if not prices or len(prices) < 5:
            return 0.0
        recent = prices[-5:]
        if recent[-1] > recent[0] * 1.03:
            reasons.append("euphoria")
            return 0.3
        return 0.0
