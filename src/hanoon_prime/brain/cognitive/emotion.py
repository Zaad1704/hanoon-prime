"""hanoon_prime.brain.cognitive.emotion — Amygdala (affective state).

Computes fear, greed, frustration from recent trade outcomes.
Modulates confidence by ±0.05 and risk scalar by [0.85, 1.25].
Learned from JULI's own realized performance.
"""
from __future__ import annotations

from collections import deque

CONF_BOUND: float = 0.05
RISK_FLOOR: float = 0.85
RISK_CEIL: float = 1.25
LOOKBACK: int = 30
MIN_SAMPLES: int = 5


class EmotionState:
    """JULI's affective state — learned from realized performance."""

    def __init__(self) -> None:
        self._outcomes: deque[bool] = deque(maxlen=LOOKBACK)
        self._pnl_history: deque[float] = deque(maxlen=LOOKBACK)

    def update(self, won: bool, pnl_pct: float) -> None:
        """Record a trade outcome."""
        self._outcomes.append(won)
        self._pnl_history.append(pnl_pct)

    def confidence_mod(self) -> float:
        """Bounded confidence nudge from affect [-CONF_BOUND, +CONF_BOUND]."""
        if len(self._outcomes) < MIN_SAMPLES:
            return 0.0
        wr = sum(self._outcomes) / len(self._outcomes)
        recent_wr = sum(list(self._outcomes)[-5:]) / min(5, len(self._outcomes))
        mood = recent_wr - 0.5
        return max(-CONF_BOUND, min(CONF_BOUND, mood * 0.1))

    def risk_scalar(self) -> float:
        """Position size multiplier [RISK_FLOOR, RISK_CEIL]."""
        if len(self._outcomes) < MIN_SAMPLES:
            return 1.0
        wr = sum(self._outcomes) / len(self._outcomes)
        if wr > 0.6:
            return min(RISK_CEIL, 1.0 + (wr - 0.6) * 0.5)
        if wr < 0.4:
            return max(RISK_FLOOR, 1.0 - (0.4 - wr) * 0.5)
        return 1.0

    def fear(self) -> float:
        """Loss aversion level [0, 1]."""
        if len(self._pnl_history) < MIN_SAMPLES:
            return 0.0
        losses = [abs(p) for p in self._pnl_history if p < 0]
        return min(1.0, len(losses) / max(len(self._pnl_history), 1))

    def snapshot(self) -> dict[str, float]:
        """Current emotional state for telemetry."""
        return {
            "confidence_mod": round(self.confidence_mod(), 4),
            "risk_scalar": round(self.risk_scalar(), 4),
            "fear": round(self.fear(), 4),
            "samples": len(self._outcomes),
        }
