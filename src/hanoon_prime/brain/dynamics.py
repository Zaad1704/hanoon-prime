"""hanoon_prime.brain.dynamics — hysteresis, score velocity, adaptive thresholds.

Smooths raw decision scores over time and applies dynamic triggering
thresholds. Prevents rapid flip-flopping between BUY/SELL.
Refractory periods suppress over-trading after execution events.

Merge of rebuild's adaptive_thresholds + score_velocity + hysteresis.
"""

from __future__ import annotations

import time
from collections import deque

from .config import (
    HYSTERESIS_DELTA,
    SCORE_VELOCITY_WINDOW,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
)


class Dynamics:
    """Score stabilization and adaptive threshold management."""

    def __init__(self, base_threshold: float = 0.58) -> None:
        self._base_threshold = base_threshold
        self._threshold = base_threshold
        self._score_history: deque[float] = deque(maxlen=SCORE_VELOCITY_WINDOW)
        self._last_direction: int = 0
        self._last_score: float = 0.0
        self._refractory_until: float = 0.0
        self._refractory_lambda: float = 2.0

    def process(self, raw_score: float, direction: int) -> tuple[float, str]:
        """Apply hysteresis + velocity + refractory. Returns (stabilized, reason)."""
        self._score_history.append(raw_score)
        velocity = self._compute_velocity()
        hysteresis_adj = self._apply_hysteresis(raw_score, direction)
        refractory_adj = self._apply_refractory()
        stabilized = raw_score + hysteresis_adj + velocity * 0.1 + refractory_adj
        stabilized = max(-1.0, min(1.0, stabilized))
        self._last_score = raw_score
        self._last_direction = direction
        return (
            stabilized,
            f"vel={velocity:.3f} hyst={hysteresis_adj:.3f} ref={refractory_adj:.3f}",
        )

    @property
    def threshold(self) -> float:
        """Current dynamic entry threshold."""
        return self._threshold

    def adapt_threshold(self, prediction_error: float) -> None:
        """Raise threshold when errors are high, lower when low."""
        if prediction_error > 0.6:
            self._threshold = min(THRESHOLD_MAX, self._threshold + 0.01)
        elif prediction_error < 0.3:
            self._threshold = max(THRESHOLD_MIN, self._threshold - 0.005)

    def set_refractory(self, duration: float = 2.0) -> None:
        """Set refractory period after trade event (neuronal hyperpolarization)."""
        self._refractory_until = time.time() + duration

    def _apply_refractory(self) -> float:
        """Exponential decay penalty after trade events."""
        now = time.time()
        if now >= self._refractory_until:
            return 0.0
        remaining = self._refractory_until - now
        val = -0.3 * (1.0 - 2.718 ** (-self._refractory_lambda * remaining))
        return float(val)

    def _compute_velocity(self) -> float:
        """Rate of score change. Positive = accelerating up."""
        if len(self._score_history) < 2:
            return 0.0
        scores = list(self._score_history)
        return scores[-1] - scores[-2]

    def _apply_hysteresis(self, score: float, direction: int) -> float:
        """Penalize direction changes (prevent flip-flopping)."""
        if direction == self._last_direction or self._last_direction == 0:
            return 0.0
        return -HYSTERESIS_DELTA if direction != 0 else 0.0
