"""hanoon_prime.brain.salience — attention, working memory, uncertainty.

Dynamically filters noise, tracks active state confidence, and dampens
signals when uncertainty is high.

Merge of rebuild's attention.py + working_memory.py + uncertainty.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import CONFIDENCE_FLOOR


@dataclass
class SalienceState:
    """Current salience/attention state."""

    attention: float = 1.0
    uncertainty: float = 0.0
    confidence_atten: float = 1.0
    noise_level: float = 0.0


class Salience:
    """Attention and uncertainty dampening for the brain pipeline."""

    def __init__(self, window: int = 50) -> None:
        self._window = window
        self._score_history: deque[float] = deque(maxlen=window)
        self._vol_history: deque[float] = deque(maxlen=window)

    def update(self, score: float, volatility: float = 0.0) -> None:
        """Feed current score and volatility for attention tracking."""
        self._score_history.append(score)
        self._vol_history.append(volatility)

    def evaluate(self, score: float, regime_mult: float = 1.0) -> SalienceState:
        """Compute attention and uncertainty dampening."""
        noise = self._compute_noise()
        uncertainty = self._compute_uncertainty()
        attention = self._compute_attention(noise, regime_mult)
        conf_atten = self._compute_confidence_attenuation(uncertainty)
        return SalienceState(
            attention=attention,
            uncertainty=uncertainty,
            confidence_atten=conf_atten,
            noise_level=noise,
        )

    def _compute_noise(self) -> float:
        """Score volatility = noise level. [0, 1]."""
        if len(self._score_history) < 5:
            return 0.0
        arr = np.array(self._score_history, dtype=float)
        return float(min(1.0, np.std(arr) * 5))

    def _compute_uncertainty(self) -> float:
        """High vol + erratic scores = uncertainty. [0, 1]."""
        if len(self._vol_history) < 5:
            return 0.0
        vol_arr = np.array(self._vol_history, dtype=float)
        vol_component = float(min(1.0, np.mean(vol_arr[-10:]) * 10))
        noise = self._compute_noise()
        return min(1.0, vol_component * 0.6 + noise * 0.4)

    @staticmethod
    def _compute_attention(noise: float, regime_mult: float) -> float:
        """Higher noise → lower attention. [0.3, 1.0]."""
        base = 1.0 - noise * 0.5
        adjusted = base * regime_mult
        return float(max(0.3, min(1.0, adjusted)))

    @staticmethod
    def _compute_confidence_attenuation(uncertainty: float) -> float:
        """High uncertainty → reduce confidence. [0.5, 1.0]."""
        atten = 1.0 - uncertainty * 0.5
        return float(max(CONFIDENCE_FLOOR, min(1.0, atten)))
