"""brain.weight_manager — Sophisticated weight lifecycle management.

Manages weight creation, adaptation, decay, and recovery.
More advanced than simple dict operations.

Source: rebuild's weight_manager.py (simplified).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from .config import DEFAULT_WEIGHTS, LEARNING_RATE, WEIGHT_DECAY

log = logging.getLogger(__name__)


@dataclass
class WeightState:
    weights: dict[str, float]
    last_adapt: float = 0.0
    adapt_count: int = 0
    sum: float = 0.0


class WeightManager:
    """Sophisticated weight lifecycle management."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        self._last_adapt: float = 0.0
        self._adapt_count: int = 0

    def adapt(self, indicator: str, won: bool, magnitude: float = 1.0) -> None:
        """Adapt weight based on outcome."""
        if indicator not in self._weights:
            return
        step = LEARNING_RATE * magnitude
        if won:
            self._weights[indicator] += step
        else:
            self._weights[indicator] -= step * 1.2  # asymmetric penalty
        self._last_adapt = time.time()
        self._adapt_count += 1
        # Periodic decay
        if self._adapt_count % 100 == 0:
            self._apply_decay()

    def _apply_decay(self) -> None:
        """Auto-generated docstring."""
        for k in self._weights:
            self._weights[k] *= WEIGHT_DECAY

    def get_weights(self) -> dict[str, float]:
        """Auto-generated docstring."""
        return dict(self._weights)

    def set_weights(self, weights: dict[str, float]) -> None:
        """Auto-generated docstring."""
        self._weights = dict(weights)

    def snapshot(self) -> WeightState:
        """Auto-generated docstring."""
        return WeightState(
            weights=dict(self._weights),
            last_adapt=self._last_adapt,
            adapt_count=self._adapt_count,
            sum=sum(self._weights.values()),
        )
