"""brain.uncertainty — Uncertainty estimator for model confidence.

Quantifies how uncertain the model is about its predictions.
High uncertainty = reduce position size, widen stops.

Source: rebuild's cognitive/uncertainty.py (simplified).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class UncertaintyState:
    aleatoric: float = 0.0  # data noise
    epistemic: float = 0.0  # model uncertainty
    total: float = 0.0
    modifier: float = 0.0


class UncertaintyEstimator:
    """Quantify model uncertainty for risk adjustment."""

    def __init__(self, history_len: int = 50) -> None:
        """Auto-generated docstring."""
        self._predictions: deque[float] = deque(maxlen=history_len)
        self._outcomes: deque[float] = deque(maxlen=history_len)

    def update(self, prediction: float, outcome: float) -> UncertaintyState:
        """Update with prediction-outcome pair."""
        self._predictions.append(prediction)
        self._outcomes.append(outcome)
        aleatoric = self._compute_aleatoric()
        epistemic = self._compute_epistemic()
        total = math.sqrt(aleatoric**2 + epistemic**2)
        mod = self._compute_modifier(total)
        return UncertaintyState(
            aleatoric=aleatoric, epistemic=epistemic, total=total, modifier=mod
        )

    def _compute_aleatoric(self) -> float:
        """Estimate data noise from prediction-outcome variance."""
        if len(self._predictions) < 5:
            return 0.5
        errors = [abs(p - o) for p, o in zip(self._predictions, self._outcomes)]
        return float(np.mean(errors[-20:]))

    def _compute_epistemic(self) -> float:
        """Estimate model uncertainty from prediction spread."""
        if len(self._predictions) < 10:
            return 0.5
        preds = list(self._predictions)[-20:]
        return float(np.std(preds))

    def _compute_modifier(self, total_uncertainty: float) -> float:
        """Bounded modifier from uncertainty."""
        if total_uncertainty > 0.6:
            return -0.03  # reduce confidence
        if total_uncertainty < 0.2:
            return 0.01  # slight boost
        return 0.0
