"""hanoon_prime.brain.plasticity — Online weight adaptation.

Dynamically adjusts the balance of brain module outputs based on
live performance feedback. Weights are updated on each trade
resolution using exponential reinforcement.

Biological analogy: Synaptic pruning — strengthens successful
pathways, weakens failing ones.
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)

MODULES: tuple[str, ...] = ("amygdala", "pfc", "hippocampus")
INITIAL_WEIGHT: float = 1.0 / 3.0
LEARNING_RATE: float = 0.05
WEIGHT_MIN: float = 0.05
WEIGHT_MAX: float = 0.90


class Plasticity:
    """Synaptic weight adaptation engine."""

    def __init__(self) -> None:
        self.weights: dict[str, float] = {m: INITIAL_WEIGHT for m in MODULES}
        self._performance: dict[str, list[float]] = {m: [] for m in MODULES}

    def synthesize(
        self,
        signals: dict[str, float],
    ) -> float:
        """Combine module signals using current synaptic weights."""
        score = 0.0
        for module, signal in signals.items():
            weight = self.weights.get(module, 0.0)
            score += weight * signal
        return max(-1.0, min(1.0, score))

    def adapt(
        self,
        module_scores: dict[str, float],
        trade_result: float,
    ) -> None:
        """Update weights based on trade outcome."""
        for module, score in module_scores.items():
            if module not in self.weights:
                continue
            perf = score * trade_result
            self._performance[module].append(perf)
            if len(self._performance[module]) > 100:
                self._performance[module] = self._performance[module][-100:]
            avg_perf = sum(self._performance[module][-20:]) / max(
                len(self._performance[module][-20:]),
                1,
            )
            self.weights[module] *= math.exp(LEARNING_RATE * avg_perf)
        self._normalize()

    def _normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total <= 0:
            self.reset()
            return
        for module in self.weights:
            self.weights[module] = max(
                WEIGHT_MIN,
                min(WEIGHT_MAX, self.weights[module] / total),
            )

    def reset(self) -> None:
        """Reset all weights to uniform."""
        for module in self.weights:
            self.weights[module] = INITIAL_WEIGHT

    def get_weights(self) -> dict[str, float]:
        """Return current weight snapshot."""
        return dict(self.weights)
