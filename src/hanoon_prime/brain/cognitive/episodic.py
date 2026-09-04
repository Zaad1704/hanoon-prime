"""hanoon_prime.brain.cognitive.episodic — Episodic memory (k-NN recall).

Recalls similar past situations from memory. When JULI sees a familiar
pattern, it recalls what happened last time and biases the current
decision accordingly. Modifier bounded by EPISODIC_MOD_BOUND (±0.03).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

MOD_BOUND: float = 0.03
K: int = 7
MAX_EPISODES: int = 2000


@dataclass
class Episode:
    """A single recorded trade situation."""

    alpha: dict[str, float]
    won: bool
    pnl_pct: float
    regime: str = "unknown"


class EpisodicMemory:
    """k-NN recall of similar past situations."""

    def __init__(self) -> None:
        self._episodes: list[Episode] = []
        self.size: int = 0

    def add(
        self,
        alpha: dict[str, float],
        won: bool,
        pnl_pct: float,
        regime: str = "unknown",
    ) -> None:
        """Record a trade outcome for future recall."""
        self._episodes.append(
            Episode(alpha=alpha, won=won, pnl_pct=pnl_pct, regime=regime)
        )
        self.size = len(self._episodes)
        if self.size > MAX_EPISODES:
            self._episodes = self._episodes[-MAX_EPISODES:]
            self.size = MAX_EPISODES

    def recall(self, alpha: dict[str, float]) -> float | None:
        """Find k most similar episodes and return bias modifier.

        Returns modifier in [-MOD_BOUND, +MOD_BOUND] or None if too few episodes.
        """
        if len(self._episodes) < K:
            return None
        distances: list[tuple[float, Episode]] = []
        for ep in self._episodes:
            d = self._distance(alpha, ep.alpha)
            distances.append((d, ep))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:K]
        if not neighbors:
            return None
        avg_pnl = sum(ep.pnl_pct for _, ep in neighbors) / len(neighbors)
        bias = max(-1.0, min(1.0, avg_pnl * 10.0))
        return max(-MOD_BOUND, min(MOD_BOUND, bias * 0.3))

    @staticmethod
    def _distance(a: dict[str, float], b: dict[str, float]) -> float:
        """Euclidean distance between two alpha dicts."""
        keys = set(a.keys()) | set(b.keys())
        if not keys:
            return 0.0
        sq_sum = sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys)
        return math.sqrt(float(sq_sum))

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view."""
        return {"episodes": self.size, "max": MAX_EPISODES}
