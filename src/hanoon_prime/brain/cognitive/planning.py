"""hanoon_prime.brain.cognitive.planning — Plan Engine (Monte Carlo sim).

Simulates future price paths to estimate trade viability before entry.
Uses recent volatility to generate synthetic paths and checks if the
target is reachable before the stop is hit. Modifier bounded ±0.03.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

MOD_BOUND: float = 0.03
N_SIMS: int = 100
HORIZON: int = 20
TARGET_MULT: float = 6.0
STOP_MULT: float = 2.0


class PlanEngine:
    """Monte Carlo forward simulation for trade planning."""

    def simulate(self, close: Any, high: Any, low: Any, direction: int = 1) -> float:
        """Simulate forward paths, return bounded modifier."""
        mu, sigma, current, atr = self._extract_params(close, high, low)
        if sigma <= 0 or atr <= 0:
            return 0.0
        stop_dist = STOP_MULT * atr
        target_dist = TARGET_MULT * atr
        wins = self._run_sims(mu, sigma, current, stop_dist, target_dist, direction)
        bias = (wins / N_SIMS - 0.5) * 2.0
        return max(-MOD_BOUND, min(MOD_BOUND, bias * 0.5))

    @staticmethod
    def _extract_params(
        close: Any, high: Any, low: Any
    ) -> tuple[float, float, float, float]:
        c = np.asarray(close, dtype=float)
        if len(c) < 10:
            return 0.0, 0.0, 0.0, 0.0
        returns = np.diff(np.log(c[-20:]))
        h, l = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
        atr = float(np.mean(h[-14:] - l[-14:])) if len(h) >= 14 else 0.0
        return float(np.mean(returns)), float(np.std(returns)), float(c[-1]), atr

    @staticmethod
    def _run_sims(
        mu: float,
        sigma: float,
        current: float,
        stop_dist: float,
        target_dist: float,
        direction: int,
    ) -> int:
        wins = 0
        for _ in range(N_SIMS):
            price = current
            for r in np.asarray(np.random.normal(mu, sigma, HORIZON)):
                price *= np.exp(r)
                move = (price - current) * direction
                if move >= target_dist:
                    wins += 1
                    break
                if move <= -stop_dist:
                    break
        return wins
