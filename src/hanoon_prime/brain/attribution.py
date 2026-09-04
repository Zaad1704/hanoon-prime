"""brain.attribution — Performance attribution, causal, direction, brain enforcer.

Attributes PnL to specific signals. Tracks causal relationships.
Monitors direction experience. Enforces brain invariants.

Source: rebuild's performance_attribution.py + causal_derivative.py +
       direction_exp.py + brain_enforcer.py (all simplified).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    ticker: str
    pnl: float
    signal_contributions: dict[str, float]
    dominant_signal: str = ""


class PerformanceAttribution:
    """Attribute PnL to specific signals."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._history: deque[dict] = deque(maxlen=500)

    def attribute(
        self, ticker: str, pnl: float, alpha: dict[str, float]
    ) -> AttributionResult:
        """Attribute PnL to signal contributions."""
        contributions = {}
        for k, v in alpha.items():
            contributions[k] = v * pnl * 0.1  # simplified attribution
        dominant = max(contributions, key=lambda k: abs(contributions[k]))
        self._history.append({"ticker": ticker, "pnl": pnl, "alpha": dict(alpha)})
        return AttributionResult(ticker, pnl, contributions, dominant)

    def get_signal_stats(self) -> dict[str, dict]:
        """Get aggregated signal performance."""
        stats: dict[str, dict] = {}
        for entry in self._history:
            for k, v in entry.get("alpha", {}).items():
                if k not in stats:
                    stats[k] = {"total_pnl": 0, "n": 0}
                stats[k]["total_pnl"] += v * entry["pnl"] * 0.1
                stats[k]["n"] += 1
        return stats


class CausalDerivative:
    """Track causal relationships between signals."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._pairs: dict[str, deque[float]] = {}

    def update(self, signal_a: str, signal_b: str, value: float) -> None:
        """Auto-generated docstring."""
        key = f"{signal_a}_{signal_b}"
        self._pairs.setdefault(key, deque(maxlen=100)).append(value)

    def get_causality(self, signal_a: str, signal_b: str) -> float:
        """Auto-generated docstring."""
        key = f"{signal_a}_{signal_b}"
        data = self._pairs.get(key, deque())
        if len(data) < 10:
            return 0.0
        import numpy as np

        arr = np.array(list(data))
        return float(np.mean(arr))


class DirectionExperience:
    """Track direction-specific experience."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._long_wr: deque[bool] = deque(maxlen=200)
        self._short_wr: deque[bool] = deque(maxlen=200)

    def record(self, direction: int, won: bool) -> None:
        """Auto-generated docstring."""
        if direction > 0:
            self._long_wr.append(won)
        else:
            self._short_wr.append(won)

    def get_wr(self, direction: int) -> float:
        """Auto-generated docstring."""
        data = self._long_wr if direction > 0 else self._short_wr
        if len(data) < 5:
            return 0.5
        return sum(data) / len(data)


class BrainEnforcer:
    """Enforce brain invariants at runtime."""

    def check_all(
        self, weights: dict[str, float], threshold: float, score: float
    ) -> list[dict]:
        """Run all brain enforcement checks."""
        issues = []
        total = sum(weights.values())
        if total < 0.5 or total > 2.0:
            issues.append(
                {"check": "weight_sum", "passed": False, "detail": f"sum={total:.4f}"}
            )
        if threshold < 0.05 or threshold > 0.95:
            issues.append(
                {
                    "check": "threshold_bounds",
                    "passed": False,
                    "detail": f"threshold={threshold}",
                }
            )
        if abs(score) > 2.0:
            issues.append(
                {
                    "check": "score_bounds",
                    "passed": False,
                    "detail": f"score={score:.4f}",
                }
            )
        return issues
