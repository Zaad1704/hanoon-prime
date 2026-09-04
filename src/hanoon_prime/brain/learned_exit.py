"""brain.learned_exit — Learned exit policy.

Derives exit health floor and trigger weights from historical trades.
Until enough labeled trades accumulate, stays neutral.

Source: rebuild's learned_exit.py (simplified).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)

_MIN_TRADES: int = 20


@dataclass
class ExitPolicy:
    health_floor: float = 0.0
    trigger_weights: Optional[dict[str, float]] = None

    def __post_init__(self) -> None:
        """Auto-generated docstring."""
        if self.trigger_weights is None:
            self.trigger_weights = {}


class LearnedExitPolicy:
    """Derive exit triggers from historical trades."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._trades: deque[dict[str, Any]] = deque(maxlen=200)

    def record(self, health_score: float, triggers: list[str], won: bool) -> None:
        """Record exit outcome for learning."""
        self._trades.append({"health": health_score, "triggers": triggers, "won": won})

    def get_policy(self) -> ExitPolicy:
        """Get learned exit policy."""
        if len(self._trades) < _MIN_TRADES:
            return ExitPolicy()  # neutral
        # Compute health floor: health level below which WR < 50%
        sorted_t = sorted(self._trades, key=lambda t: t["health"])
        n = len(sorted_t)
        bottom_half = sorted_t[: n // 2]
        top_half = sorted_t[n // 2 :]
        wr_bottom = sum(1 for t in bottom_half if t["won"]) / max(len(bottom_half), 1)
        wr_top = sum(1 for t in top_half if t["won"]) / max(len(top_half), 1)
        health_floor = 0.0
        if wr_bottom < 0.5 and wr_top >= 0.5:
            health_floor = bottom_half[-1]["health"] if bottom_half else 0.0
        # Compute trigger weights
        trigger_stats: dict[str, dict[str, int]] = {}
        for t in self._trades:
            for trig in t["triggers"]:
                if trig not in trigger_stats:
                    trigger_stats[trig] = {"before_win": 0, "before_loss": 0}
                if t["won"]:
                    trigger_stats[trig]["before_win"] += 1
                else:
                    trigger_stats[trig]["before_loss"] += 1
        weights = {}
        for trig, stats in trigger_stats.items():
            total = stats["before_win"] + stats["before_loss"]
            if total > 0:
                wr = stats["before_win"] / total
                weights[trig] = wr - 0.5  # positive = good trigger
        return ExitPolicy(health_floor=health_floor, trigger_weights=weights)
