"""monitor.decision_health — Track decision quality over time.

Monitors the quality of entry/exit decisions by tracking outcomes.
Detects degradation and alerts.

Source: rebuild's decision_health.py (simplified).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class DecisionHealth:
    overall_score: float = 1.0
    entry_quality: float = 0.5
    exit_quality: float = 0.5
    recent_accuracy: float = 0.5
    n_decisions: int = 0
    issues: Optional[list[str]] = None

    def __post_init__(self) -> None:
        """Auto-generated docstring."""
        if self.issues is None:
            self.issues = []


class DecisionHealthTracker:
    """Track decision quality over time."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._entries: deque[dict[str, Any]] = deque(maxlen=200)
        self._exits: deque[dict[str, Any]] = deque(maxlen=200)

    def record_entry(self, ticker: str, score: float, won: bool) -> None:
        """Auto-generated docstring."""
        self._entries.append({"ticker": ticker, "score": score, "won": won})

    def record_exit(self, ticker: str, reason: str, won: bool) -> None:
        """Auto-generated docstring."""
        self._exits.append({"ticker": ticker, "reason": reason, "won": won})

    def get_health(self) -> DecisionHealth:
        """Auto-generated docstring."""
        if not self._entries and not self._exits:
            return DecisionHealth()
        entry_wr = sum(1 for e in self._entries if e["won"]) / max(
            len(self._entries), 1
        )
        exit_wr = sum(1 for e in self._exits if e["won"]) / max(len(self._exits), 1)
        recent = list(self._entries)[-20:]
        recent_wr = sum(1 for e in recent if e["won"]) / max(len(recent), 1)
        issues = []
        if entry_wr < 0.40:
            issues.append(f"Entry WR={entry_wr:.2f} < 0.40")
        if exit_wr < 0.40:
            issues.append(f"Exit WR={exit_wr:.2f} < 0.40")
        score = (entry_wr + exit_wr) / 2
        return DecisionHealth(
            overall_score=score,
            entry_quality=entry_wr,
            exit_quality=exit_wr,
            recent_accuracy=recent_wr,
            n_decisions=len(self._entries),
            issues=issues,
        )
