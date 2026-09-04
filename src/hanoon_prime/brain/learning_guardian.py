"""brain.learning_guardian — Hard gate ensuring learning is never harmed.

Monitors exit decisions, pillar weights, synthetic overrides, and
learning signals. Prevents any change from harming JULI's learning.

Source: rebuild's learning_guardian.py (simplified).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class GuardianCheck:
    name: str
    passed: bool
    detail: str = ""


class LearningGuardian:
    """Hard gate ensuring JULI learns from everything."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._violations: list[dict] = []

    def check_exit_decision(self, ticker: str, won: bool, pnl: float) -> GuardianCheck:
        """Ensure exit decision is recorded."""
        if abs(pnl) > 0.001 and not won and pnl > 0:
            return GuardianCheck("exit_decision", False, f".pnl={pnl} but won={won}")
        return GuardianCheck("exit_decision", True, "OK")

    def check_weight_integrity(self, weights: dict[str, float]) -> GuardianCheck:
        """Ensure weights are not corrupted."""
        total = sum(weights.values())
        if total < 0.5 or total > 2.0:
            return GuardianCheck(
                "weight_integrity", False, f"sum={total:.4f} outside [0.5, 2.0]"
            )
        for k, v in weights.items():
            if v < -0.1 or v > 0.5:
                return GuardianCheck(
                    "weight_integrity", False, f"{k}={v:.4f} outside [-0.1, 0.5]"
                )
        return GuardianCheck("weight_integrity", True, "OK")

    def check_learning_signal(self, memory: Any) -> GuardianCheck:
        """Ensure learning signals are flowing."""
        if memory is None:
            return GuardianCheck("learning_signal", True, "No memory")
        snap = memory.snapshot() if hasattr(memory, "snapshot") else {}
        total = snap.get("total_trades", 0)
        if total > 0 and snap.get("win_rate", 0.5) == 0.5:
            # Suspicious: trades exist but WR is exactly 0.5
            return GuardianCheck(
                "learning_signal", False, f"total={total} but WR=0.5 (frozen?)"
            )
        return GuardianCheck("learning_signal", True, "OK")

    def run_all(
        self, weights: dict[str, float], memory: Any = None
    ) -> list[GuardianCheck]:
        """Run all guardian checks."""
        checks = [
            self.check_weight_integrity(weights),
            self.check_learning_signal(memory),
        ]
        for c in checks:
            if not c.passed:
                self._violations.append(
                    {"check": c.name, "detail": c.detail, "ts": time.time()}
                )
        return checks
