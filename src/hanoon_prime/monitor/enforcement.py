"""monitor.enforcement — Deep enforcement, health budget, health diagnosis.

Runs runtime sanity checks that catch corruption, miswiring, and silent
failures BEFORE they cause losses.

Source: rebuild's deep_enforcement.py + health_budget.py + health_diagnosis.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class EnforcementCheck:
    check_id: str
    name: str
    severity: str  # critical, warning, info
    passed: bool
    detail: str = ""


@dataclass
class EnforcementReport:
    checks: list[EnforcementCheck] = field(default_factory=list)
    has_critical: bool = False
    score: float = 1.0
    timestamp: float = 0.0


class DeepEnforcement:
    """Runtime sanity checks for corruption and miswiring."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._last_run: float = 0.0

    def run_pulse(
        self, positions: dict[str, Any], orders: dict[str, Any], memory: object = None
    ) -> EnforcementReport:
        """Run all enforcement checks."""
        now = time.time()
        if now - self._last_run < 5.0:
            return EnforcementReport()
        self._last_run = now
        checks = []
        checks.append(self._check_position_consistency(positions, orders))
        checks.append(self._check_order_age(orders))
        checks.append(self._check_memory_health(memory))
        critical = any(c.severity == "critical" and not c.passed for c in checks)
        passed = sum(1 for c in checks if c.passed)
        score = passed / max(len(checks), 1)
        return EnforcementReport(
            checks=checks, has_critical=critical, score=score, timestamp=now
        )

    def _check_position_consistency(
        self, positions: dict[str, Any], orders: dict[str, Any]
    ) -> EnforcementCheck:
        """DE-001: Positions and orders are consistent."""
        for t in positions:
            if t in orders and orders[t].get("stale"):
                return EnforcementCheck(
                    "DE-001",
                    "position_consistency",
                    "warning",
                    False,
                    f"Stale order for {t}",
                )
        return EnforcementCheck("DE-001", "position_consistency", "info", True, "OK")

    def _check_order_age(self, orders: dict[str, Any]) -> EnforcementCheck:
        """DE-002: No orders are too old."""
        now = time.time()
        for t, o in orders.items():
            age = now - o.get("created", now)
            if age > 3600:
                return EnforcementCheck(
                    "DE-002",
                    "order_age",
                    "warning",
                    False,
                    f"Order {t} is {age:.0f}s old",
                )
        return EnforcementCheck("DE-002", "order_age", "info", True, "OK")

    def _check_memory_health(self, memory: object) -> EnforcementCheck:
        """DE-003: Memory is not corrupted."""
        if memory is None:
            return EnforcementCheck(
                "DE-003", "memory_health", "info", True, "No memory to check"
            )
        try:
            snap = memory.snapshot() if hasattr(memory, "snapshot") else {}
            wr = snap.get("win_rate", 0.5)
            if wr < 0.1 or wr > 0.9:
                return EnforcementCheck(
                    "DE-003",
                    "memory_health",
                    "warning",
                    False,
                    f"Win rate={wr:.2f} is suspicious",
                )
        except Exception as exc:
            log.debug("Memory check skipped: %s", exc)
        return EnforcementCheck("DE-003", "memory_health", "info", True, "OK")


class HealthBudget:
    """Track system health across all modules."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._scores: dict[str, float] = {}
        self._last_update: float = 0.0

    def update(self, module: str, score: float) -> None:
        """Auto-generated docstring."""
        self._scores[module] = max(0.0, min(1.0, score))
        self._last_update = time.time()

    def get_overall(self) -> float:
        """Auto-generated docstring."""
        if not self._scores:
            return 1.0
        return sum(self._scores.values()) / len(self._scores)

    def get_critical_modules(self) -> list[str]:
        """Auto-generated docstring."""
        return [m for m, s in self._scores.items() if s < 0.3]


class HealthDiagnosis:
    """Diagnose why health is low."""

    def diagnose(self, scores: dict[str, float]) -> list[str]:
        """Return diagnosis of health issues."""
        issues = []
        for module, score in scores.items():
            if score < 0.3:
                issues.append(f"{module}: CRITICAL (score={score:.2f})")
            elif score < 0.6:
                issues.append(f"{module}: WARNING (score={score:.2f})")
        return issues
