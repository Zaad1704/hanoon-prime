"""brain.weight_enforcer — Weight integrity enforcer.

Prevents weight drift and repairs corruption. Three defense layers:
1. ADAPT GUARD: after every N adapt calls, normalize weights to sum=1.0
2. PER-ADAPT CAP: no single weight may exceed MAX_WEIGHT
3. STARTUP REPAIR: on _load, detect and fix corrupted weights

Source: rebuild's weight_enforcer.py (lines 1-193).
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import DEFAULT_WEIGHTS

log = logging.getLogger(__name__)

MAX_WEIGHT: float = 0.20
MIN_WEIGHT_SUM: float = 0.80
MAX_WEIGHT_SUM: float = 1.50
NORMALIZE_EVERY_N: int = 25
WEIGHT_FLOOR: float = 0.01


class WeightEnforcer:
    """Weight integrity enforcer — prevents drift and repairs corruption."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._adapt_count: int = 0

    def on_adapt(self, weights: dict[str, float], indicator: str) -> None:
        """Called after every adapt. Enforces caps + periodic normalization."""
        self._adapt_count += 1
        current = weights.get(indicator, 0.01)
        if current > MAX_WEIGHT:
            weights[indicator] = MAX_WEIGHT
            log.debug("Capped %s from %.4f → %.4f", indicator, current, MAX_WEIGHT)
        if self._adapt_count % NORMALIZE_EVERY_N == 0:
            self._normalize(weights)

    def _normalize(self, weights: dict[str, float]) -> None:
        """Proportionally normalize weights to sum to 1.0."""
        if not weights:
            return
        total = sum(weights.values())
        if total <= 0:
            log.warning("All weights zero — resetting to defaults")
            weights.update(DEFAULT_WEIGHTS)
            return
        if 0.90 <= total <= 1.10:
            return  # within 10% of target
        scale = 1.0 / total
        for k in weights:
            weights[k] *= scale
        for k in weights:
            if weights[k] < WEIGHT_FLOOR:
                weights[k] = WEIGHT_FLOOR
        log.debug("Normalized weights (sum %.4f → %.4f)", total, sum(weights.values()))

    def repair_on_load(self, weights: dict[str, float]) -> bool:
        """Repair weights on memory load. Returns True if repair was needed."""
        if not weights:
            return False
        repaired = False
        for k in list(weights.keys()):
            if weights[k] > MAX_WEIGHT:
                weights[k] = MAX_WEIGHT
                repaired = True
            if weights[k] < WEIGHT_FLOOR:
                weights[k] = WEIGHT_FLOOR
                repaired = True
        for k, v in DEFAULT_WEIGHTS.items():
            if k not in weights:
                weights[k] = v
                repaired = True
        for k in list(weights.keys()):
            if k not in DEFAULT_WEIGHTS:
                del weights[k]
                repaired = True
        total = sum(weights.values())
        if total < 0.90 or total > 1.10:
            self._normalize(weights)
            repaired = True
        if repaired:
            log.info("Weight repair complete")
        return repaired

    def check_integrity(self, weights: dict[str, float]) -> dict:
        """Check weight integrity without modifying. Returns diagnostic."""
        if not weights:
            return {"ok": True, "reason": "empty"}
        total = sum(weights.values())
        max_w = max(weights.values())
        min_w = min(weights.values())
        issues = []
        if total < MIN_WEIGHT_SUM:
            issues.append(f"sum={total:.4f} < {MIN_WEIGHT_SUM}")
        elif total > MAX_WEIGHT_SUM:
            issues.append(f"sum={total:.4f} > {MAX_WEIGHT_SUM}")
        if max_w > MAX_WEIGHT:
            issues.append(f"max={max_w:.4f} > {MAX_WEIGHT}")
        if min_w < WEIGHT_FLOOR:
            issues.append(f"min={min_w:.6f} < {WEIGHT_FLOOR}")
        return {
            "ok": len(issues) == 0,
            "sum": total,
            "max": max_w,
            "min": min_w,
            "issues": issues,
        }


_enforcer: Optional[WeightEnforcer] = None


def get_enforcer() -> WeightEnforcer:
    """Get the singleton enforcer instance."""
    global _enforcer
    if _enforcer is None:
        _enforcer = WeightEnforcer()
    return _enforcer
