"""hanoon_prime.brain.guardians — safety limits, circuit breakers, learning stability.

Prevents the brain from degrading: checks weight health, learning
stability, and circuit-breaker conditions.

Merge of rebuild's brain_enforcer.py + learning_guardian.py + regression_guard.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    CIRCUIT_BREAKER_THRESHOLD,
    MIN_ACTIVE_INDICATORS,
    WEIGHT_MAX,
    WEIGHT_MIN,
    MAX_WEIGHT_Drift,
)


@dataclass
class GuardianVerdict:
    """Safety check result."""

    safe: bool = True
    reason: str = ""
    actions: list[str] | None = None


class Guardians:
    """System health checks and safety limits."""

    def __init__(self) -> None:
        self._consecutive_failures: int = 0

    def check_weights(self, weights: dict[str, float]) -> GuardianVerdict:
        """Check if weights are healthy (not all decayed)."""
        active = sum(1 for w in weights.values() if abs(w) > 0.01)
        total = sum(abs(v) for v in weights.values())
        if active < MIN_ACTIVE_INDICATORS:
            self._consecutive_failures += 1
            return GuardianVerdict(
                safe=False,
                reason=f"Only {active}/{MIN_ACTIVE_INDICATORS} active indicators",
                actions=["reset_weights"],
            )
        if total < 0.1:
            self._consecutive_failures += 1
            return GuardianVerdict(
                safe=False,
                reason="Total weight too low",
                actions=["reset_weights"],
            )
        self._consecutive_failures = 0
        return GuardianVerdict()

    def check_learning_stability(self, pred_error: float) -> GuardianVerdict:
        """Check if learning is diverging."""
        if pred_error > CIRCUIT_BREAKER_THRESHOLD:
            return GuardianVerdict(
                safe=False,
                reason=f"Pred error {pred_error:.3f} > {CIRCUIT_BREAKER_THRESHOLD}",
                actions=["slow_learning"],
            )
        return GuardianVerdict()

    def check_drift(
        self, weights: dict[str, float], baseline: dict[str, float]
    ) -> GuardianVerdict:
        """Check if weights drifted too far from baseline."""
        for key in weights:
            if key in baseline:
                drift = abs(weights[key] - baseline[key])
                if drift > MAX_WEIGHT_Drift:
                    return GuardianVerdict(
                        safe=False,
                        reason=f"Weight {key} drifted {drift:.3f}",
                        actions=["clamp_weight"],
                    )
        return GuardianVerdict()
