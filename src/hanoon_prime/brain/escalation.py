"""brain.escalation — Compute escalation for deeper deliberation.

When stakes are high (borderline score) or uncertainty is high,
allocate more compute for wider evidence packs.

Source: rebuild's escalation.py (simplified).
"""

from __future__ import annotations

from dataclasses import dataclass

_LEVEL_MAX: int = 3
_TOP_K_STEP: int = 2
_HOSTILE_REGIMES = frozenset({"ranging", "volatile", "choppy"})


@dataclass
class EscalationState:
    level: int = 0
    top_k: int = 5
    reason: str = ""


class Escalation:
    """Compute escalation for deeper deliberation."""

    def __init__(self, base_top_k: int = 5) -> None:
        """Auto-generated docstring."""
        self._base_top_k = base_top_k

    def compute(
        self,
        score: float,
        threshold: float,
        regime: str = "normal",
        uncertainty: float = 0.0,
    ) -> EscalationState:
        """Determine escalation level."""
        level = 0
        reasons = []
        # Borderline score
        margin = abs(abs(score) - threshold)
        if margin < 0.1:
            level += 1
            reasons.append("borderline")
        # Hostile regime
        if regime in _HOSTILE_REGIMES:
            level += 1
            reasons.append("hostile_regime")
        # High uncertainty
        if uncertainty > 0.5:
            level += 1
            reasons.append("high_uncertainty")
        level = min(level, _LEVEL_MAX)
        top_k = self._base_top_k + level * _TOP_K_STEP
        return EscalationState(level=level, top_k=top_k, reason="+".join(reasons))
