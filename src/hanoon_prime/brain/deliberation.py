"""hanoon_prime.brain.deliberation — bidirectional deliberation engine.

Takes the base Thought from cortex and cross-evaluates against:
- Episodic memory (have I seen this pattern?)
- Affective state (fear/greed)
- Salience (noise/uncertainty)
- Regime (market context)
- HALIM advisory (external AI)

All modifiers BOUNDED — no single module dominates.
Outputs a refined score and direction.

Merge of rebuild's thinker.py + bidirectional_deliberation.py + consensus.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import (
    AFFECTIVE_MOD_BOUND,
    CONFIDENCE_FLOOR,
    EPISODIC_MOD_BOUND,
    HALIM_MOD_BOUND,
    SIGNAL_THRESHOLD,
)


@dataclass
class DeliberationResult:
    """Output of the deliberation engine."""

    score: float = 0.0
    direction: int = 0
    confidence: float = 0.5
    verdict: str = "HOLD"
    trace: dict[str, float] = field(default_factory=dict)


class Deliberator:
    """Bidirectional deliberation — synthesizes all brain signals."""

    def __init__(self, threshold: float = SIGNAL_THRESHOLD) -> None:
        self._threshold = threshold

    def deliberate(
        self,
        base_score: float,
        base_confidence: float,
        episodic_mod: float = 0.0,
        affective_mod: float = 0.0,
        salience_atten: float = 1.0,
        regime_mult: float = 1.0,
        halim_mod: float = 0.0,
    ) -> DeliberationResult:
        """Synthesize all modifiers into final decision."""
        trace: dict[str, float] = {
            "base_score": base_score,
            "episodic_mod": episodic_mod,
            "affective_mod": affective_mod,
            "halim_mod": halim_mod,
            "salience_atten": salience_atten,
            "regime_mult": regime_mult,
        }
        raw = base_score
        raw += self._bound(episodic_mod, EPISODIC_MOD_BOUND)
        raw += self._bound(affective_mod, AFFECTIVE_MOD_BOUND)
        raw += self._bound(halim_mod, HALIM_MOD_BOUND)
        raw *= regime_mult
        raw *= salience_atten
        raw = max(-1.0, min(1.0, raw))
        confidence = base_confidence * salience_atten
        confidence = max(CONFIDENCE_FLOOR, min(0.95, confidence))
        direction = self._direction(raw)
        verdict = self._verdict(raw, direction)
        trace["final_score"] = raw
        trace["final_confidence"] = confidence
        return DeliberationResult(
            score=round(raw, 4),
            direction=direction,
            confidence=round(confidence, 4),
            verdict=verdict,
            trace=trace,
        )

    def _direction(self, score: float) -> int:
        if abs(score) < 0.1:
            return 0
        return 1 if score > 0 else -1

    def _verdict(self, score: float, direction: int) -> str:
        if abs(score) < self._threshold:
            return "HOLD"
        if direction > 0:
            return "BUY"
        if direction < 0:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _bound(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))
