"""hanoon_prime.brain.thinker — Bidirectional deliberation.

Fuses the 5 cognitive pillars (semantic, episodic, emotion, planning,
metacognition) + Nash arbitration into a single bounded modifier.
No single pillar can dominate. The thinker is stateless across ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import BarSeries
from .cognitive.emotion import EmotionState
from .cognitive.episodic import EpisodicMemory
from .cognitive.metacognition import Metacognition
from .cognitive.nash import NashBrain as NashArbitrator
from .cognitive.planning import PlanEngine
from .cognitive.semantic import SemanticMemory

TOTAL_MOD_BOUND: float = 0.06


@dataclass
class ThinkingResult:
    """Output of bidirectional deliberation."""

    modifier: float = 0.0
    confidence_mod: float = 0.0
    risk_scalar: float = 1.0
    trace: dict[str, float] = field(default_factory=dict)


@dataclass
class Signal:
    """Context bundled with a think call: regime, price series, threshold."""

    regime: str = "unknown"
    bars: BarSeries | None = None
    threshold: float = 0.58


class Thinker:
    """Bidirectional deliberation engine — fuses cognitive pillars."""

    def __init__(self) -> None:
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.emotion = EmotionState()
        self.planning = PlanEngine()
        self.metacognition = Metacognition()
        self.nash = NashArbitrator()

    def think(
        self,
        alpha: dict[str, float],
        score: float,
        direction: int,
        signal: Signal | None = None,
    ) -> ThinkingResult:
        """Run deliberation across all pillars."""
        sig = signal or Signal()
        mods = self._compute_mods(alpha, score, direction, sig)
        total = max(-TOTAL_MOD_BOUND, min(TOTAL_MOD_BOUND, sum(mods.values())))
        return ThinkingResult(
            modifier=round(total, 6),
            confidence_mod=round(self.emotion.confidence_mod(), 4),
            risk_scalar=round(self.emotion.risk_scalar(), 4),
            trace={k: round(v, 6) for k, v in mods.items()},
        )

    def _compute_mods(
        self,
        alpha: dict[str, float],
        score: float,
        direction: int,
        signal: Signal,
    ) -> dict[str, float]:
        m: dict[str, float] = {}
        m["semantic"] = self.semantic.evaluate(alpha, signal.regime)
        epi = self.episodic.recall(alpha)
        m["episodic"] = epi if epi is not None else 0.0
        bars = signal.bars
        m["planning"] = (
            self.planning.simulate(bars.close, bars.high, bars.low, direction)
            if bars is not None and direction != 0
            else 0.0
        )
        m["metacognition"] = self.metacognition.evaluate(score, signal.threshold)
        m["nash"] = self.nash.evaluate(alpha, direction)
        return m
