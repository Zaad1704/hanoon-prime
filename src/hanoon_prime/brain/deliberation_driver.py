"""hanoon_prime.brain.deliberation_driver — bridge the bounded Deliberator.

deliberation.py holds Juli's bounded chain-of-thought engine: it fuses
episodic/affective/HALIM/salience/regime modifiers into a refined score with an
attributable ``trace`` (the per-contributor CoT). The live cortex already
federates these modifiers, but without a trace. This driver runs the
Deliberator over the live cycle's bounded signals so the CoT is observable per
decision — strictly diagnostic (``DELIBERATION_TRACE_ENABLED``): candidate
score is published, NEVER blended into the verdict, so R1 (cortex is the sole
verdict emitter) holds. Flip a future opt-in to *adopt* the candidate.
"""
from __future__ import annotations

from ..cortex import Thought
from .deliberation import DeliberationResult, Deliberator, Modifiers


class DeliberationDriver:
    """Run the bounded Deliberator CoT over a live cycle's signals.

    Maps the orchestrator's fast-path signals onto the Deliberator's bounded
    modifier slots. Each mod is clamped by the Deliberator (episodic ±0.10,
    affective ±0.05, halim ±0.03) — the same per-mod bounds the fast path
    uses — so the trace reflects a faithful, attributable chain of thought.
    """

    def __init__(self) -> None:
        self._deliberator = Deliberator()

    def compose(
        self,
        base: Thought,
        regime_mul: float,
        halim: float,
        episodic: float,
        thinker_mod: float,
        news_bias: float,
    ) -> DeliberationResult:
        """Compose a bounded, traced deliberation over the cycle's signals."""
        mods = Modifiers(
            episodic_mod=episodic,
            affective_mod=thinker_mod,
            halim_mod=halim,
            salience_atten=1.0 + news_bias,
            regime_mult=regime_mul,
        )
        return self._deliberator.deliberate(base.score, base.confidence, mods)


__all__ = ["DeliberationDriver"]
