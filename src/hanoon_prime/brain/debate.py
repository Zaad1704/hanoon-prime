"""brain.debate — Multi-Agent Debate Layer for Entry Qualification.

Following the TradingAgents pattern, runs a structured 3-agent debate:
1. BULL — argues FOR entering
2. BEAR — argues AGAINST
3. ARBITER — weighs both, emits APPROVE/REJECT with reasoning

Source: rebuild's debate.py (simplified — uses scoring, not HALIM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

_DEBATE_TTL: float = 30.0  # seconds


@dataclass
class DebateVerdict:
    approved: bool
    confidence: float
    bull_score: float
    bear_score: float
    reasoning: str


class DebateLayer:
    """3-agent debate for entry qualification."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._cache: dict[str, tuple[float, DebateVerdict]] = {}

    def debate(
        self,
        ticker: str,
        alpha: dict[str, float],
        score: float,
        regime: str = "normal",
        ev: float = 0.0,
        r_r: float = 3.0,
    ) -> DebateVerdict:
        """Run BULL/BEAR/ARBITER debate."""
        import time

        now = time.time()
        cached = self._cache.get(ticker)
        if cached and now - cached[0] < _DEBATE_TTL:
            return cached[1]

        bull = self._bull_case(alpha, score, ev, r_r)
        bear = self._bear_case(alpha, score, regime)
        verdict = self._arbitrate(bull, bear, score, regime)
        self._cache[ticker] = (now, verdict)
        return verdict

    def _bull_case(
        self, alpha: dict[str, float], score: float, ev: float, r_r: float
    ) -> float:
        """Score the bullish case."""
        strength = 0.0
        if ev > 0.05:
            strength += 0.3
        if r_r >= 3.0:
            strength += 0.2
        if score > 0.5:
            strength += 0.3
        # Momentum indicators
        for k in ("momentum", "adx", "rsi"):
            v = alpha.get(k, 0)
            if v > 0.6:
                strength += 0.05
        return min(1.0, strength)

    def _bear_case(self, alpha: dict[str, float], score: float, regime: str) -> float:
        """Score the bearish case."""
        weakness = 0.0
        if score < 0.3:
            weakness += 0.3
        if regime in ("volatile", "choppy"):
            weakness += 0.2
        # Risk indicators
        for k in ("vpin", "spread_tightness"):
            v = alpha.get(k, 0)
            if v > 0.8:
                weakness += 0.05
        return min(1.0, weakness)

    def _arbitrate(
        self, bull: float, bear: float, score: float, regime: str
    ) -> DebateVerdict:
        """Arbiter weighs both cases."""
        net = bull - bear
        approved = net > 0.1 and abs(score) > 0.3
        confidence = 0.5 + net * 0.3
        confidence = max(0.1, min(0.9, confidence))
        reasoning = f"BULL={bull:.2f} BEAR={bear:.2f} NET={net:.2f} " f"regime={regime}"
        return DebateVerdict(
            approved=approved,
            confidence=confidence,
            bull_score=bull,
            bear_score=bear,
            reasoning=reasoning,
        )

    @property
    def modifier(self) -> float:
        """Bounded modifier from last debate."""
        return 0.0  # debate affects approval, not score
