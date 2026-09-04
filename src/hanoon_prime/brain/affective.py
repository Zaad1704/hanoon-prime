"""hanoon_prime.brain.affective — sentiment, fear/greed, trading wisdom.

Quantifies market fear/greed from recent outcomes and adjusts the
score modifier. Bounded: ±AFFECTIVE_MOD_BOUND.

Merge of rebuild's emotion.py + sentiment.py + trading_wisdom.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import AFFECTIVE_MOD_BOUND


@dataclass
class AffectiveState:
    """Current emotional/fear-greed state."""

    fear: float = 0.0
    greed: float = 0.0
    modifier: float = 0.0
    streak: int = 0


class Affective:
    """Tracks fear/greed from recent trade outcomes."""

    def __init__(self, window: int = 20) -> None:
        self._window = window
        self._outcomes: deque[bool] = deque(maxlen=window)
        self._pnl_history: deque[float] = deque(maxlen=window)

    def update(self, won: bool, pnl_pct: float = 0.0) -> None:
        """Record a trade outcome for emotional state."""
        self._outcomes.append(won)
        self._pnl_history.append(pnl_pct)

    def evaluate(self) -> AffectiveState:
        """Compute current affective modifier."""
        if len(self._outcomes) < 3:
            return AffectiveState()
        recent = list(self._outcomes)
        wins = sum(recent)
        total = len(recent)
        wr = wins / total
        streak = self._compute_streak(recent)
        fear = self._compute_fear(wr, streak)
        greed = self._compute_greed(wr, streak)
        modifier = self._compute_modifier(fear, greed)
        return AffectiveState(
            fear=fear,
            greed=greed,
            modifier=modifier,
            streak=streak,
        )

    def _compute_streak(self, outcomes: list[bool]) -> int:
        """Current consecutive win/loss streak (positive=wins, negative=losses)."""
        if not outcomes:
            return 0
        last = outcomes[-1]
        count = 0
        for o in reversed(outcomes):
            if o == last:
                count += 1
            else:
                break
        return count if last else -count

    @staticmethod
    def _compute_fear(wr: float, streak: int) -> float:
        """Fear increases with losses. [0, 1]."""
        loss_component = max(0.0, (0.5 - wr) * 2.0)
        streak_component = max(0.0, min(1.0, abs(min(streak, 0)) / 3.0))
        return min(1.0, loss_component * 0.6 + streak_component * 0.4)

    @staticmethod
    def _compute_greed(wr: float, streak: int) -> float:
        """Greed increases with wins. [0, 1]."""
        win_component = max(0.0, (wr - 0.5) * 2.0)
        streak_component = max(0.0, min(1.0, max(streak, 0) / 3.0))
        return min(1.0, win_component * 0.6 + streak_component * 0.4)

    @staticmethod
    def _compute_modifier(fear: float, greed: float) -> float:
        """Convert fear/greed to bounded modifier."""
        raw = greed - fear
        return max(-AFFECTIVE_MOD_BOUND, min(AFFECTIVE_MOD_BOUND, raw))
