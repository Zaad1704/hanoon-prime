"""hanoon_prime.brain.reflection — post-trade analysis and weight adaptation.

Single-writer for all brain learning state. On trade close:
1. Adapt indicator weights (asymmetric punishment)
2. Store episode in k-NN memory
3. Update prediction error calibration
4. Record lesson if pattern is notable

Merge of rebuild's self_reflection.py + performance_attribution.py + learning_mixin.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    LEARNING_RATE,
    PENALTY_SCALE,
    REWARD_SCALE,
    WEIGHT_DECAY,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from .episodic import EpisodicMemory
from .memory import JuliMemory


@dataclass
class TradeClose:
    """Bundle of data captured when a trade closes, for reflection."""

    ticker: str
    won: bool
    pnl_pct: float
    direction: int
    alpha: dict[str, float]
    predicted_score: float = 0.0
    regime: str = "unknown"


class Reflector:
    """Post-trade reflection and learning."""

    def __init__(self, memory: JuliMemory, episodic: EpisodicMemory) -> None:
        self._memory = memory
        self._episodic = episodic

    def on_trade_close(self, trade: TradeClose) -> None:
        """Full reflection pipeline on trade close."""
        tick = trade.ticker
        won = trade.won
        pnl_pct = trade.pnl_pct
        direction = trade.direction
        alpha = trade.alpha
        predicted_score = trade.predicted_score
        self._adapt_weights(won, direction, alpha)
        outcome = pnl_pct if direction > 0 else -pnl_pct
        # NOTE: episodic k-NN is written once by the orchestrator (the
        # single writer for per-close episodes) — do not add here too,
        # or real closes count double in recall.
        self._memory.add_episode(
            [alpha.get(k, 0.5) for k in list(alpha.keys())[:11]],
            outcome,
            tick,
        )
        self._memory.update_pred_error(predicted_score, 1.0 if won else 0.0)
        self._memory.record_outcome(won)
        self._memory.record_score(tick, predicted_score)
        if abs(pnl_pct) > 0.05:
            self._memory.add_lesson(
                {
                    "ticker": tick,
                    "won": won,
                    "pnl_pct": pnl_pct,
                    "regime": trade.regime,
                    "pattern": f"{'win' if won else 'loss'}_{abs(pnl_pct):.1%}",
                }
            )

    def _adapt_weights(
        self, won: bool, direction: int, alpha: dict[str, float]
    ) -> None:
        """Asymmetric weight update from trade outcome."""
        weights = self._memory.get_weights()
        factor = REWARD_SCALE if won else -PENALTY_SCALE
        for key in weights:
            signal_val = alpha.get(key, 0.0)
            delta = LEARNING_RATE * factor * signal_val * direction
            weights[key] = max(WEIGHT_MIN, min(WEIGHT_MAX, weights[key] + delta))
        for key in weights:
            weights[key] *= WEIGHT_DECAY
        self._memory.set_weights(weights)
