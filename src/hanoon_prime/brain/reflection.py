"""hanoon_prime.brain.reflection — post-trade analysis and weight adaptation.

Single-writer for all brain learning state. On trade close:
1. Adapt indicator weights (asymmetric punishment)
2. Store episode in k-NN memory
3. Update prediction error calibration
4. Record lesson if pattern is notable

Merge of rebuild's self_reflection.py + performance_attribution.py + learning_mixin.py.
"""

from __future__ import annotations

from typing import Any

from .config import (
    DEFAULT_WEIGHTS,
    LEARNING_RATE,
    PENALTY_SCALE,
    PRED_ERR_EMA_ALPHA,
    PRED_ERR_MIN_SAMPLES,
    REWARD_SCALE,
    WEIGHT_DECAY,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from .episodic import EpisodicMemory
from .memory import JuliMemory


class Reflector:
    """Post-trade reflection and learning."""

    def __init__(self, memory: JuliMemory, episodic: EpisodicMemory) -> None:
        self._memory = memory
        self._episodic = episodic

    def on_trade_close(
        self,
        ticker: str,
        won: bool,
        pnl_pct: float,
        direction: int,
        alpha: dict[str, float],
        predicted_score: float = 0.0,
    ) -> None:
        """Full reflection pipeline on trade close."""
        self._adapt_weights(won, direction, alpha)
        outcome = pnl_pct if direction > 0 else -pnl_pct
        self._episodic.add(alpha, outcome)
        self._memory.add_episode(
            [alpha.get(k, 0.5) for k in list(alpha.keys())[:11]],
            outcome,
            ticker,
        )
        self._memory.update_pred_error(predicted_score, 1.0 if won else 0.0)
        self._memory.record_outcome(won)
        self._memory.record_score(ticker, predicted_score)
        if abs(pnl_pct) > 0.05:
            self._memory.add_lesson(
                {
                    "ticker": ticker,
                    "won": won,
                    "pnl_pct": pnl_pct,
                    "regime": "unknown",
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
        total = sum(abs(v) for v in weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        self._memory.set_weights(weights)
