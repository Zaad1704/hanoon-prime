"""brain.scoring — Advanced scoring: ensemble, consensus, GAT, velocity.

Consolidates rebuild's expert_ensemble.py, consensus.py,
cross_indicator_gat.py, and score_velocity.py into one module.

All components are bounded, learned from outcomes, never gates.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .config import DEFAULT_WEIGHTS

# ── Expert Ensemble (MWU) ────────────────────────────────────────────
_MWU_LR: float = 0.05
_MWU_FLOOR: float = 0.01
_MWU_CEIL: float = 0.30
_MWU_LOOKBACK: int = 50


class ExpertEnsemble:
    """Multiplicative Weights Update for online indicator weight learning."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._weights: dict[str, float] = {
            k: 1.0 / len(DEFAULT_WEIGHTS) for k in DEFAULT_WEIGHTS
        }
        self._history: deque[dict[str, float]] = deque(maxlen=_MWU_LOOKBACK)

    def record_predictions(
        self, signals: dict[str, float], actual_direction: int
    ) -> None:
        """Record which experts were correct after each bar."""
        self._history.append(dict(signals))
        if len(self._history) < 5:
            return
        for ind in self._weights:
            correct = 0
            for snap in self._history:
                pred = snap.get(ind, 0.0)
                if (pred > 0 and actual_direction > 0) or (
                    pred < 0 and actual_direction < 0
                ):
                    correct += 1
            wr = correct / len(self._history)
            self._weights[ind] *= math.exp(_MWU_LR * (wr - 0.5))
            self._weights[ind] = max(_MWU_FLOOR, min(_MWU_CEIL, self._weights[ind]))
        self._normalize()

    def _normalize(self) -> None:
        """Auto-generated docstring."""
        total = sum(self._weights.values())
        if total > 0:
            for k in self._weights:
                self._weights[k] /= total

    def get_weights(self) -> dict[str, float]:
        """Auto-generated docstring."""
        return dict(self._weights)

    def consensus_boost(self) -> float:
        """Boost when experts agree on direction."""
        if not self._history:
            return 0.0
        last = self._history[-1]
        pos = sum(1 for v in last.values() if v > 0)
        neg = sum(1 for v in last.values() if v < 0)
        n = len(last)
        agreement = max(pos, neg) / max(n, 1)
        return (agreement - 0.5) * 0.08  # bounded ±0.04


# ── Consensus Tracker ────────────────────────────────────────────────
_BOOST_MAX: float = 0.04
_PENALTY_MAX: float = 0.04


@dataclass
class ConsensusState:
    score: float = 0.0
    agreement: float = 0.0
    n_agree: int = 0
    n_total: int = 0


class ConsensusTracker:
    """Track directional consensus across indicators."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._state = ConsensusState()

    def update(
        self, signals: dict[str, float], weights: Optional[dict[str, float]] = None
    ) -> ConsensusState:
        """Compute weighted consensus score."""
        if not signals:
            return self._state
        w = weights or {k: 1.0 for k in signals}
        total_w = sum(abs(w.get(k, 0)) for k in signals) or 1.0
        weighted_sum = (
            sum(signals.get(k, 0) * abs(w.get(k, 0)) for k in signals) / total_w
        )
        n_pos = sum(1 for v in signals.values() if v > 0)
        n_neg = sum(1 for v in signals.values() if v < 0)
        n_total = len(signals)
        agreement = max(n_pos, n_neg) / max(n_total, 1)
        self._state = ConsensusState(
            score=weighted_sum,
            agreement=agreement,
            n_agree=max(n_pos, n_neg),
            n_total=n_total,
        )
        return self._state

    @property
    def modifier(self) -> float:
        """Bounded modifier from consensus."""
        a = self._state.agreement
        if a > 0.7:
            return min(_BOOST_MAX, (a - 0.7) * 0.13)
        if a < 0.5:
            return max(-_PENALTY_MAX, (a - 0.5) * 0.13)
        return 0.0


# ── Score Velocity ───────────────────────────────────────────────────
@dataclass
class VelocityState:
    velocity: float = 0.0
    acceleration: float = 0.0
    rising_fast: bool = False


class ScoreVelocity:
    """Track score velocity for anticipation signals."""

    def __init__(self, window: int = 5) -> None:
        """Auto-generated docstring."""
        self._window = window
        self._history: deque[float] = deque(maxlen=window)
        self._prev_velocity: float = 0.0

    def update(self, score: float) -> VelocityState:
        """Update velocity tracking with new score."""
        self._history.append(score)
        if len(self._history) < 2:
            return VelocityState()
        scores = list(self._history)
        velocity = scores[-1] - scores[-2]
        acceleration = velocity - self._prev_velocity
        self._prev_velocity = velocity
        return VelocityState(
            velocity=velocity,
            acceleration=acceleration,
            rising_fast=velocity > 0.05 and acceleration > 0,
        )

    @property
    def modifier(self) -> float:
        """Bounded anticipation modifier."""
        if len(self._history) < 2:
            return 0.0
        vel = self._history[-1] - self._history[-2]
        return max(-0.03, min(0.03, vel * 0.3))


# ── Cross-Indicator GAT (simplified) ────────────────────────────────
class CrossIndicatorGAT:
    """Simplified graph attention for indicator relationships."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._attention: dict[str, float] = {
            k: 1.0 / len(DEFAULT_WEIGHTS) for k in DEFAULT_WEIGHTS
        }

    def compute_attention(self, signals: dict[str, float]) -> dict[str, float]:
        """Compute attention-weighted signals."""
        result = {}
        for k, v in signals.items():
            attn = self._attention.get(k, 1.0 / len(DEFAULT_WEIGHTS))
            result[k] = v * attn
        return result

    def update_attention(self, signals: dict[str, float], outcome: float) -> None:
        """Update attention based on outcome."""
        for k, v in signals.items():
            if abs(v) > 0.5:
                self._attention[k] *= 1.0 + 0.01 * outcome
                self._attention[k] = max(0.1, min(2.0, self._attention[k]))
