"""hanoon_prime.cortex — signal scoring and entry verdict.

R1: This is the ONLY module that produces verdict strings (BUY, SELL, HOLD).
All other modules compute indicators or probabilities. Validated by
tests/test_contract.py.

Architecture (simplified from the 880-line thinker.py):
  1. Receive raw indicators from cerebellum (5 values, mixed scales)
  2. Z-score normalize each against rolling history (scale-invariant)
  3. score = tanh(Σ w_i × z_i)  → symmetric [-1, +1]
  4. Verdict: BUY if score > +threshold, SELL if score < -threshold

No modifiers, no gates, no percentile trickery. Just the math.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cerebellum import INDICATOR_NAMES, compute_alpha
from .edge import compute_ev, kelly_fraction, score_to_win_prob
from .immune import (
    CONFIDENCE_FLOOR,
    ENTRY_THRESHOLD,
    INDICATOR_WEIGHTS,
    SHORT_ALLOWED,
    Z_CLIP,
    Z_NORM_WINDOW,
)


@dataclass
class Thought:
    """The cortex's output — verdict + diagnostic metadata."""

    verdict: str = "HOLD"
    confidence: float = 0.5
    score: float = 0.0
    direction: int = 0
    z_scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    trace: dict[str, float] = field(default_factory=dict)


class Cortex:
    """Z-score normalization + tanh scoring + dual-direction verdict.

    Maintains rolling z-score history for each of the 5 indicators.
    At each evaluation the raw indicator values are z-scored against
    the rolling history (scale-invariant across tickers), then fed
    through a tanh of weighted z-scores to produce a symmetric [-1, +1]
    signal.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        threshold: float = ENTRY_THRESHOLD,
        z_window: int = Z_NORM_WINDOW,
    ) -> None:
        self._weights: dict[str, float] = dict(weights or INDICATOR_WEIGHTS)
        self._threshold: float = threshold
        self._z_window: int = z_window
        self._z_history: dict[str, deque[float]] = {
            name: deque(maxlen=z_window) for name in INDICATOR_NAMES
        }

    def evaluate(self, raw: dict[str, float]) -> Thought:
        """Z-score normalize raw indicators → tanh score → verdict."""
        z_scores: dict[str, float] = {}
        for name in INDICATOR_NAMES:
            raw_val = float(raw.get(name, 0.0))
            hist = self._z_history[name]
            z = self._z_score(raw_val, hist)
            z_scores[name] = z
            hist.append(raw_val)

        score = self._tanh_score(z_scores)
        verdict, direction = self._verdict(score)
        win_prob = score_to_win_prob(score)
        ev = compute_ev(win_prob)
        kelly = kelly_fraction(win_prob)
        confidence = self._confidence(abs(score))

        reasons: list[str] = [] if verdict != "HOLD" else [self._hold_reason(score)]
        return Thought(
            verdict=verdict,
            confidence=round(confidence, 4),
            score=round(score, 4),
            direction=direction,
            z_scores={k: round(v, 4) for k, v in z_scores.items()},
            reasons=reasons,
            trace={
                "win_prob": round(win_prob, 4),
                "gross_ev": round(ev["gross_ev"], 4),
                "kelly": round(kelly, 4),
            },
        )

    def _z_score(self, val: float, hist: deque[float]) -> float:
        """Z-score *val* against rolling history, clipped to ±Z_CLIP."""
        if len(hist) < 2:
            return 0.0
        arr = np.array(hist, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-12:
            return 0.0
        z = (val - mean) / std
        return float(max(-Z_CLIP, min(Z_CLIP, z)))

    def _tanh_score(self, z: dict[str, float]) -> float:
        """score = tanh(Σ w_i × z_i) — symmetric around 0."""
        weighted = sum(self._weights[name] * z[name] for name in INDICATOR_NAMES)
        return float(math.tanh(weighted))

    def _verdict(self, score: float) -> tuple[str, int]:
        """Dual-direction verdict: BUY/SELL/HOLD."""
        if score >= self._threshold:
            return "BUY", 1
        if score <= -self._threshold and SHORT_ALLOWED:
            return "SELL", -1
        return "HOLD", 0

    @staticmethod
    def _confidence(abs_score: float) -> float:
        """Map |tanh score| ∈ [0, 1] → confidence ∈ [0.5, 0.95]."""
        return float(max(CONFIDENCE_FLOOR, min(0.95, 0.5 + abs_score * 0.45)))

    @staticmethod
    def _hold_reason(score: float) -> str:
        if abs(score) < ENTRY_THRESHOLD:
            return f"|{score:.3f}| < {ENTRY_THRESHOLD}"
        if not SHORT_ALLOWED and score <= -ENTRY_THRESHOLD:
            return "SHORT disabled"
        return f"score {score:.3f} below threshold"


def deliberate(
    cortex: Cortex,
    close: Any,
    volume: Any,
    buy_volume: Any | None = None,
    bid_sizes: Any | None = None,
    ask_sizes: Any | None = None,
) -> Thought:
    """Full pipeline: cerebellum → cortex.evaluate → Thought.

    This is the public entry point for the brain's verdict decision.
    It accepts raw market data (not pre-computed alpha) so callers
    don't need to know about cerebellum internals.
    """
    raw = close
    alpha = compute_alpha(
        close=raw,
        volume=volume,
        buy_volume=buy_volume,
        bid_sizes=bid_sizes,
        ask_sizes=ask_sizes,
    )
    return cortex.evaluate(alpha)
