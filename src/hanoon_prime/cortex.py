"""hanoon_prime.cortex — signal scoring and entry verdict.

R1: This is the ONLY module that produces verdict strings (BUY, SELL, HOLD).
All other modules compute indicators or probabilities. Validated by
tests/test_contract.py.

Architecture (simplified from the 880-line thinker.py):
  1. Receive raw indicators (core 5 + tech set from cerebellum/indicators)
  2. Z-score normalize each against rolling history (scale-invariant)
  3. score = tanh(Σ w_i × z_i)  → symmetric [-1, +1]
  4. Verdict: BUY if score > +threshold, SELL if score < -threshold

Learning integration (v2.1): the Cortex scores EVERY weighted indicator,
not just the core 5. Weights are learned across all 27 keys by
brain/reflection.py and hot-swapped here via ``set_weights`` after each
trade close — the live tanh therefore evolves with every trade. Backtests
keep the historical 5-core behavior via ``core_weights_only=True`` so the
calibration pipeline and legacy backtests are unchanged.

No modifiers, no gates, no percentile trickery. Just the math.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

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
from .types import BarSeries


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

    Z-score history is created LAZILY per indicator key (on first
    ``evaluate``), so any alpha key set — the 5 core names, the 27-key
    live set, or any subset — is scored correctly without pre-registering
    keys. Weights are hot-swappable: the orchestrator refreshes them from
    the learned memory after every trade close.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        threshold: float = ENTRY_THRESHOLD,
        z_window: int = Z_NORM_WINDOW,
        core_weights_only: bool = False,
    ) -> None:
        self._weights: dict[str, float] = dict(weights or INDICATOR_WEIGHTS)
        self._core_weights_only: bool = core_weights_only
        self._threshold: float = threshold
        self._z_window: int = z_window
        self._z_history: dict[str, deque[float]] = {
            name: deque(maxlen=z_window) for name in INDICATOR_NAMES
        }

    def set_weights(self, weights: dict[str, float]) -> None:
        """Hot-swap indicator weights (called by the orchestrator after learning)."""
        self._weights = dict(weights)

    def evaluate(self, raw: dict[str, float]) -> Thought:
        """Z-score normalize raw indicators → tanh score → verdict."""
        z_scores: dict[str, float] = {}
        for name, raw_val in raw.items():
            hist = self._z_history.get(name)
            if hist is None:
                hist = deque(maxlen=self._z_window)
                self._z_history[name] = hist
            z = self._z_score(raw_val, hist)
            z_scores[name] = z
            hist.append(raw_val)

        score = self._tanh_score(z_scores, present=set(raw.keys()))
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

    def _tanh_score(
        self, z: dict[str, float], present: set[str] | None = None
    ) -> float:
        """score = tanh(Σ w_i × z_i / Σ w_i) over available indicators.

        Iterates the WEIGHTS (not a hard-coded name list) so the 22 tech
        indicators participate in the score and learned weight drift is
        expressed. The sum is renormalized over indicators PRESENT in the
        current alpha (a proper weighted average), so partial snapshots —
        e.g. the 5-core fallback when tech indicators lack data — are not
        diluted by absent keys. With the full core-5 weight set and all 5
        present this reproduces the legacy sum exactly (Σw = 1.0).
        In core_weights_only mode (backtests, calibration) only the
        historical 5 core names contribute.
        """
        names = INDICATOR_NAMES if self._core_weights_only else tuple(self._weights)
        keys = tuple(n for n in names if present is None or n in present)
        if not keys:
            return 0.0
        weighted = sum(float(self._weights.get(n, 0.0)) * z.get(n, 0.0) for n in keys)
        total_w = sum(float(self._weights.get(n, 0.0)) for n in keys)
        if total_w <= 1e-12:
            return 0.0
        return float(math.tanh(weighted / total_w))

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


def deliberate(cortex: Cortex, bars: BarSeries) -> Thought:
    """Full pipeline: cerebellum → cortex.evaluate → Thought.

    This is the public entry point for the brain's verdict decision.
    It accepts raw market data (not pre-computed alpha) so callers
    don't need to know about cerebellum internals.
    """
    alpha = compute_alpha(
        close=bars.close,
        volume=bars.volume,
        buy_volume=bars.buy_volume,
        bid_sizes=bars.bid_sizes,
        ask_sizes=bars.ask_sizes,
    )
    return cortex.evaluate(alpha)
