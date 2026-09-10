"""hanoon_prime.cortex — signal scoring and entry verdict.

R1: ONLY module that produces verdict strings (BUY, SELL, HOLD).
Validated by tests/test_contract.py.

Pipeline: indicators → z-score normalize → tanh(Σ w_i z_i / Σ |w_i|)
→ BUY if score > +threshold, SELL if score < -threshold.
Weights are learned across all 27 keys and hot-swapped after every trade
close; backtests keep the legacy 5-core path via ``core_weights_only=True``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .cerebellum import INDICATOR_NAMES, compute_alpha
from .contrarian import contrarian_direction
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

    def get_weights(self) -> dict[str, float]:
        """Current effective weights (live view for the strategy genome)."""
        return dict(self._weights)

    def evaluate(
        self, raw: dict[str, float], prior_top: float | None = None
    ) -> Thought:
        """Z-score normalize → tanh score → verdict (prior_top widens cap)."""
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
        if cdir := contrarian_direction(z_scores):
            verdict, direction = ("BUY", cdir) if cdir > 0 else ("SELL", cdir)
        win_prob = score_to_win_prob(score, prior_top=prior_top)
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
        """score = tanh(Σ w_i × z_i / Σ |w_i|) over available indicators.

        Iterates the WEIGHTS (not a hard-coded name list) so the 22 tech
        indicators participate and learned weight drift is expressed.
        Normalization is over the indicators PRESENT in the current alpha,
        so partial snapshots are not diluted by absent keys.

        Dividing by Σ|w| (not signed Σw) is deliberate: negative learned
        weights stay meaningful, a negative-sum regime budget cannot flip the
        score sign, and a drifted budget cannot zero it. The previous signed
        guard (``total_w <= 1e-12 → 0.0``) was triggered by any negative-sum
        vector — e.g. a regime vector at -2.86 — forcing a zeroed score the
        orchestrator read as SHORT. Only a genuinely degenerate budget
        (all weights ~0) now forces the score flat.
        """
        names = INDICATOR_NAMES if self._core_weights_only else tuple(self._weights)
        keys = tuple(n for n in names if present is None or n in present)
        if not keys:
            return 0.0
        weighted = sum(float(self._weights.get(n, 0.0)) * z.get(n, 0.0) for n in keys)
        total_abs_w = sum(abs(float(self._weights.get(n, 0.0))) for n in keys)
        if total_abs_w <= 1e-12:
            return 0.0
        return float(math.tanh(weighted / total_abs_w))

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

    def _hold_reason(self, score: float) -> str:
        if abs(score) < self._threshold:
            return f"|{score:.3f}| < {self._threshold:.3f}"
        if not SHORT_ALLOWED and score <= -self._threshold:
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
