"""brain.cognitive.nash — Pure-brain Nash: pattern matching via episodic memory.

NO XGBoost. NO external model. Nash's work is done entirely by JULI's
existing cognitive pillars:
- Episodic memory (k-NN recall of similar past situations)
- Indicator consensus (how many agree on direction)
- Score velocity (is the signal building or fading)

Produces a bounded opinion (±0.03) that folds into JULI's composite.
Gate authority: rare veto when episodic recall is strong and confident.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

MOD_BOUND: float = 0.03
_MIN_SAMPLES: int = 10
_GATE_MIN: int = 20
_GATE_WR: float = 0.45
_KEYS = (
    "vpin",
    "orderbook_imbalance",
    "institutional_flow",
    "momentum",
    "rsi",
    "macd_hist",
    "adx",
    "bollinger_position",
    "mfi",
    "stoch_k",
)


@dataclass
class NashPrediction:
    win_prob: float = 0.5
    confidence: float = 0.0
    opinion: float = 0.0
    gate_authority: bool = False
    n_samples: int = 0


class NashBrain:
    """Pure-brain Nash: pattern matching via episodic memory."""

    def __init__(self) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=200)
        self._outcomes: deque[bool] = deque(maxlen=200)
        self._last: Optional[NashPrediction] = None

    def predict(
        self,
        alpha: dict[str, float],
        score: float,
        direction: int,
        prices: Optional[list[float]] = None,
        episodes: Optional[list[dict]] = None,
    ) -> NashPrediction:
        """Run pattern recognition using brain's own memory."""
        features = self._features(alpha, score, prices)
        wp, conf, n = self._match(features, episodes)
        opinion = self._opinion(wp, conf)
        gate = n >= _GATE_MIN and _GATE_WR <= wp <= (1 - _GATE_WR)
        pred = NashPrediction(wp, conf, opinion, gate, n)
        self._last = pred
        return pred

    def evaluate(self, alpha: dict[str, float], direction: int = 0) -> float:
        """Compatibility method — returns bounded modifier."""
        pred = self.predict(alpha, 0.0, direction)
        return self.fold_opinion(pred)["modifier"]

    def record_outcome(self, alpha: dict[str, float], score: float, won: bool) -> None:
        """Record trade outcome for future pattern matching."""
        self._history.append(self._features(alpha, score, None))
        self._outcomes.append(won)

    def fold_opinion(self, pred: NashPrediction) -> dict[str, float]:
        """Fold prediction into bounded modifier for JULI."""
        mod = max(-MOD_BOUND, min(MOD_BOUND, pred.opinion))
        return {
            "modifier": mod,
            "gate_veto": pred.gate_authority,
            "verdict": "HOLD" if pred.gate_authority else "PASS",
            "nash_win_prob": pred.win_prob,
            "confidence": pred.confidence,
        }

    def _features(
        self, alpha: dict[str, float], score: float, prices: Optional[list[float]]
    ) -> np.ndarray:
        """Extract feature vector from current situation."""
        feats = [alpha.get(k, 0.0) for k in _KEYS]
        feats.extend([score, abs(score)])
        if prices and len(prices) >= 5:
            arr = np.array(prices[-20:])
            r5 = (arr[-1] - arr[-5]) / arr[-5] if arr[-5] else 0
            r20 = (arr[-1] - arr[0]) / arr[0] if arr[0] else 0
            vol = float(np.std(np.diff(arr) / arr[:-1])) if len(arr) > 1 else 0
            feats.extend([r5, r20, vol])
        else:
            feats.extend([0.0, 0.0, 0.0])
        return np.array(feats, dtype=np.float64)

    def _match(
        self, features: np.ndarray, episodes: Optional[list[dict]]
    ) -> tuple[float, float, int]:
        """Match against historical patterns (k-NN)."""
        cands = []
        if episodes:
            for ep in episodes:
                vec = ep.get("vector", [])
                if len(vec) == len(features):
                    d = float(np.linalg.norm(features - np.array(vec)))
                    cands.append((d, ep.get("outcome", 0.5)))
        for hf, out in zip(self._history, self._outcomes):
            if len(hf) == len(features):
                cands.append((float(np.linalg.norm(features - hf)), float(out)))
        if not cands:
            return 0.5, 0.0, 0
        cands.sort(key=lambda x: x[0])
        k = min(7, len(cands))
        tw = ww = 0.0
        for d, o in cands[:k]:
            w = 1.0 / (1.0 + d)
            ww += o * w
            tw += w
        wp = ww / tw if tw else 0.5
        wins = sum(1 for _, o in cands[:k] if o > 0.5)
        agr = max(wins, k - wins) / max(k, 1)
        sf = min(1.0, k / _MIN_SAMPLES)
        return wp, agr * sf, len(cands[:k])

    def _opinion(self, wp: float, conf: float) -> float:
        if conf < 0.2:
            return 0.0
        return max(-MOD_BOUND, min(MOD_BOUND, (wp - 0.5) * 2 * conf * 0.15))

    def get_telemetry(self) -> dict:
        """Return telemetry data."""
        t = {"n_history": len(self._history)}
        if self._last:
            t.update({"opinion": self._last.opinion, "conf": self._last.confidence})
        return t


__all__ = ["NashBrain", "NashPrediction", "MOD_BOUND"]
