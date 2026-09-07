"""brain.meta_label — Meta-labeling shadow layer (López de Prado port).

The cortex stays the sole verdict source (R1). This layer learns WHEN
cortex verdicts historically fail — a secondary model trained on real
trade outcomes — and answers with a bounded SIZE scalar only: it can
shrink an admitted entry, never create, flip, or block one.

Features at entry: confidence, |score|, volatility percentile, canonical
regime, horizon. Online logistic regression (SGD, bounded weights),
updated once per real trade close by the orchestrator (single writer).
Until META_MIN_SAMPLES accumulate the scalar stays 1.0 — thin-data
safety, the same contract as the EV gate.

Shadow semantics: while warming up the model still predicts and scores
itself (Brier) on every close, so its calibration is measurable before
its first sizing intervention.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

from .learning_config import (
    META_CUT,
    META_FILE,
    META_LR,
    META_MIN_SAMPLES,
    META_SIZE_MIN,
    META_WEIGHT_MAX,
)

log = logging.getLogger(__name__)

REGIMES: tuple[str, ...] = ("trend_up", "trend_down", "range", "vol", "unknown")
HORIZONS: tuple[str, ...] = ("scalp", "momentum", "swing")
_Z_CLAMP: float = 30.0


def feature_vector(
    conf: float, score: float, vol_pct: float, regime: str, horizon: str
) -> list[float]:
    """Build the meta-model feature vector from entry context."""
    vec = [1.0, float(conf), min(1.0, abs(float(score))), float(vol_pct)]
    vec += [1.0 if regime == r else 0.0 for r in REGIMES]
    vec += [1.0 if horizon == h else 0.0 for h in HORIZONS]
    return vec


def _sigmoid(z: float) -> float:
    """Bounded logistic."""
    return 1.0 / (1.0 + math.exp(-max(-_Z_CLAMP, min(_Z_CLAMP, z))))


class MetaLabelModel:
    """Online logistic meta-model → bounded size scalar in [META_SIZE_MIN, 1]."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or META_FILE
        self._lock = threading.RLock()
        self._dim = 4 + len(REGIMES) + len(HORIZONS)
        self._w: list[float] = [0.0] * self._dim
        self._n: int = 0
        self._brier_n: int = 0
        self._brier_sum: float = 0.0
        self._load()

    def p_win(self, features: list[float]) -> float:
        """Predict P(trade wins | entry context)."""
        with self._lock:
            z = sum(w * x for w, x in zip(self._w, features))
            return _sigmoid(z)

    def record(self, features: list[float], won: bool) -> None:
        """One real trade close → SGD step + Brier update."""
        with self._lock:
            p = _sigmoid(sum(w * x for w, x in zip(self._w, features)))
            grad = (1.0 if won else 0.0) - p
            for i, x in enumerate(features):
                self._w[i] = max(
                    -META_WEIGHT_MAX,
                    min(META_WEIGHT_MAX, self._w[i] + META_LR * grad * x),
                )
            self._n += 1
            self._brier_sum += (p - (1.0 if won else 0.0)) ** 2
            self._brier_n += 1
            self._save()

    def size_scalar(
        self, conf: float, score: float, vol_pct: float, regime: str, horizon: str
    ) -> float:
        """Bounded size multiplier in [META_SIZE_MIN, 1.0]; 1.0 while cold."""
        with self._lock:
            if self._n < META_MIN_SAMPLES:
                return 1.0
            feats = feature_vector(conf, score, vol_pct, regime, horizon)
            p = _sigmoid(sum(w * x for w, x in zip(self._w, feats)))
            if p >= META_CUT:
                return 1.0
            frac = max(0.0, min(1.0, p / META_CUT))
            return round(META_SIZE_MIN + (1.0 - META_SIZE_MIN) * frac, 4)

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view: sample count, calibration, active flag."""
        with self._lock:
            brier = self._brier_sum / self._brier_n if self._brier_n else None
            return {
                "n": self._n,
                "brier": round(brier, 4) if brier is not None else None,
                "sizing_active": self._n >= META_MIN_SAMPLES,
            }

    def _load(self) -> None:
        """Load persisted weights; corrupt/missing file → fresh model."""
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            w = d.get("w", [])
            self._w = [
                max(-META_WEIGHT_MAX, min(META_WEIGHT_MAX, float(x))) for x in w
            ][: self._dim]
            self._w += [0.0] * (self._dim - len(self._w))
            self._n = int(d.get("n", 0))
            self._brier_n = int(d.get("brier_n", 0))
            self._brier_sum = float(d.get("brier_sum", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("Meta-label load failed (starting fresh): %s", exc)

    def _save(self) -> None:
        """Persist atomically (tmp + replace)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "w": [round(x, 6) for x in self._w],
                        "n": self._n,
                        "brier_n": self._brier_n,
                        "brier_sum": round(self._brier_sum, 6),
                    }
                )
            )
            tmp.replace(self._path)
        except OSError as exc:
            log.debug("Meta-label save failed: %s", exc)
