"""brain.meta_label — Meta-labeling shadow layer (López de Prado port).

Cortex stays the sole verdict source (R1). This layer learns WHEN cortex
verdicts historically fail — secondary model trained on real trade outcomes.

When META_DNN_ENABLED is True, the DNN gatekeeper returns admit=False to
VETO trades below META_WIN_THRESHOLD.  Shallow logistic is the fallback.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .meta_label_dnn import MetaDNN

from .learning_config import (
    META_CUT,
    META_DNN_ENABLED,
    META_FILE,
    META_LR,
    META_MIN_SAMPLES,
    META_SIZE_MIN,
    META_WEIGHT_MAX,
    META_WIN_THRESHOLD,
)
from .self_correction import ABSTAIN_P_WIN, CalibrationMonitor, PredictionLedger
from .self_correction_policy import apply_derating, evaluate_weight

log = logging.getLogger(__name__)

_DNN_INSTANCE: Any = None  # lazy singleton (MetaDNN | None)


def _get_dnn() -> Any:
    global _DNN_INSTANCE
    if _DNN_INSTANCE is None:
        from .meta_label_dnn import MetaDNN

        _DNN_INSTANCE = MetaDNN()
    return _DNN_INSTANCE


WEIGHT_EVAL_TTL_S: float = 60.0  # seconds between derating re-evaluations
_WEIGHT_CACHE: tuple[float, float, str] | None = None  # (ts, weight, reason)


def dnn_influence_weight(drift_z: float) -> tuple[float, str]:
    """Current DNN influence weight. TTL-cached; fail-safe; never raises."""
    global _WEIGHT_CACHE
    now = time.time()
    if _WEIGHT_CACHE is not None and now - _WEIGHT_CACHE[0] < WEIGHT_EVAL_TTL_S:
        return _WEIGHT_CACHE[1], _WEIGHT_CACHE[2]
    ledger = PredictionLedger()
    pairs, total = ledger.calibration_data()
    snapshot = CalibrationMonitor.summarize(pairs)
    try:
        weight, reason = evaluate_weight(snapshot, drift_z, total)
    except Exception as exc:  # derating must never break the gate
        log.warning("derating evaluation failed: %s", exc)
        weight, reason = 0.5, "evaluation failed"
    _WEIGHT_CACHE = (now, weight, reason)
    return weight, reason


def gate_with_derating(
    dnn: Any, features: list[float], admit: bool, p_win: float, size_scale: float
) -> tuple[bool, float, float]:
    """Fail-closed DNN gate: unusable DNN or zero weight -> block; else derate; always log."""
    abstain_reason = getattr(dnn, "abstain_reason", None)
    try:
        drift_z = float(dnn.drift_zscore())
    except (AttributeError, TypeError, ValueError):
        drift_z = 0.0
    weight, w_reason = dnn_influence_weight(drift_z)
    stepped_aside = abstain_reason is not None or weight <= 0.0
    if stepped_aside:
        out = (False, ABSTAIN_P_WIN, 0.0)
    else:
        out = apply_derating(
            admit, p_win, size_scale, weight, threshold=META_WIN_THRESHOLD
        )
    try:
        PredictionLedger().log_decision(
            p_win=out[1],
            features=features,
            size_scale=out[2],
            admitted=out[0],
            abstained=stepped_aside,
            weight=weight,
            drift_z=drift_z,
        )
    except Exception as exc:  # the ledger must never break the gate
        log.debug("ledger log failed: %s", exc)
    if stepped_aside:
        log.info("DNN blocked (%s)", abstain_reason or w_reason)
    return out


REGIMES: tuple[str, ...] = ("trend_up", "trend_down", "range", "vol", "unknown")
HORIZONS: tuple[str, ...] = ("scalp", "momentum", "swing")
_Z_CLAMP: float = 30.0


def feature_vector(
    conf: float,
    score: float,
    vol_pct: float,
    regime: str,
    horizon: str,
) -> list[float]:
    """Build the meta-model feature vector from entry context."""
    vec = [1.0, float(conf), min(1.0, abs(float(score))), float(vol_pct)]
    vec += [1.0 if regime == r else 0.0 for r in REGIMES]
    vec += [1.0 if horizon == h else 0.0 for h in HORIZONS]
    return vec


def _sigmoid(z: float) -> float:
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

    @property
    def dnn_p_win(self) -> float:
        """Last DNN P(Win) from the most recent gate() call (0.0 if unavailable)."""
        try:
            return float(_get_dnn()._last_p_win)
        except Exception:
            return 0.0

    def p_win(self, features: list[float]) -> float:
        """Predict P(trade wins | entry context)."""
        with self._lock:
            return _sigmoid(sum(w * x for w, x in zip(self._w, features)))

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

    def gate(
        self,
        conf: float,
        score: float,
        vol_pct: float,
        regime: str,
        horizon: str,
        direction: int = 1,
        atr_ratio: float = 0.0,
        obi: float = 0.0,
        vpin: float = 0.0,
        price_entropy: float = 1.0,
        vol_entropy: float = 1.0,
    ) -> tuple[bool, float, float]:
        """DNN gatekeeper: (admit, p_win, size_scale). Shallow fallback runs
        only when the DNN is disabled; DNN failures block."""
        if META_DNN_ENABLED:
            result = self._dnn_gate(
                conf,
                score,
                vol_pct,
                direction,
                atr_ratio,
                obi,
                vpin,
                price_entropy,
                vol_entropy,
            )
            if result is not None:
                return result
        return self._fallback_gate(conf, score, vol_pct, regime, horizon)

    def _dnn_gate(
        self,
        conf: float,
        score: float,
        vol_pct: float,
        direction: int,
        atr_ratio: float,
        obi: float,
        vpin: float,
        price_entropy: float = 1.0,
        vol_entropy: float = 1.0,
    ) -> tuple[bool, float, float]:
        try:
            from .meta_label_dnn import expand_features

            feats = expand_features(
                conf,
                score,
                vol_pct,
                direction,
                atr_ratio,
                obi,
                vpin,
                price_entropy,
                vol_entropy,
            )
            admit: bool
            p: float
            scale: float
            admit, p, scale = _get_dnn().infer(feats)
            return gate_with_derating(_get_dnn(), feats, admit, p, scale)
        except Exception as exc:
            # Fail-closed: never fall through to the always-admitting
            # shallow fallback while the DNN is enabled. Block instead.
            log.warning("DNN gatekeeper failed — blocking: %s", exc)
            return False, ABSTAIN_P_WIN, 0.0

    def _fallback_gate(
        self, conf: float, score: float, vol_pct: float, regime: str, horizon: str
    ) -> tuple[bool, float, float]:
        with self._lock:
            feats = feature_vector(conf, score, vol_pct, regime, horizon)
            p = _sigmoid(sum(w * x for w, x in zip(self._w, feats)))
            scale = (
                1.0
                if p >= META_CUT
                else round(
                    META_SIZE_MIN
                    + (1.0 - META_SIZE_MIN) * max(0.0, min(1.0, p / META_CUT)),
                    4,
                )
            )
            return True, p, scale

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view: sample count, calibration, active flag."""
        with self._lock:
            brier = self._brier_sum / self._brier_n if self._brier_n else None
            return {
                "n": self._n,
                "brier": round(brier, 4) if brier is not None else None,
                "sizing_active": self._n >= META_MIN_SAMPLES,
            }

    def dnn_snapshot(self) -> dict[str, Any]:
        """Telemetry view: DNN gatekeeper live state."""
        if META_DNN_ENABLED:
            try:
                result: dict[str, Any] = _get_dnn().live_snapshot()
                return result
            except Exception as exc:
                log.debug("DNN snapshot failed: %s", exc)
        return {"enabled": False, "gate_active": False}

    def _load(self) -> None:
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
