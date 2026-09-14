"""brain.rpe — multi-timescale dopamine reward-prediction error.

Biological grounding (Masset et al. 2025, Nature 642:682): dopamine
neurons carry value signals at heterogeneous timescales. Three channels:

- phasic (fast, ``RPE_ALPHA_FAST``): immediate signed prediction error
  ``r - V`` — the surprise a single trade should teach. Fades after a few
  consistent outcomes, exactly as a curiosity signal should.
- tonic (slow, ``RPE_ALPHA_SLOW``): long-run drift of expected value;
  tonic RPE is the deviation from the baseline trend.
- meta (per-regime, ``RPE_ALPHA_META``): value expectations conditioned on
  the trading regime, so a surprise in ``trend`` reads differently than
  the same surprise in ``choppy``.

Each channel is a value estimate ``V`` updated ``V += alpha * (r - V)``
with ``r = 1`` on a win and ``0`` on a loss. The predicted win probability
of the entry seeds the channels on the first trade (the brain's own
calibrated prior); downstream consumers read the signed channels for
affect and ``surprise`` (``|phasic|``) as a curiosity-driven learning-rate
modulator.

Persists to ``runtime/juli_rpe.json`` (``HANOO_RPE_FILE`` keeps tests
hermetic, mirroring ``RealizedStats``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .learning_config import (
    RPE_ALPHA_FAST,
    RPE_ALPHA_META,
    RPE_ALPHA_SLOW,
    RPE_FILE,
    RPE_LR_GAIN,
    RPE_LR_MAX,
    RPE_LR_MIN,
)

log = logging.getLogger(__name__)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Bound a float to ``[lo, hi]``."""
    return max(lo, min(hi, float(value)))


def lr_modulator(surprise: float) -> float:
    """Map a surprise magnitude to a bounded learning-rate multiplier.

    Unsurprising trades (``surprise`` near 0) learn at ``RPE_LR_MIN``-ish
    pace; full surprise (``|phasic| = 1``) lifts the multiplier toward
    ``RPE_LR_MAX``. Never returns outside ``[RPE_LR_MIN, RPE_LR_MAX]``.
    """
    scale = 1.0 + RPE_LR_GAIN * abs(float(surprise))
    return _clamp(scale, RPE_LR_MIN, RPE_LR_MAX)


class MultiTimescaleRPE:
    """Three-channel dopamine value estimator with signed prediction error."""

    def __init__(self, filepath: Path | None = None, persist: bool = True) -> None:
        self._path: Path | None = None
        if filepath is not None:
            self._path = Path(filepath)
        elif persist:
            env_file = os.environ.get("HANOO_RPE_FILE", "").strip()
            self._path = Path(env_file) if env_file else RPE_FILE
        self._lock = threading.RLock()
        self._v_fast: float = 0.5
        self._v_slow: float = 0.5
        self._v_meta: dict[str, float] = {}
        self._count: int = 0
        self._last_phasic: float = 0.0
        self._last_tonic: float = 0.0
        if self._path is not None:
            self._load()

    def update(
        self, predicted_win_prob: float, won: bool, regime: str
    ) -> dict[str, Any]:
        """Fold one closed trade into every channel; returns the channels.

        The signed phasic/tonic/meta prediction errors are computed first,
        then each value estimate moves toward the outcome ``r``. On the
        first trade the predicted win probability seeds the fast and slow
        channels so the brain starts from its own calibrated prior.
        """
        r = 1.0 if won else 0.0
        with self._lock:
            if self._count == 0:
                prior = _clamp(predicted_win_prob)
                self._v_fast = prior
                self._v_slow = prior
            regime_key = regime or "unknown"
            v_meta = float(self._v_meta.get(regime_key, 0.5))
            phasic = r - self._v_fast
            tonic = r - self._v_slow
            meta = r - v_meta
            self._v_fast += RPE_ALPHA_FAST * phasic
            self._v_slow += RPE_ALPHA_SLOW * tonic
            self._v_meta[regime_key] = v_meta + RPE_ALPHA_META * meta
            self._count += 1
            self._last_phasic = phasic
            self._last_tonic = tonic
            result = {
                "phasic": round(_clamp(phasic, -1.0, 1.0), 6),
                "tonic": round(_clamp(tonic, -1.0, 1.0), 6),
                "meta": round(_clamp(meta, -1.0, 1.0), 6),
                "v_fast": round(_clamp(self._v_fast), 6),
                "v_slow": round(_clamp(self._v_slow), 6),
                "v_meta": {k: round(_clamp(v), 6) for k, v in self._v_meta.items()},
                "surprise": round(_clamp(abs(phasic), 0.0, 1.0), 6),
            }
            self._save()
            return result

    @property
    def phasic_rpe(self) -> float:
        """Most recent signed phasic prediction error."""
        return self._last_phasic

    @property
    def tonic_rpe(self) -> float:
        """Most recent signed tonic prediction error."""
        return self._last_tonic

    @property
    def surprise(self) -> float:
        """Most recent surprise magnitude (|phasic|)."""
        return abs(self._last_phasic)

    @property
    def tonic_value(self) -> float:
        """Slow expected-value estimate (the long-run tone)."""
        return _clamp(self._v_slow)

    @property
    def count(self) -> int:
        """Number of closed trades observed."""
        return self._count

    def snapshot(self) -> dict[str, Any]:
        """Read-only dump for telemetry and persistence checks."""
        with self._lock:
            return {
                "count": self._count,
                "v_fast": round(_clamp(self._v_fast), 6),
                "v_slow": round(_clamp(self._v_slow), 6),
                "v_meta": dict(self._v_meta),
                "phasic_rpe": round(_clamp(self._last_phasic, -1.0, 1.0), 6),
                "tonic_rpe": round(_clamp(self._last_tonic, -1.0, 1.0), 6),
            }

    def _load(self) -> None:
        """Restore channels from disk (start fresh on any corruption)."""
        if self._path is None or not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            self._v_fast = _clamp(float(d.get("v_fast", 0.5)))
            self._v_slow = _clamp(float(d.get("v_slow", 0.5)))
            self._v_meta = {
                str(k): _clamp(float(v)) for k, v in dict(d.get("v_meta", {})).items()
            }
            self._count = int(d.get("count", 0))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            log.warning("RPE state load failed (starting fresh): %s", e)

    def _save(self) -> None:
        """Atomically persist all channels."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "v_fast": self._v_fast,
            "v_slow": self._v_slow,
            "v_meta": self._v_meta,
            "count": self._count,
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, default=str))
        tmp.replace(self._path)
