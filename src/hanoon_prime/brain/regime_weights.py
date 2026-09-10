"""brain.regime_weights — per-regime indicator weight vectors.

The cortex's global learned weights average across regimes; the same
indicator can genuinely mean the opposite in a trend vs a chop. This
module keeps one weight vector per canonical regime, updated by the
Reflector's asymmetric gradient only for the regime the trade CLOSED
in. A regime vector applies (blended) once it has REGIME_MIN_TRADES
real closes; below that the structural default applies.

Single-writer: the orchestrator's close path. Every vector is routed
through the weight enforcer on learn and on load — clamped to the
enforcer band ([-0.20, +0.20]) and renormalized to |sum| ≈ 1 so a drifted
regime budget can never zero or sign-flip the cortex score.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_WEIGHTS,
    LEARNING_RATE,
    PENALTY_SCALE,
    REWARD_SCALE,
    WEIGHT_DECAY,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from .learning_config import REGIME_FILE, REGIME_MIN_TRADES
from .weight_enforcer import get_enforcer

log = logging.getLogger(__name__)


class RegimeWeights:
    """Per-regime weight vectors with thin-data fallback to defaults."""

    def __init__(self, path: Path | None = None) -> None:
        # HANOO_REGIME_FILE keeps smoke runs out of the production vectors.
        if path is None:
            env_file = os.environ.get("HANOO_REGIME_FILE", "").strip()
            path = Path(env_file) if env_file else REGIME_FILE
        self._path = path
        self._lock = threading.RLock()
        self._vectors: dict[str, dict[str, float]] = {}
        self._counts: dict[str, int] = {}
        self._lr = LEARNING_RATE
        self._reward = REWARD_SCALE
        self._penalty = PENALTY_SCALE
        self._wmin = WEIGHT_MIN
        self._wmax = WEIGHT_MAX
        self._decay = WEIGHT_DECAY
        self._load()

    def learn(
        self, regime: str, alpha: dict[str, float], won: bool, direction: int
    ) -> None:
        """One asymmetric gradient step on this regime's vector only."""
        with self._lock:
            vec = dict(self._vectors.get(regime, DEFAULT_WEIGHTS))
            factor = self._reward if won else -self._penalty
            for key in vec:
                signal_val = alpha.get(key, 0.0)
                delta = self._lr * factor * signal_val * direction
                vec[key] = max(self._wmin, min(self._wmax, vec[key] + delta))
            for key in vec:
                vec[key] *= self._decay
            get_enforcer().repair_on_load(vec)
            self._vectors[regime] = vec
            self._counts[regime] = self._counts.get(regime, 0) + 1
            self._save()

    def weights_for(self, regime: str) -> dict[str, float] | None:
        """This regime's vector once trained; None → use global weights."""
        with self._lock:
            if self._counts.get(regime, 0) < REGIME_MIN_TRADES:
                return None
            return dict(self._vectors.get(regime, {}))

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view: per-regime trade counts + active flags."""
        with self._lock:
            return {
                r: {
                    "n": self._counts.get(r, 0),
                    "active": self._counts.get(r, 0) >= REGIME_MIN_TRADES,
                }
                for r in sorted(set(self._counts) | set(self._vectors))
            }

    def _load(self) -> None:
        """Load persisted vectors; corrupt/missing file → fresh.

        Vectors are routed through the weight enforcer on load so drifted
        budgets (unbounded negative sums back-fill the cortex with a corrupt
        vector that zeroes the score via the old signed-sum guard) are
        clamped to the enforcer band and renormalized to |sum| ≈ 1 before
        they can reach the cortex. Repaired files are re-persisted.
        """
        if not self._path.exists():
            return
        repaired_any = False
        try:
            d = json.loads(self._path.read_text())
            for regime, vec in d.get("vectors", {}).items():
                if isinstance(vec, dict) and vec:
                    cleaned = {
                        k: float(v) for k, v in vec.items() if k in DEFAULT_WEIGHTS
                    }
                    repaired = get_enforcer().repair_on_load(cleaned)
                    self._vectors[regime] = cleaned
                    repaired_any = repaired_any or repaired
            for regime, n in d.get("counts", {}).items():
                self._counts[regime] = int(n)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("Regime weights load failed (fresh): %s", exc)
        if repaired_any:
            log.info("Regime weight vectors repaired on load; persisting")
            self._save()

    def _save(self) -> None:
        """Persist atomically (tmp + replace)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"vectors": self._vectors, "counts": self._counts})
            )
            tmp.replace(self._path)
        except OSError as exc:
            log.debug("Regime weights save failed: %s", exc)
