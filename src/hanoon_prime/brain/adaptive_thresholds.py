"""brain.adaptive_thresholds — learned, bounded exit thresholds (state).

State container + getters. Constants/learners live elsewhere so each file
stays under R3's 200-line cap even after black's 88-char expansion:
- ``threshold_limits`` holds DEFAULTS / BOUNDS / clamp / LR constants.
- ``adaptive_learning`` holds the nine update rules.

Thread-safe (RLock, single-writer); persistence is best-effort —
failures are logged, never raised (R15-compliant).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .adaptive_learning import apply_learning
from .threshold_limits import BOUNDS, DEFAULTS, LR_DOWN, LR_UP, PILLAR_KEYS, clamp

log = logging.getLogger(__name__)


class AdaptiveThresholds:
    """Learned exit thresholds with hard bounds (single-writer)."""

    def __init__(self, filepath: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self._values: dict[str, Any] = dict(DEFAULTS)
        self._tier_values: list[list[float]] = [
            list(t) for t in DEFAULTS["profit_lock_tiers"]
        ]
        self._update_count: int = 0
        self._path: Optional[Path] = filepath or self._persist_path()
        self._load()

    def get_exit_threshold(self, win_rate: float) -> float:
        """Exit threshold = base + scale × win_rate, clamped to [base, max]."""
        with self._lock:
            b = self._values["exit_base"]
            s = self._values["exit_scale"]
            mx = self._values["exit_max"]
        return clamp(b + s * win_rate, b, mx)

    def get_watch_threshold(self, win_rate: float) -> float:
        """WATCH fires earlier at 70% of the exit threshold."""
        return self.get_exit_threshold(win_rate) * 0.7

    def get_stale_minutes(self, tightened: bool = False) -> float:
        """Adaptive stale-force window (minutes)."""
        key = "stale_minutes_tighten" if tightened else "stale_minutes_force"
        with self._lock:
            return float(self._values[key])

    def get_stale_loss_pct(self, tightened: bool = False) -> float:
        """Adaptive stale loss-% floor."""
        key = "stale_loss_pct_tighten" if tightened else "stale_loss_pct_force"
        with self._lock:
            return float(self._values[key])

    def get_consolidation_thresholds(self) -> tuple[int, float]:
        """Return (pulses, min_profit_pct) for the consolidation exit."""
        with self._lock:
            return int(self._values["consolidation_pulses"]), float(
                self._values["consolidation_min_profit"]
            )

    def get_consolidation_flat_pct(self) -> float:
        """Adaptive flat-range % below which price is considered flat."""
        with self._lock:
            return float(self._values["consolidation_flat_pct"])

    def get_ride_winners_params(self) -> tuple[float, float]:
        """Return (min_pnl_pct, max_exit_likelihood) for ride-winners."""
        with self._lock:
            return float(self._values["ride_winners_pnl"]), float(
                self._values["ride_winners_exit_lik"]
            )

    def get_trail_keep_ratio(self) -> float:
        """Adaptive trailing-stop keep-ratio (fraction of peak gain held)."""
        with self._lock:
            return float(self._values["trail_keep_ratio"])

    def get_profit_lock_tiers(self) -> list[tuple[float, float]]:
        """Adaptive profit-lock tiers as (min_gain_pct, locked_min_pct)."""
        with self._lock:
            return [(t[0], t[1]) for t in self._tier_values]

    def get_exit_pillar_weights(self) -> dict[str, float]:
        """Advisory weights for the exit-likelihood pillars."""
        with self._lock:
            return {k: float(self._values[k]) for k in PILLAR_KEYS}

    def get_value(self, key: str) -> float:
        """Generic typed getter for any scalar threshold."""
        with self._lock:
            return float(self._values.get(key, DEFAULTS.get(key, 0.0)))

    def update_from_outcome(
        self,
        won: bool,
        hold_minutes: float,
        pnl_pct: float,
        peak_pct: float,
        exit_reason: str,
        exit_likelihood: float,
    ) -> None:
        """Retune thresholds from one closed trade (delegates to learners)."""
        with self._lock:
            apply_learning(
                self._adjust,
                self._tier_values,
                won=won,
                hold_minutes=hold_minutes,
                pnl_pct=pnl_pct,
                peak_pct=peak_pct,
                exit_reason=exit_reason,
                exit_likelihood=exit_likelihood,
            )
            self._update_count += 1
            self._save()

    def _adjust(self, key: str, delta: float) -> None:
        """Adjust one threshold, clamped to its bounds."""
        cur = self._values.get(key, DEFAULTS.get(key, 0.0))
        low, high = BOUNDS.get(key, (0.0, 1.0))
        self._values[key] = clamp(cur + delta, low, high)

    def telemetry(self) -> dict[str, Any]:
        """Snapshot of learned thresholds for telemetry."""
        with self._lock:
            return {
                "values": dict(self._values),
                "tier_values": [list(t) for t in self._tier_values],
                "update_count": self._update_count,
            }

    def _persist_path(self) -> Optional[Path]:
        from .config import STATE_DIR

        return STATE_DIR / "adaptive_thresholds.json"

    def _save(self) -> None:
        """Best-effort persistence; log on failure (never raise)."""
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "values": self._values,
                "tier_values": self._tier_values,
                "update_count": self._update_count,
                "ts": time.time(),
            }
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            log.debug("adaptive_thresholds save failed: %s", exc)

    def _load(self) -> None:
        """Restore thresholds from disk; log on failure (never raise)."""
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for k, v in data.get("values", {}).items():
                if k in DEFAULTS and not isinstance(v, (list, dict)):
                    low, high = BOUNDS.get(k, (0.0, 1.0))
                    self._values[k] = clamp(float(v), low, high)
            tiers = data.get("tier_values", [])
            if tiers and len(tiers) == len(DEFAULTS["profit_lock_tiers"]):
                self._tier_values = [list(t) for t in tiers]
            self._update_count = data.get("update_count", 0)
        except (OSError, ValueError, TypeError) as exc:
            log.debug("adaptive_thresholds load failed: %s", exc)


_thresholds: Optional[AdaptiveThresholds] = None
_lock = threading.Lock()


def get_adaptive_thresholds() -> AdaptiveThresholds:
    """Thread-safe singleton accessor."""
    global _thresholds
    if _thresholds is None:
        with _lock:
            _thresholds = _thresholds or AdaptiveThresholds()
    return _thresholds


__all__ = ["AdaptiveThresholds", "get_adaptive_thresholds"]
