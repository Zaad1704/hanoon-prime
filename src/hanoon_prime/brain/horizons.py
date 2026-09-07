"""brain.horizons — multi-horizon ladder (rebuild horizon.py port).

Six horizons classified from bars (never blocks), driving per-horizon
stop/target multipliers and exit windows. ``scalp`` enabled by default;
the webapp activates more via /config horizons (runtime/horizons.json).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import STATE_DIR

log = logging.getLogger(__name__)

HORIZONS = ("scalp", "multihour", "swing", "multiday", "multiweek", "longterm")
_STATE_PATH: Path = STATE_DIR / "horizons.json"

# ATR% (mean true range / price) → horizon preference (rebuild parity)
_ATR_RUNGS = (0.008, 0.015, 0.03, 0.05, 0.08, 1.0)
_HORIZON_ATR = tuple(zip(HORIZONS, _ATR_RUNGS))
_STEP_UP: dict[str, str] = dict(zip(HORIZONS, HORIZONS[1:] + ("longterm",)))
_STEP_DOWN: dict[str, str] = {v: k for k, v in _STEP_UP.items()}
# Bounded Halim nudge triggers
_CATALYSTS = ("catalyst", "earnings", "guidance", "upgrade", "breakout", "news")
_RISK_WORDS = ("risk", "uncertain", "warning")


@dataclass(frozen=True)
class HorizonParams:
    """Per-horizon trading parameters (rebuild HORIZON_CONFIG parity)."""

    atr_stop_mult: float
    atr_target_mult: float
    stale_minutes: float
    patience: float  # entry-bar divisor (longer horizon = lower bar)

    @property
    def rr(self) -> float:
        """Realized reward:risk of the ATR multipliers."""
        return self.atr_target_mult / self.atr_stop_mult


HORIZON_PARAMS: dict[str, HorizonParams] = {
    "scalp": HorizonParams(2.0, 6.0, 120.0, 1.00),
    "multihour": HorizonParams(2.5, 7.5, 240.0, 0.97),
    "swing": HorizonParams(3.0, 9.0, 480.0, 0.94),
    "multiday": HorizonParams(3.5, 10.5, 960.0, 0.92),
    "multiweek": HorizonParams(4.0, 12.0, 1440.0, 0.90),
    "longterm": HorizonParams(4.0, 12.0, 2880.0, 0.90),
}


def _halim_nudge(base: str, verdict: dict[str, Any] | None) -> str:
    """Bounded one-step horizon nudge from Halim's verdict text."""
    if not verdict:
        return base
    conf = float(verdict.get("confidence", 0.0) or 0.0)
    reason = str(verdict.get("reasoning", "") or "").lower()
    if conf >= 0.6 and any(w in reason for w in _CATALYSTS):
        return _STEP_UP.get(base, base)
    if conf >= 0.7 and any(w in reason for w in _RISK_WORDS):
        return _STEP_DOWN.get(base, base)
    return base


def classify(
    close: np.ndarray | list[float],
    high: np.ndarray | list[float] | None = None,
    low: np.ndarray | list[float] | None = None,
    regime: str = "unknown",
    halim_verdict: dict[str, Any] | None = None,
) -> str:
    """Classify horizon from bars + intel (degenerate input → ``scalp``).

    Blend (rebuild parity): ATR%-base → consistency bump → regime shrink
    → bounded one-step Halim nudge.
    """
    try:
        c = np.asarray(close, dtype=np.float64)
        if len(c) < 10 or float(c[-1]) <= 0:
            return "scalp"
        h = np.asarray(high, dtype=np.float64) if high is not None else c
        lo = np.asarray(low, dtype=np.float64) if low is not None else c
        tr = np.maximum(
            h[1:] - lo[1:],
            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])),
        )
        atr_pct = float(np.mean(tr)) / float(c[-1])
        rets = np.diff(c) / np.maximum(np.abs(c[:-1]), 1e-9)
        consistency = (
            abs(float(np.sum(np.sign(rets[-8:])))) / 8.0 if len(rets) >= 8 else 0.0
        )
        base = next((n for n, thr in _HORIZON_ATR if atr_pct <= thr), "scalp")
        if consistency >= 0.6:
            base = _STEP_UP.get(base, base)
        if regime in ("ranging", "volatile"):
            base = _STEP_DOWN.get(base, base)
        base = _halim_nudge(base, halim_verdict)
        return base if base in HORIZONS else "scalp"
    except Exception:
        return "scalp"


def params_for(horizon: str) -> HorizonParams:
    """Per-horizon parameters (scalp params for unknown names)."""
    return HORIZON_PARAMS.get(horizon, HORIZON_PARAMS["scalp"])


def holds_through_close(horizon: str) -> bool:
    """True for horizons that must survive EOD flatten (rebuild v4 lesson)."""
    return horizon in ("multiday", "multiweek", "longterm")


class HorizonManager:
    """Enabled-horizon state (webapp-facing) + closest-active mapping."""

    def __init__(self, enabled: set[str] | None = None) -> None:
        self._lock = threading.RLock()
        self._enabled: set[str] = enabled or {"scalp"}
        self._load()
        if not self._enabled:
            self._enabled = {"scalp"}

    def enabled(self) -> list[str]:
        """Sorted list of enabled horizons."""
        with self._lock:
            return sorted(self._enabled)

    def set_enabled(self, horizons: set[str] | list[str] | str) -> list[str]:
        """Set enabled horizons (set/list/comma-string). Empty keeps current."""
        if isinstance(horizons, str):
            horizons = {h.strip() for h in horizons.split(",") if h.strip()}
        with self._lock:
            self._enabled = (
                {h for h in horizons if h in HORIZONS} if horizons else self._enabled
            )
            if not self._enabled:
                self._enabled = {"scalp"}  # safe default
            self._save()
            return sorted(self._enabled)

    def active(self, horizon: str) -> str:
        """Map a classified horizon to the closest ENABLED horizon."""
        with self._lock:
            if horizon in self._enabled:
                return horizon
            idx = HORIZONS.index(horizon) if horizon in HORIZONS else 0
            ranked = sorted((abs(HORIZONS.index(n) - idx), n) for n in self._enabled)
            return ranked[0][1] if ranked else "scalp"

    def _save(self) -> None:
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(sorted(self._enabled)))
            tmp.replace(_STATE_PATH)
        except Exception as exc:
            log.debug("horizon save failed: %s", exc)

    def _load(self) -> None:
        try:
            if _STATE_PATH.exists():
                data = json.loads(_STATE_PATH.read_text())
                if isinstance(data, list):
                    self._enabled = {h for h in data if h in HORIZONS}
        except Exception as exc:
            log.debug("horizon load failed: %s", exc)


_manager: HorizonManager | None = None


def get_horizon_manager() -> HorizonManager:
    """Process-wide horizon manager singleton."""
    global _manager
    if _manager is None:
        _manager = HorizonManager()
    return _manager


__all__ = [
    "HORIZONS",
    "HORIZON_PARAMS",
    "HorizonManager",
    "HorizonParams",
    "classify",
    "get_horizon_manager",
    "holds_through_close",
    "params_for",
]
