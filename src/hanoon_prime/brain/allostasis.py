"""brain.allostasis — homeostatic setpoints and interoceptive alarm.

Biological grounding: Sterling (2012) — allostasis predicts needs instead
of reacting; Keramati & Gutkin (2014) — homeostatic reinforcement learning.
Each trading regime keeps its own learned "expected edge" setpoint, moved
slowly from realized outcome statistics (a rolling norm, not a fixed
constant). The interoceptive signal is the deviation of the realized edge
from that regime's setpoint; sustained deviation beyond ``ALLOS_MARGIN``
for ``ALLOS_VIOLATION_MIN`` learned updates trips dyshomeostasis — the
brain then tightens its discipline aggressively, like cortisol under
chronic stress.

Advisory only: the module never changes entries directly. It feeds
``pillar_awareness`` (dynamic fallen threshold), ``Dynamics`` (aggressive
threshold tightening on dyshomeostasis) and shared state (telemetry +
webapp). Persists to ``runtime/juli_allostasis.json`` (``HANOO_ALLOSTASIS_FILE``
override keeps tests hermetic).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .learning_config import (
    ALLOS_ALPHA,
    ALLOS_FILE,
    ALLOS_MARGIN,
    ALLOS_MIN_TRADES,
    ALLOS_VIOLATION_MIN,
)

log = logging.getLogger(__name__)


def _default_state() -> dict[str, Any]:
    """Fresh per-regime homeostatic state (break-even prior)."""
    return {
        "setpoint": 0.0,
        "last_edge": 0.0,
        "deviation": 0.0,
        "violations": 0,
        "n": 0,
    }


class AllostaticController:
    """Per-regime expected-edge setpoints with dyshomeostasis detection."""

    def __init__(self, filepath: Path | None = None, persist: bool = True) -> None:
        self._path: Path | None = None
        if filepath is not None:
            self._path = Path(filepath)
        elif persist:
            env_file = os.environ.get("HANOO_ALLOSTASIS_FILE", "").strip()
            self._path = Path(env_file) if env_file else ALLOS_FILE
        self._lock = threading.RLock()
        self._regimes: dict[str, dict[str, Any]] = {}
        if self._path is not None:
            self._load()

    def update(self, record: dict[str, Any] | None, regime: str) -> dict[str, Any]:
        """Advance one regime's setpoint from a win/loss record.

        The setpoint is only adapted once the regime has enough closes
        (``ALLOS_MIN_TRADES``); deviation is the signed edge gap, and
        repeats beyond ``ALLOS_MARGIN`` accumulate toward dyshomeostasis.
        """
        edge = float((record or {}).get("edge", 0.0))
        trades = int((record or {}).get("trades", 0) or 0)
        with self._lock:
            key = regime or "unknown"
            s = self._regimes.setdefault(key, _default_state())
            s["n"] = trades
            s["last_edge"] = edge
            if trades >= ALLOS_MIN_TRADES:
                s["setpoint"] += ALLOS_ALPHA * (edge - s["setpoint"])
            s["deviation"] = edge - float(s["setpoint"])
            strained = trades >= ALLOS_MIN_TRADES and abs(s["deviation"]) > ALLOS_MARGIN
            s["violations"] = s["violations"] + 1 if strained else 0
            state = self._snapshot_one(key)
            self._save()
            return state

    def setpoint(self, regime: str) -> dict[str, Any]:
        """Telemetry for one regime (never raises on an unknown key)."""
        with self._lock:
            return self._snapshot_one(regime or "unknown")

    def is_dyshomeostatic(self, regime: str) -> bool:
        """Sustained setpoint deviation beyond the margin, per regime."""
        with self._lock:
            return bool(self._snapshot_one(regime or "unknown")["dyshomeostatic"])

    def snapshot(self) -> dict[str, Any]:
        """Copy of every regime's homeostatic state (no locks held)."""
        with self._lock:
            return {key: self._snapshot_one(key) for key in sorted(self._regimes)}

    def _snapshot_one(self, key: str) -> dict[str, Any]:
        s = self._regimes.get(key) or _default_state()
        return {
            "regime": key,
            "setpoint": round(float(s["setpoint"]), 4),
            "deviation": round(float(s["deviation"]), 4),
            "dyshomeostatic": bool(s["violations"] >= ALLOS_VIOLATION_MIN),
            "violations": int(s["violations"]),
            "trades": int(s["n"]),
        }

    def _load(self) -> None:
        """Restore persisted states (best-effort; fresh on corruption)."""
        if self._path is None or not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            self._regimes = {
                str(k): dict(v) for k, v in dict(d.get("regimes", {})).items()
            }
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            log.warning("Allostasis load failed (starting fresh): %s", e)

    def _save(self) -> None:
        """Atomically persist every regime's homeostatic state."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"regimes": self._regimes}, default=str))
        tmp.replace(self._path)
