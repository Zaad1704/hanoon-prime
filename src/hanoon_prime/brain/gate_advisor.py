"""brain.gate_advisor — realized-WR entry-bar tuning (Halim gate-advisor port).

Single-writer learning loop member: the orchestrator calls
``record_outcome(won)`` on every real trade close; ``threshold_delta()``
and ``is_tightening()`` are read on every tick. The advisor RAISES the
entry bar after losing streaks and RELIEVES it after winning ones —
bounded so it can never deadlock a fresh brain (thin data → 0.0).

This is advisory by construction: it tunes the bar the verdict must clear
and the size of admitted entries. It never produces, flips, or blocks a
verdict — cortex remains the sole verdict source (R1).
"""

from __future__ import annotations

import threading
import time
from collections import deque

from .config import (
    ADVISOR_DELTA_MAX,
    ADVISOR_LOOSEN_MAX,
    ADVISOR_LOOSEN_STEP,
    ADVISOR_LOOSEN_WR,
    ADVISOR_MIN_TRADES,
    ADVISOR_TIGHTEN_STEP,
    ADVISOR_TIGHTEN_WR,
)
from .realized_ev import RealizedStats

_WINDOW: int = 20


class GateAdvisor:
    """Realized-win-rate threshold advisor (bounded, always-on)."""

    def __init__(self, realized: RealizedStats | None = None) -> None:
        self._realized = realized
        self._recent: deque[bool] = deque(maxlen=_WINDOW)
        self._delta: float = 0.0
        self._lock = threading.RLock()

    def record_outcome(self, won: bool) -> None:
        """Feed one real trade outcome; retune the delta from the window."""
        with self._lock:
            self._recent.append(bool(won))
            n = len(self._recent)
            if n < ADVISOR_MIN_TRADES:
                self._delta = 0.0
                return
            wr = sum(1 for w in self._recent if w) / n
            if wr < ADVISOR_TIGHTEN_WR:
                self._delta = min(ADVISOR_DELTA_MAX, self._delta + ADVISOR_TIGHTEN_STEP)
            elif wr > ADVISOR_LOOSEN_WR:
                self._delta = max(
                    -ADVISOR_LOOSEN_MAX, self._delta - ADVISOR_LOOSEN_STEP
                )
            else:
                self._delta = 0.0

    def threshold_delta(self) -> float:
        """Current entry-bar raise (subtracted from the score pipeline)."""
        with self._lock:
            return round(self._delta, 4)

    def is_tightening(self) -> bool:
        """True while the learned gate demands half-size entries."""
        with self._lock:
            return self._delta >= ADVISOR_TIGHTEN_STEP

    def snapshot(self) -> dict[str, float | int | bool]:
        """Telemetry view."""
        with self._lock:
            n = len(self._recent)
            wr = (sum(1 for w in self._recent if w) / n) if n else 0.0
            return {
                "delta": round(self._delta, 4),
                "window": n,
                "recent_wr": round(wr, 4),
                "tightening": self.is_tightening(),
                "ts": time.time(),
            }
