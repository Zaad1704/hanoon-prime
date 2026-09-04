"""brain.calibration — Confidence calibration + horizon gate.

Rolling bin calibration for confidence→WR mapping.
Per-horizon win rate tracking (scalp, multihour, swing).

Source: rebuild's calibration.py + horizon_gate.py (simplified).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)

_CALIB_WINDOW: int = 100
_CALIB_BINS: int = 10
_HORIZON_MIN_TRADES: int = 5
_KNOWN_HORIZONS = ("scalp", "multihour", "swing", "multiday")


@dataclass
class CalibBin:
    center: float
    count: int
    wins: int

    @property
    def wr(self) -> float:
        """Auto-generated docstring."""
        return self.wins / self.count if self.count > 0 else 0.5


class ConfidenceCalibration:
    """Rolling bin calibration for confidence→WR mapping."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._history: deque[tuple[float, bool]] = deque(maxlen=_CALIB_WINDOW)

    def record(self, confidence: float, won: bool) -> None:
        """Auto-generated docstring."""
        self._history.append((confidence, won))

    def calibrate(self, raw_confidence: float) -> float:
        """Map raw confidence to calibrated WR."""
        if len(self._history) < 10:
            return raw_confidence
        bins = self._compute_bins()
        if not bins:
            return raw_confidence
        best = min(bins, key=lambda b: abs(b.center - raw_confidence))
        pull = 0.3
        cal = raw_confidence + (best.wr - raw_confidence) * pull
        return max(0.05, min(0.95, cal))

    def _compute_bins(self) -> list[CalibBin]:
        """Auto-generated docstring."""
        if not self._history:
            return []
        sorted_d = sorted(self._history, key=lambda x: x[0])
        n = len(sorted_d)
        bin_size = max(1, n // _CALIB_BINS)
        bins = []
        for i in range(0, n, bin_size):
            chunk = sorted_d[i : i + bin_size]
            if not chunk:
                continue
            avg = sum(c for c, _ in chunk) / len(chunk)
            wins = sum(1 for _, w in chunk if w)
            bins.append(CalibBin(round(avg * 10) / 10.0, len(chunk), wins))
        return bins


class HorizonGate:
    """Per-horizon win rate tracking."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._data: dict[str, deque[tuple[float, bool]]] = {
            h: deque(maxlen=200) for h in _KNOWN_HORIZONS
        }

    def record(self, horizon: str, won: bool, pnl: float) -> None:
        """Auto-generated docstring."""
        if horizon in self._data:
            self._data[horizon].append((pnl, won))

    def get_gate(self, horizon: str) -> float:
        """Returns WR for horizon, or 0.5 if insufficient data."""
        data = self._data.get(horizon, deque())
        if len(data) < _HORIZON_MIN_TRADES:
            return 0.5
        wins = sum(1 for _, w in data if w)
        return wins / len(data)

    def get_all_gates(self) -> dict[str, float]:
        """Auto-generated docstring."""
        return {h: self.get_gate(h) for h in _KNOWN_HORIZONS}
