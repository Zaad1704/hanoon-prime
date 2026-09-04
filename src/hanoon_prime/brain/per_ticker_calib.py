"""brain.per_ticker_calib — Per-ticker confidence→WR calibration.

Tracks confidence→win-rate curves per ticker. A ticker's calibration
is blended with the global curve: per-ticker * 0.4 + global * 0.6.
Catches ticker-specific overconfidence (volatile biotech vs stable ETF).

Source: rebuild's per_ticker_calib.py (lines 1-200).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_MIN_SAMPLES: int = 10
_BLEND: float = 0.4  # per-ticker weight
_BINS: int = 5
_STATE_PATH = Path("runtime/per_ticker_calib.json")
_LOCK = threading.Lock()


class PerTickerCalibration:
    """Per-ticker confidence→win-rate calibration."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._data: dict[str, list[tuple[float, bool]]] = {}
        self._last_save: float = 0.0
        self._load()

    def record(self, ticker: str, confidence: float, won: bool) -> None:
        """Record a (confidence, outcome) pair for a ticker."""
        if not ticker:
            return
        self._data.setdefault(ticker, []).append((confidence, won))
        if len(self._data[ticker]) > 500:
            self._data[ticker] = self._data[ticker][-500:]
        self._save()

    def get_curve(
        self, ticker: str, global_curve: Optional[dict[float, float]] = None
    ) -> dict[float, float]:
        """Get blended calibration curve for a ticker."""
        data = self._data.get(ticker, [])
        if len(data) < _MIN_SAMPLES:
            return global_curve or {0.55: 0.5, 0.65: 0.5, 0.75: 0.5, 0.85: 0.5}
        ticker_curve = self._compute_bins(data)
        if global_curve is None:
            return ticker_curve
        blended = {}
        for key in set(ticker_curve) | set(global_curve):
            t = ticker_curve.get(key, 0.5)
            g = global_curve.get(key, 0.5)
            blended[key] = t * _BLEND + g * (1 - _BLEND)
        return blended

    def _compute_bins(self, data: list[tuple[float, bool]]) -> dict[float, float]:
        """Compute confidence→WR bins from data."""
        if not data:
            return {}
        sorted_d = sorted(data, key=lambda x: x[0])
        n = len(sorted_d)
        bin_size = max(1, n // _BINS)
        bins: dict[float, float] = {}
        for i in range(0, n, bin_size):
            chunk = sorted_d[i : i + bin_size]
            if not chunk:
                continue
            avg = sum(c for c, _ in chunk) / len(chunk)
            wr = sum(1 for _, w in chunk if w) / len(chunk)
            bins[round(avg * 10) / 10.0] = wr
        return bins

    def calibrate(
        self,
        ticker: str,
        raw_confidence: float,
        global_curve: Optional[dict[float, float]] = None,
    ) -> float:
        """Adjust confidence using per-ticker calibration."""
        curve = self.get_curve(ticker, global_curve)
        best_bin = min(curve.keys(), key=lambda b: abs(b - raw_confidence))
        realized_wr = curve[best_bin]
        pull = 0.3
        cal = raw_confidence + (realized_wr - raw_confidence) * pull
        return max(0.05, min(0.95, cal))

    def get_stats(self, ticker: str) -> dict[str, Any]:
        """Get calibration stats for a ticker."""
        data = self._data.get(ticker, [])
        if not data:
            return {"n": 0, "wr": 0.5}
        n = len(data)
        wins = sum(1 for _, w in data if w)
        return {"n": n, "wr": wins / n if n > 0 else 0.5}

    def _save(self) -> None:
        """Debounced save to disk."""
        now = time.time()
        if now - self._last_save < 60.0:
            return
        self._last_save = now
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {"ticker_data": self._data}
            tmp = _STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(_STATE_PATH)
        except Exception as exc:
            log.debug("Per-ticker save failed: %s", exc)

    def _load(self) -> None:
        """Load from disk."""
        if not _STATE_PATH.exists():
            return
        try:
            data = json.loads(_STATE_PATH.read_text())
            self._data = data.get("ticker_data", {}) or {}
            self._last_save = time.time()
        except Exception as exc:
            log.debug("Per-ticker load failed: %s", exc)


_percal: Optional[PerTickerCalibration] = None


def get_per_ticker_calib() -> PerTickerCalibration:
    """Get the singleton per-ticker calibration instance."""
    global _percal
    if _percal is None:
        _percal = PerTickerCalibration()
    return _percal
