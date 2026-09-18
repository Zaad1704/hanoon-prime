"""monitor.exec_quality — execution-quality gauges: slippage, fill, ack latency."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any

import numpy as np

MAX_SAMPLES: int = 500  # rolling window for slippage / ack percentiles


class ExecQuality:
    """Bounded sample aggregator for live entry-fill execution quality.

    Each live bracket entry fills once; we compare the actual fill to the
    intended signal price (slippage bps), the intended size (fill ratio),
    and the time from placement to position confirmation (ack latency).
    All samples are dropped off a fixed-size deque; summary() gives
    percentiles suitable for a single-glance gauge.
    """

    def __init__(self, maxlen: int = MAX_SAMPLES) -> None:
        self._lock = threading.Lock()
        self._slip_bps: deque[float] = deque(maxlen=maxlen)
        self._ack_ms: deque[float] = deque(maxlen=maxlen)
        self._fills: int = 0

    def record(
        self,
        direction: int,
        expected: float,
        actual: float,
        submitted: float,
        confirmed: float,
    ) -> None:
        """Record one entry fill: bps slippage + ack latency (epoch seconds)."""
        if not math.isfinite(expected) or expected <= 0.0:
            return
        if not math.isfinite(actual) or actual <= 0.0:
            return
        slip = (actual - expected) / expected * 1e4
        if direction < 0:
            slip = -slip
        ack = max(0.0, (confirmed - submitted) * 1000.0)
        with self._lock:
            if math.isfinite(slip):
                self._slip_bps.append(slip)
            self._ack_ms.append(ack)
            self._fills += 1

    def summary(self) -> dict[str, Any]:
        """Percentile summary of slippage bps and ack latency."""
        with self._lock:
            slips = list(self._slip_bps)
            acks = list(self._ack_ms)
            fills = self._fills
        return {
            "fills": fills,
            "slippage_bps_mean": _pmean(slips),
            "slippage_bps_p50": _pctile(slips, 50),
            "slippage_bps_p95": _pctile(slips, 95),
            "ack_ms_mean": _pmean(acks),
            "ack_ms_p50": _pctile(acks, 50),
            "ack_ms_p95": _pctile(acks, 95),
            "last_updated": time.time(),
        }


def _pctile(values: list[float], p: float) -> float:
    """Return the p-th percentile (0 when empty)."""
    if not values:
        return 0.0
    return round(float(np.percentile(values, p)), 2)


def _pmean(values: list[float]) -> float:
    """Return the arithmetic mean (0 when empty)."""
    if not values:
        return 0.0
    return round(float(np.mean(values)), 2)
