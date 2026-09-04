"""ops.tbt — Time-Based Trigger rotation.

Rotates time-based triggers for multi-timeframe analysis.

Source: rebuild's senses/live_feed/tbt_rotation.py (simplified).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)

_TIMEFRAMES = [60, 300, 900, 3600]  # 1m, 5m, 15m, 1h


@dataclass
class TBTSnapshot:
    timeframe: int
    trigger_active: bool = False
    signal_strength: float = 0.0


class TBTRotation:
    """Time-Based Trigger rotation for multi-timeframe analysis."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._bars: dict[int, deque[float]] = {
            tf: deque(maxlen=100) for tf in _TIMEFRAMES
        }
        self._last_update: dict[int, float] = {tf: 0.0 for tf in _TIMEFRAMES}

    def update(self, timeframe: int, close: float) -> TBTSnapshot:
        """Update with new bar data."""
        if timeframe not in self._bars:
            return TBTSnapshot(timeframe)
        self._bars[timeframe].append(close)
        self._last_update[timeframe] = time.time()
        bars = list(self._bars[timeframe])
        if len(bars) < 5:
            return TBTSnapshot(timeframe)
        # Simple momentum trigger
        recent = bars[-5:]
        momentum = (recent[-1] - recent[0]) / recent[0] if recent[0] else 0
        active = abs(momentum) > 0.005
        return TBTSnapshot(timeframe, active, momentum)

    def get_all_triggers(self) -> dict[int, TBTSnapshot]:
        """Auto-generated docstring."""
        return {
            tf: self.update(tf, self._bars[tf][-1] if self._bars[tf] else 0)
            for tf in _TIMEFRAMES
        }
