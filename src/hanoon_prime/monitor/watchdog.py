"""hanoon_prime.monitor.watchdog — Stale tick detection + panic mode.

Monitors tick freshness and triggers panic mode (auto-flatten) when
ticks go stale for too long. Runs on the monitor daemon thread.
"""
from __future__ import annotations

import time
from typing import Any

STALE_THRESHOLD: float = 30.0
PANIC_THRESHOLD: float = 60.0


class Watchdog:
    """Stale tick detection + panic mode."""

    def __init__(self) -> None:
        self._tick_times: dict[str, float] = {}
        self._panic_mode: bool = False
        self._last_heartbeat: float = time.time()

    def tick_received(self, ticker: str) -> None:
        """Record that a tick was received for this ticker."""
        self._tick_times[ticker] = time.time()
        self._last_heartbeat = time.time()

    def check_stale(self) -> list[str]:
        """Return list of stale tickers."""
        now = time.time()
        stale = []
        for ticker, ts in self._tick_times.items():
            if now - ts > STALE_THRESHOLD:
                stale.append(ticker)
        return stale

    def check_panic(self) -> bool:
        """Check if panic mode should be activated."""
        if self._panic_mode:
            return True
        now = time.time()
        if now - self._last_heartbeat > PANIC_THRESHOLD:
            self._panic_mode = True
            return True
        return False

    def clear_panic(self) -> None:
        """Clear panic mode."""
        self._panic_mode = False

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view."""
        return {
            "panic_mode": self._panic_mode,
            "tickers_tracked": len(self._tick_times),
            "stale_count": len(self.check_stale()),
            "last_heartbeat_age": round(time.time() - self._last_heartbeat, 1),
        }
