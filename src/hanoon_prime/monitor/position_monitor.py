"""hanoon_prime.monitor.position_monitor — Position health monitor daemon.

Runs on a background thread, re-scores each open position through JULI's
live brain, feeds exit signals to the shared BrainState, and detects
stale ticks. The main loop reads exit signals from shared state.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from ..brain.shared_state import BrainState
from .exit_scoring import ExitScorer
from .watchdog import Watchdog

log = logging.getLogger(__name__)


class PositionMonitor:
    """Background daemon that monitors position health."""

    def __init__(self, state: BrainState, pulse_sec: float = 5.0) -> None:
        self.state = state
        self._pulse_sec = pulse_sec
        self.scorer = ExitScorer()
        self.watchdog = Watchdog()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start the monitor daemon thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="position-monitor"
        )
        self._thread.start()
        log.info("Position monitor started (pulse=%.0fs)", self._pulse_sec)

    def stop(self) -> None:
        """Stop the monitor daemon thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("Position monitor stopped")

    def _loop(self) -> None:
        """Main monitor loop."""
        while self._running:
            try:
                self._cycle()
            except Exception as e:
                log.error("Monitor cycle error: %s", e)
            time.sleep(self._pulse_sec)

    def _cycle(self) -> None:
        """One monitor cycle: check exits, check watchdog."""
        if self.watchdog.check_panic():
            self.state.update(panic_mode=True)
            log.warning("PANIC MODE: auto-flatten triggered")
            return
        stale = self.watchdog.check_stale()
        if stale:
            log.warning("Stale ticks: %s", stale)
        self.state.update(
            panic_mode=False,
            monitor_heartbeat=time.time(),
        )
