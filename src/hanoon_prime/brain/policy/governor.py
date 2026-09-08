"""brain.policy.governor — entry pacing, not thinking.

Separate pacing from scoring: the governor caps how many entries may be
decided per cycle and how often a ticker may be re-entered. Single-threaded,
fast-path only, holds no locks.
"""

from __future__ import annotations

import time

from ...immune import ENTRY_REUSE_COOLDOWN_SEC, MAX_ENTRIES_PER_CYCLE


class Governor:
    """Per-cycle entry budget + per-ticker reuse cooldown."""

    def __init__(self) -> None:
        """Zero budget; cooldown timestamps empty."""
        self._cycle_used: int = 0
        self._last_entry: dict[str, float] = {}

    def begin_cycle(self) -> None:
        """Reset the per-cycle admitted count at the start of juli.tick."""
        self._cycle_used = 0

    def may_enter(self, ticker: str) -> tuple[bool, str]:
        """(True, "ok") when the ticker may be decided as an entry this cycle.

        Consumes the cycle budget on approval; the reuse cooldown is keyed
        by ticker and only written when a bracket is actually placed.
        """
        if self._cycle_used >= MAX_ENTRIES_PER_CYCLE:
            return False, "cycle_budget"
        if time.time() - self._last_entry.get(ticker, 0.0) < ENTRY_REUSE_COOLDOWN_SEC:
            return False, "reuse_cooldown"
        self._cycle_used += 1
        return True, "ok"

    def note_entry(self, ticker: str) -> None:
        """Stamp the cooldown AFTER a real bracket was placed."""
        self._last_entry[ticker] = time.time()


__all__ = ["Governor"]
