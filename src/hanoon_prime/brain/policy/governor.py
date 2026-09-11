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
        """(True, "ok") when the ticker *may* be decided as an entry this cycle.

        Only checks — does NOT consume the cycle budget.  Call ``claim_entry``
        *after* the entry is actually admitted so that downstream vetoes
        (e.g. ``not_sized``, ``portfolio_risk``) don't waste budget.
        """
        if self._cycle_used >= MAX_ENTRIES_PER_CYCLE:
            return False, "cycle_budget"
        if time.time() - self._last_entry.get(ticker, 0.0) < ENTRY_REUSE_COOLDOWN_SEC:
            return False, "reuse_cooldown"
        return True, "ok"

    def claim_entry(self) -> None:
        """Consume one unit of the per-cycle entry budget.

        Called only when ``decide_entry`` produces an ENTER verdict.
        """
        self._cycle_used += 1

    def note_entry(self, ticker: str) -> None:
        """Stamp the cooldown AFTER a real bracket was placed."""
        self._last_entry[ticker] = time.time()

    def note_exit(self, ticker: str) -> None:
        """Stamp the cooldown AFTER a position is closed or a close attempt fails.

        Prevents immediate re-entry on the same ticker after a stop-out or
        close-order death — the core whipsaw guard.  Shares the same
        ``_last_entry`` stamp so ``may_enter`` blocks reuse for
        ``ENTRY_REUSE_COOLDOWN_SEC`` seconds.
        """
        self._last_entry[ticker] = time.time()


__all__ = ["Governor"]
