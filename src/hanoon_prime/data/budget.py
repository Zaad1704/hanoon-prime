"""hanoon_prime.data.budget — IB data subscription budget manager.

Allocates finite IB data slots (TBT, DOM, L1) across open positions
and scanner candidates. Open positions always get priority; candidate
seats are rotated across the whole discovered pool so every ticker
eventually gets live data + a real score within IB's line allowance.

IB Limits:
- TBT (Tick-by-Tick): ~10 tickers max
- DOM (Level 2): 5-60 tickers
- L1 (reqMktData): 100 tickers default
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MAX_TBT: int = 15
MAX_DOM: int = 20
MAX_L1: int = 100
# Rotation batch: candidates re-admitted per allocate cycle so streamed
# seats pivot across the pool over time (bounded churn on IB lines).
ROTATE_PER_CYCLE: int = 20


@dataclass
class SubSlot:
    """One data subscription slot."""

    ticker: str
    tier: str  # "TBT", "DOM", "L1"
    is_position: bool
    allocated_at: float = field(default_factory=time.time)


class DataBudget:
    """Manages IB data subscription allocations (LRU seat rotation)."""

    def __init__(self) -> None:
        self.slots: dict[str, SubSlot] = {}
        self._last_alloc: float = 0.0
        self._last_seen: dict[str, float] = {}  # last serve time per candidate

    def allocate(
        self,
        positions: set[str],
        candidates: list[str],
    ) -> tuple[dict[str, str], set[str]]:
        """Compute target subscriptions. Returns (to_sub, to_unsub)."""
        for sym in positions:
            self._last_seen.setdefault(sym, time.time())
        target: dict[str, str] = {}
        tbt_count = 0
        dom_count = 0
        l1_count = 0

        for sym in positions:
            if tbt_count < MAX_TBT:
                target[sym] = "TBT"
                tbt_count += 1
            elif dom_count < MAX_DOM:
                target[sym] = "DOM"
                dom_count += 1

        for sym in self._rotated(candidates):
            if sym in target:
                continue
            if tbt_count < MAX_TBT:
                target[sym] = "TBT"
                tbt_count += 1
            elif l1_count < MAX_L1:
                target[sym] = "L1"
                l1_count += 1

        to_sub, to_unsub = self._diff(target)
        self._apply(target, positions)
        return to_sub, to_unsub

    def _rotated(self, candidates: list[str]) -> list[str]:
        """Order candidates so un/under-served names get seats first.

        New names and least-recently-served names lead the seat fill,
        while currently-slotted names keep their seat unless the pool
        outgrows capacity — then the OLDEST slots rotate out in batches
        of ROTATE_PER_CYCLE to admit never-served ones. Every discovered
        ticker is eventually streamed with enough bars for a real score.
        """
        keep = {c for c in candidates if c in self.slots}
        fresh = sorted(keep, key=lambda c: self.slots[c].allocated_at, reverse=True)
        hungry = sorted(
            (c for c in candidates if c not in keep),
            key=lambda c: self._last_seen.get(c, 0.0),
        )[:ROTATE_PER_CYCLE]
        return hungry + fresh

    def _diff(self, target: dict[str, str]) -> tuple[dict[str, str], set[str]]:
        """Compare current slots with target. Returns only actual changes."""
        to_sub: dict[str, str] = {}
        to_unsub: set[str] = set()
        for sym, slot in list(self.slots.items()):
            if sym not in target:
                to_unsub.add(sym)
            elif slot.tier != target[sym]:
                to_sub[sym] = target[sym]
        for sym, tier in target.items():
            if sym not in self.slots:
                to_sub[sym] = tier
        return to_sub, to_unsub

    def _apply(self, target: dict[str, str], positions: set[str]) -> None:
        """Update internal slot state, preserving existing non-changing slots."""
        new_slots: dict[str, SubSlot] = {}
        for sym, tier in target.items():
            is_pos = sym in positions
            if sym in self.slots and self.slots[sym].tier == tier:
                new_slots[sym] = self.slots[sym]
            else:
                new_slots[sym] = SubSlot(ticker=sym, tier=tier, is_position=is_pos)
            self._last_seen[sym] = time.time()
        self.slots = new_slots
        self._last_alloc = time.time()

    def get_tbt_tickers(self) -> list[str]:
        """Return tickers with TBT premium data."""
        return [s.ticker for s in self.slots.values() if s.tier == "TBT"]

    def get_all_tracked(self) -> set[str]:
        """Return all tracked tickers."""
        return set(self.slots.keys())

    def remove(self, ticker: str) -> None:
        """Remove a ticker from tracking."""
        self.slots.pop(ticker, None)
        self._last_seen.pop(ticker, None)

    def count_tiers(self) -> dict[str, int]:
        """Count subscriptions per tier."""
        counts: dict[str, int] = {}
        for slot in self.slots.values():
            counts[slot.tier] = counts.get(slot.tier, 0) + 1
        return counts
