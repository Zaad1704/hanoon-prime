"""hanoon_prime.data.budget — IB data subscription budget manager.

Allocates finite IB data slots (TBT, DOM, L1) across open positions
and scanner candidates. Open positions always get priority.

IB Limits:
- TBT (Tick-by-Tick): ~10 tickers max
- DOM (Level 2): 5-60 tickers
- L1 (reqMktData): 100 tickers default
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

MAX_TBT: int = 5
MAX_DOM: int = 10
MAX_L1: int = 50


@dataclass
class SubSlot:
    """One data subscription slot."""

    ticker: str
    tier: str  # "TBT", "DOM", "L1"
    is_position: bool
    allocated_at: float = field(default_factory=time.time)


class DataBudget:
    """Manages IB data subscription allocations."""

    def __init__(self) -> None:
        self.slots: dict[str, SubSlot] = {}
        self._last_alloc: float = 0.0

    def allocate(
        self,
        positions: set[str],
        candidates: list[str],
    ) -> tuple[dict[str, str], set[str]]:
        """Compute target subscriptions. Returns (to_sub, to_unsub)."""
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

        for sym in candidates:
            if sym in target:
                continue
            if tbt_count < MAX_TBT:
                target[sym] = "TBT"
                tbt_count += 1
            elif l1_count < MAX_L1:
                target[sym] = "L1"
                l1_count += 1

        to_sub, to_unsub = self._diff(target)
        self._apply(target)
        return to_sub, to_unsub

    def _diff(self, target: dict[str, str]) -> tuple[dict[str, str], set[str]]:
        """Compare current slots with target. Returns changes."""
        to_sub: dict[str, str] = {}
        to_unsub: set[str] = set()

        for sym, slot in list(self.slots.items()):
            if sym not in target:
                to_unsub.add(sym)
            elif slot.tier != target[sym]:
                to_unsub.add(sym)
                to_sub[sym] = target[sym]
            else:
                to_sub[sym] = slot.tier

        for sym, tier in target.items():
            if sym not in self.slots:
                to_sub[sym] = tier

        return to_sub, to_unsub

    def _apply(self, target: dict[str, str]) -> None:
        """Update internal slot state."""
        new_slots: dict[str, SubSlot] = {}
        for sym, tier in target.items():
            is_pos = sym in {s.ticker for s in self.slots.values() if s.is_position}
            new_slots[sym] = SubSlot(
                ticker=sym,
                tier=tier,
                is_position=is_pos,
            )
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

    def count_tiers(self) -> dict[str, int]:
        """Count subscriptions per tier."""
        counts: dict[str, int] = {}
        for slot in self.slots.values():
            counts[slot.tier] = counts.get(slot.tier, 0) + 1
        return counts
