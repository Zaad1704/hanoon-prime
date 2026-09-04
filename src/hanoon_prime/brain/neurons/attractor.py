"""hanoon_prime.brain.neurons.attractor — Attractor memory for pattern storage.

Stores winning/losing trading patterns as attractor basins in the
spike space. During sleep replay, these patterns are replayed to
stabilize synaptic configurations.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Attractor:
    """A stored attractor pattern from a completed trade.

    Attributes:
        ticker: Symbol of the traded asset.
        center: Pattern vector (normalized indicator values at entry).
        won: Whether the trade was profitable.
        pnl_pct: P&L percentage.
        trade_count: Number of times this pattern was observed.
        wins: Number of winning occurrences.
        losses: Number of losing occurrences.
        created_at: Timestamp of creation.
    """

    ticker: str = ""
    center: List[float] = field(default_factory=list)
    won: bool = False
    pnl_pct: float = 0.0
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    created_at: float = field(default_factory=time.time)

    def update(self, won: bool, pnl: float) -> None:
        """Update statistics after trade outcome."""
        self.trade_count += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.pnl_pct = pnl
        self.won = won


class AttractorMemory:
    """Associative memory storing trading patterns as attractors.

    Provides pattern recall and sleep replay functionality.
    """

    def __init__(self, max_patterns: int = 500) -> None:
        self._attractors: List[Attractor] = []
        self._max_patterns = max_patterns
        self._ticker_index: dict[str, List[int]] = {}

    def store(
        self,
        ticker: str,
        pattern: List[float],
        won: bool,
        pnl_pct: float,
    ) -> Attractor:
        """Store a new trade pattern as an attractor."""
        # Check for existing pattern
        if ticker in self._ticker_index:
            for idx in self._ticker_index[ticker]:
                existing = self._attractors[idx]
                if self._patterns_match(existing.center, pattern):
                    existing.update(won, pnl_pct)
                    return existing

        # Create new attractor
        attractor = Attractor(
            ticker=ticker,
            center=list(pattern),
            won=won,
            pnl_pct=pnl_pct,
        )
        self._attractors.append(attractor)

        # Index by ticker
        if ticker not in self._ticker_index:
            self._ticker_index[ticker] = []
        self._ticker_index[ticker].append(len(self._attractors) - 1)

        # Prune if over capacity
        if len(self._attractors) > self._max_patterns:
            self._prune()

        return attractor

    def _patterns_match(self, stored: List[float], new: List[float]) -> bool:
        """Check if two patterns are similar."""
        if len(stored) != len(new):
            return False
        if not stored:
            return False
        diff = sum(abs(s - n) for s, n in zip(stored, new)) / len(stored)
        return diff < 0.1

    def get_patterns(self, ticker: Optional[str] = None) -> List[Attractor]:
        """Get all patterns, optionally filtered by ticker."""
        if ticker is None:
            return list(self._attractors)
        indices = self._ticker_index.get(ticker, [])
        return [self._attractors[i] for i in indices]

    def clear(self) -> None:
        """Clear all stored patterns."""
        self._attractors.clear()
        self._ticker_index.clear()

    def __len__(self) -> int:
        return len(self._attractors)

    def __iter__(self):
        return iter(self._attractors)


__all__ = ["AttractorMemory", "Attractor"]
