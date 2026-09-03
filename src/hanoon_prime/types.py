"""hanoon_prime.types — shared dataclass definitions.

Position and Trade are used by both hands.py (execution) and
hippocampus.py (learning). Kept here to avoid circular imports and
keep each module under the 200-line limit (R3).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    """Tracks a single open position during simulation."""

    ticker: str
    entry_idx: int
    entry_price: float
    shares: float
    direction: int
    stop_price: float
    target_price: float
    peak_price: float
    score: float
    atr: float
    entry_time: float = field(default_factory=lambda: __import__("time").time())


@dataclass
class Trade:
    """A completed trade."""

    ticker: str
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    shares: float
    pnl_pct: float
    direction: int
    exit_reason: str
    won: bool
    score: float
