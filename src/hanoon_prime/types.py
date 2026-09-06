"""hanoon_prime.types — shared dataclass definitions.

Position and Trade are used by both hands.py (execution) and
hippocampus.py (learning). Kept here to avoid circular imports and
keep each module under the 200-line limit (R3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


@dataclass
class BarSeries:
    """Raw OHLCV (+ depth) bar data shared across the brain pipeline.

    Fields are intentionally ``Any`` so a single type serves array-backed
    backtests (numpy slices) and scalar live bars alike.
    """

    close: Any
    high: Any
    low: Any
    volume: Any
    buy_volume: Any = None
    bid_sizes: Any = None
    ask_sizes: Any = None


@dataclass
class FillInfo:
    """Execution fill bookkeeping routed to the consolidation engine."""

    qty: float = 1.0
    avg_price: float = 0.0
    fees: float = 0.0


@dataclass
class ExitLevels:
    """Static stop/target levels attached to an open position."""

    stop: float
    target: float


@dataclass
class BracketOrder:
    """Static OCA stop+target pair for position protection."""

    action: str
    qty: int
    stop: float
    target: float
    oca: str


@dataclass
class TradeStats:
    """Aggregated per-ticker trade statistics fed to metrics assembly."""

    n: int
    wins: int
    losses: int
    wr: float
    ev: float
    ret: float
    aw: float
    al: float
    rr: float
    dd: float
    sr: float
