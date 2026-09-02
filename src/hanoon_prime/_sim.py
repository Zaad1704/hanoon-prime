"""hanoon_prime._sim — backtest simulation engine.

Runs the actual JULI pipeline bar-by-bar. Returns trades + equity curve.
Metrics calculation is in _metrics.py (split for file-length compliance).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .brain import Brain
from .constants import MAX_LOSS_PER_TRADE, MAX_POSITION_NOTIONAL
from .data import compute_buy_volume, estimate_bid_ask
from .edge import kelly_fraction


@dataclass
class SimPosition:
    """Tracks a single open position during backtest."""
    ticker: str
    entry_idx: int
    entry_price: float
    shares: float
    direction: int
    stop_price: float
    target_price: float
    peak_price: float
    score: float


@dataclass
class SimTrade:
    """A completed trade."""
    ticker: str
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    shares: float
    pnl_pct: float
    score: float
    direction: int
    exit_reason: str
    won: bool


def _enter_position(ticker: str, idx: int, brain: Brain, thought, entry_price: float) -> Optional[SimPosition]:
    """Create a position from a Thought, or return None if sizing fails."""
    size_shares = brain.size_position(thought.trace.get("win_prob", 0.25), entry_price)
    if size_shares <= 0:
        return None
    notional = min(entry_price * size_shares, MAX_POSITION_NOTIONAL)
    size_shares = notional / entry_price
    risk_amount = min(size_shares * entry_price * 0.03, MAX_LOSS_PER_TRADE)
    if risk_amount <= 0:
        return None
    d = thought.direction
    return SimPosition(
        ticker=ticker, entry_idx=idx, entry_price=entry_price,
        shares=size_shares, direction=d,
        stop_price=entry_price * (0.97 if d > 0 else 1.03),
        target_price=entry_price * (1.12 if d > 0 else 0.88),
        peak_price=entry_price, score=thought.score,
    )


def _check_exit(pos: SimPosition, low_i: float, high_i: float, idx: int) -> Optional[tuple[float, str]]:
    """Check if position should exit. Returns (exit_price, reason) or None."""
    d = pos.direction
    current = (high_i + low_i) / 2.0

    # Fixed stop + target (no trailing — simpler, proven R:R)
    if d > 0:
        if low_i <= pos.stop_price:
            return (pos.stop_price, "stop_loss")
        if high_i >= pos.target_price:
            return (pos.target_price, "target_hit")
    else:
        if high_i >= pos.stop_price:
            return (pos.stop_price, "stop_loss")
        if low_i <= pos.target_price:
            return (pos.target_price, "target_hit")

    if idx - pos.entry_idx >= 390:
        return (current, "timeout")

    return None


def _close_position(pos, exit_price, exit_idx, reason, brain, equity_curve):
    """Close position, record trade, update equity curve."""
    d = pos.direction
    if d > 0:
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
    else:
        pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

    from .constants import FIXED_FEE, FEE_RATE
    notional = pos.entry_price * pos.shares
    fees = 2 * (FIXED_FEE + FEE_RATE * notional)
    pnl_pct -= fees / notional if notional > 0 else 0.0
    won = pnl_pct > 0

    if brain is not None:
        brain.record_trade(ticker=pos.ticker, won=won, pnl_pct=pnl_pct, alpha_snapshot={})
    ec_pnl = pnl_pct * pos.shares * pos.entry_price / 1000.0
    equity_curve.append(equity_curve[-1] + ec_pnl)

    return SimTrade(
        ticker=pos.ticker, entry_idx=pos.entry_idx, exit_idx=exit_idx,
        entry_price=pos.entry_price, exit_price=exit_price, shares=pos.shares,
        pnl_pct=pnl_pct, score=pos.score, direction=d,
        exit_reason=reason, won=won,
    )


def simulate_ticker(ticker: str, close, high, low, volume, window: int = 30) -> tuple:
    """Run the JULI pipeline bar-by-bar. Returns (trades, equity_curve)."""
    buy_vol = compute_buy_volume(close, high, low, volume)
    bid_sizes, ask_sizes = estimate_bid_ask(volume, buy_vol)
    brain = Brain()

    trades: list[SimTrade] = []
    equity_curve: list[float] = [0.0]
    position: Optional[SimPosition] = None

    for i in range(window, len(close) - 1):
        try:
            brain.check_safety_nets()
        except RuntimeError:
            break  # backtest: safety pause = stop trading

        c_slice = close[i - window : i + 1]
        v_slice = volume[i - window : i + 1]
        bv_slice = buy_vol[i - window : i + 1]

        thought = brain.deliberate_entry(
            close=c_slice.tolist(), volume=v_slice.tolist(),
            buy_volume=bv_slice.tolist(), bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )

        entry_price = close[i + 1]
        if thought.verdict == "ENTER" and position is None:
            position = _enter_position(ticker, i, brain, thought, entry_price)

        if position is not None:
            exit_result = _check_exit(position, float(low[i + 1]), float(high[i + 1]), i + 1)
            if exit_result:
                price, reason = exit_result
                trades.append(_close_position(position, price, i + 1, reason, brain, equity_curve))
                position = None

    return trades, equity_curve
