"""hanoon_prime.hands — bar-by-bar execution simulation.

ATR-based stops (2.0×ATR), targets (6.0×ATR), timeout, dual LONG/SHORT.
Used for backtest validation only — live mode uses IB bracket orders.

Pipeline: cerebellum → cortex → entry/exit → journal → learning
"""
from __future__ import annotations

from typing import Any, Optional

from .cerebellum import compute_alpha
from .cortex import Cortex, Thought
from .edge import score_to_win_prob
from .eyes import compute_buy_volume, estimate_bid_ask, rolling_atr
from .hippocampus import Hippocampus
from .immune import (
    ATR_PERIOD,
    ATR_STOP_MULT,
    ATR_TARGET_MULT,
    EDGE_LOOKBACK,
    FEE_RATE,
    FIXED_FEE,
    TIMEOUT_BARS,
)
from .types import Position, Trade


def _make_position(d: int, entry: float, atr: float) -> tuple[float, float]:
    """Return (stop, target) for direction d."""
    if d > 0:
        return entry - atr * ATR_STOP_MULT, entry + atr * ATR_TARGET_MULT
    return entry + atr * ATR_STOP_MULT, entry - atr * ATR_TARGET_MULT


def _enter_position(
    ticker: str, idx: int, thought: Thought,
    entry_price: float, atr: float, win_prob: float, brain: Hippocampus,
) -> Optional[Position]:
    """Create a position from a Thought, or None if sizing fails."""
    shares = brain.size_position(win_prob, entry_price, atr)
    if shares <= 0:
        return None
    d = thought.direction
    stop, target = _make_position(d, entry_price, atr)
    return Position(
        ticker=ticker, entry_idx=idx, entry_price=entry_price,
        shares=shares, direction=d, stop_price=stop, target_price=target,
        peak_price=entry_price, score=thought.score, atr=atr,
    )


def _check_exit(
    pos: Position, low_i: float, high_i: float, idx: int,
    timeout_bars: int = TIMEOUT_BARS,
) -> Optional[tuple[float, str]]:
    """Check ATR stop/target + timeout. Returns (price, reason) or None."""
    d = pos.direction
    mid = (high_i + low_i) / 2.0
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
    if idx - pos.entry_idx >= timeout_bars:
        return (mid, "timeout")
    return None


def _compute_pnl(pos: Position, exit_price: float) -> float:
    """Compute P&L percentage after fees."""
    d = pos.direction
    gross = ((exit_price - pos.entry_price) / pos.entry_price if d > 0
             else (pos.entry_price - exit_price) / pos.entry_price)
    notional = pos.entry_price * pos.shares
    fees = 2 * (FIXED_FEE + FEE_RATE * notional)
    return gross - fees / notional if notional > 0 else 0.0


def _close_position(
    pos: Position, exit_price: float, exit_idx: int, reason: str,
    brain: Optional[Hippocampus], equity: list[float],
    z_scores: dict[str, float],
) -> Trade:
    """Close position: P&L, equity update, learning feedback."""
    d = pos.direction
    pnl_pct = _compute_pnl(pos, exit_price)
    won = pnl_pct > 0
    if brain is not None:
        brain.record_trade(
            ticker=pos.ticker, won=won, pnl_pct=pnl_pct,
            direction=d, z_scores=z_scores,
        )
    equity.append(equity[-1] + pnl_pct * pos.shares * pos.entry_price / 1000.0)
    return Trade(
        ticker=pos.ticker, entry_idx=pos.entry_idx, exit_idx=exit_idx,
        entry_price=pos.entry_price, exit_price=exit_price,
        shares=pos.shares, pnl_pct=pnl_pct, direction=d,
        exit_reason=reason, won=won, score=pos.score,
    )


def _evaluate_bar(
    i: int, w: int, close: Any, high: Any, low: Any,
    volume: Any, buy_vol: Any, cortex: Cortex,
) -> Thought:
    """Compute Cerebellum alpha and evaluate with Cortex."""
    v_w, bv_w = volume[i - w : i + 1], buy_vol[i - w : i + 1]
    bids, asks = estimate_bid_ask(v_w, bv_w)
    return cortex.evaluate(compute_alpha(
        close=close[i - w : i + 1], volume=v_w,
        buy_volume=bv_w, bid_sizes=bids, ask_sizes=asks,
    ))


def _try_enter(
    i: int, window: int, close: Any, high: Any, low: Any,
    volume: Any, buy_vol: Any, cortex: Cortex, ticker: str,
    brain: Hippocampus, position: Optional[Position],
) -> Optional[Position]:
    """Try to enter a new position if no position open."""
    if position is not None:
        return position
    thought = _evaluate_bar(i, window, close, high, low, volume, buy_vol, cortex)
    if thought.direction == 0 or brain.check_entry_allowed() is False:
        return position
    atr_val = rolling_atr(high[: i + 1], low[: i + 1], close[: i + 1], ATR_PERIOD)
    return _enter_position(
        ticker, i, thought, float(close[i + 1]),
        atr_val, score_to_win_prob(thought.score), brain,
    )


def _try_exit(
    i: int, position: Optional[Position], low: Any, high: Any,
    brain: Optional[Hippocampus], equity: list[float],
    last_z: dict[str, float],
) -> tuple[Optional[Position], Optional[Trade]]:
    """Try to exit an open position. Returns (position, trade)."""
    if position is None or position.entry_idx >= i:
        return position, None
    exit_r = _check_exit(position, float(low[i + 1]), float(high[i + 1]), i + 1)
    if not exit_r:
        return position, None
    trade = _close_position(
        position, exit_r[0], i + 1, exit_r[1], brain, equity, last_z,
    )
    return None, trade


def _process_bar(
    i: int, window: int, close: Any, high: Any, low: Any,
    volume: Any, buy_vol: Any, cortex: Cortex, ticker: str,
    brain: Hippocampus, position: Optional[Position],
    last_z: dict[str, float], equity_curve: list[float],
) -> tuple[Optional[Position], dict[str, float], Optional[Trade]]:
    """Process one bar: evaluate, enter, exit. Returns (pos, z, trade)."""
    thought = _evaluate_bar(i, window, close, high, low, volume, buy_vol, cortex)
    z_scores = thought.z_scores
    if position is None and thought.direction != 0 and not brain.check_entry_allowed():
        atr_val = rolling_atr(high[: i + 1], low[: i + 1], close[: i + 1], ATR_PERIOD)
        position = _enter_position(
            ticker, i, thought, float(close[i + 1]),
            atr_val, score_to_win_prob(thought.score), brain,
        )
    position, trade = _try_exit(i, position, low, high, brain, equity_curve, last_z)
    return position, z_scores, trade


def simulate_ticker(
    ticker: str, close: Any, high: Any, low: Any, volume: Any,
    window: int = EDGE_LOOKBACK, brain: Optional[Hippocampus] = None,
) -> tuple[list[Trade], list[float]]:
    """Run the JULI pipeline bar-by-bar. Returns (trades, equity_curve)."""
    buy_vol = compute_buy_volume(close, high, low, volume)
    brain = brain or Hippocampus()
    cortex = brain.cortex
    trades: list[Trade] = []
    equity_curve: list[float] = [0.0]
    position: Optional[Position] = None
    last_z: dict[str, float] = {}
    for i in range(window, len(close) - 1):
        position, last_z, trade = _process_bar(
            i, window, close, high, low, volume, buy_vol,
            cortex, ticker, brain, position, last_z, equity_curve,
        )
        if trade is not None:
            trades.append(trade)
    return trades, equity_curve
