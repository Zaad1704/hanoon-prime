"""hanoon_prime.hands — bar-by-bar execution simulation.

ATR-based stops (2.0×ATR), targets (6.0×ATR), timeout, dual LONG/SHORT.
Used for backtest validation only — live mode uses IB bracket orders.

Pipeline: cerebellum → cortex → entry/exit → journal → learning

To stay under ruff's PLR0913 (max 5 args) and keep the bar-by-bar
pipeline readable, market bar data travels as a ``BarSeries`` and the
per-ticker execution context as ``EvalContext`` / ``SimState`` value
objects — all arithmetic is unchanged, only the parameter packaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
from .types import BarSeries, Position, Trade


@dataclass(slots=True)
class EvalContext:
    """Per-ticker execution context carried through the bar loop."""

    brain: Hippocampus
    window: int
    ticker: str

    @property
    def cortex(self) -> Cortex:
        """Expose the underlying Cortex for direct scoring/inspection."""
        return self.brain.cortex


@dataclass(slots=True)
class EntryContext:
    """Everything needed to size + place an entry."""

    price: float
    atr: float
    win_prob: float


@dataclass(slots=True)
class ExitContext:
    """Everything needed to close a position."""

    price: float
    idx: int
    reason: str


@dataclass(slots=True)
class SimState:
    """Mutable accumulator shared by enter/exit/close steps."""

    equity: list[float]
    last_z: dict[str, float]


def _make_position(d: int, entry: float, atr: float) -> tuple[float, float]:
    """Return (stop, target) for direction d."""
    if d > 0:
        return entry - atr * ATR_STOP_MULT, entry + atr * ATR_TARGET_MULT
    return entry + atr * ATR_STOP_MULT, entry - atr * ATR_TARGET_MULT


def _enter_position(
    ticker: str,
    idx: int,
    thought: Thought,
    entry: EntryContext,
    brain: Hippocampus,
) -> Optional[Position]:
    """Create a position from a Thought, or None if sizing fails."""
    shares = brain.size_position(entry.win_prob, entry.price, entry.atr)
    if shares <= 0:
        return None
    d = thought.direction
    stop, target = _make_position(d, entry.price, entry.atr)
    return Position(
        ticker=ticker,
        entry_idx=idx,
        entry_price=entry.price,
        shares=shares,
        direction=d,
        stop_price=stop,
        target_price=target,
        peak_price=entry.price,
        score=thought.score,
        atr=entry.atr,
    )


def _check_exit(
    pos: Position,
    low_i: float,
    high_i: float,
    idx: int,
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
    gross = (
        (exit_price - pos.entry_price) / pos.entry_price
        if d > 0
        else (pos.entry_price - exit_price) / pos.entry_price
    )
    notional = pos.entry_price * pos.shares
    fees = 2 * (FIXED_FEE + FEE_RATE * notional)
    return gross - fees / notional if notional > 0 else 0.0


def _close_position(
    pos: Position,
    exit: ExitContext,
    brain: Optional[Hippocampus],
    state: SimState,
) -> Trade:
    """Close position: P&L, equity update, learning feedback."""
    d = pos.direction
    pnl_pct = _compute_pnl(pos, exit.price)
    won = pnl_pct > 0
    if brain is not None:
        brain.record_trade(
            ticker=pos.ticker,
            won=won,
            pnl_pct=pnl_pct,
            direction=d,
            z_scores=state.last_z,
        )
    state.equity.append(
        state.equity[-1] + pnl_pct * pos.shares * pos.entry_price / 1000.0
    )
    return Trade(
        ticker=pos.ticker,
        entry_idx=pos.entry_idx,
        exit_idx=exit.idx,
        entry_price=pos.entry_price,
        exit_price=exit.price,
        shares=pos.shares,
        pnl_pct=pnl_pct,
        direction=d,
        exit_reason=exit.reason,
        won=won,
        score=pos.score,
    )


def _evaluate_bar(i: int, ctx: EvalContext, bars: BarSeries) -> Thought:
    """Compute Cerebellum alpha and evaluate with Cortex."""
    w = ctx.window
    v_w, bv_w = (
        bars.volume[i - w : i + 1],
        bars.buy_volume[i - w : i + 1],
    )
    bids, asks = estimate_bid_ask(v_w, bv_w)
    return ctx.cortex.evaluate(
        compute_alpha(
            close=bars.close[i - w : i + 1],
            volume=v_w,
            buy_volume=bv_w,
            bid_sizes=bids,
            ask_sizes=asks,
        )
    )


def _try_enter(
    i: int,
    ctx: EvalContext,
    bars: BarSeries,
    position: Optional[Position],
) -> Optional[Position]:
    """Try to enter a new position if no position open."""
    if position is not None:
        return position
    thought = _evaluate_bar(i, ctx, bars)
    if thought.direction == 0 or ctx.brain.check_entry_allowed() is False:
        return position
    atr_val = rolling_atr(
        bars.high[: i + 1], bars.low[: i + 1], bars.close[: i + 1], ATR_PERIOD
    )
    return _enter_position(
        ctx.ticker,
        i,
        thought,
        EntryContext(
            price=float(bars.close[i + 1]),
            atr=atr_val,
            win_prob=score_to_win_prob(thought.score),
        ),
        ctx.brain,
    )


def _try_exit(
    i: int,
    position: Optional[Position],
    bars: BarSeries,
    brain: Optional[Hippocampus],
    state: SimState,
) -> tuple[Optional[Position], Optional[Trade]]:
    """Try to exit an open position. Returns (position, trade)."""
    if position is None or position.entry_idx >= i:
        return position, None
    exit_r = _check_exit(
        position, float(bars.low[i + 1]), float(bars.high[i + 1]), i + 1
    )
    if not exit_r:
        return position, None
    trade = _close_position(
        position,
        ExitContext(price=exit_r[0], idx=i + 1, reason=exit_r[1]),
        brain,
        state,
    )
    return None, trade


def _process_bar(
    i: int,
    ctx: EvalContext,
    bars: BarSeries,
    position: Optional[Position],
    state: SimState,
) -> tuple[Optional[Position], dict[str, float], Optional[Trade]]:
    """Process one bar: evaluate, enter, exit. Returns (pos, z, trade)."""
    thought = _evaluate_bar(i, ctx, bars)
    z_scores = thought.z_scores
    if position is None and thought.direction != 0 and ctx.brain.check_entry_allowed():
        atr_val = rolling_atr(
            bars.high[: i + 1], bars.low[: i + 1], bars.close[: i + 1], ATR_PERIOD
        )
        position = _enter_position(
            ctx.ticker,
            i,
            thought,
            EntryContext(
                price=float(bars.close[i + 1]),
                atr=atr_val,
                win_prob=score_to_win_prob(thought.score),
            ),
            ctx.brain,
        )
    position, trade = _try_exit(i, position, bars, ctx.brain, state)
    state.last_z = z_scores
    return position, state.last_z, trade


def simulate_ticker(
    ticker: str,
    bars: BarSeries,
    window: int = EDGE_LOOKBACK,
    brain: Optional[Hippocampus] = None,
) -> tuple[list[Trade], list[float]]:
    """Run the JULI pipeline bar-by-bar. Returns (trades, equity_curve)."""
    buy_vol = compute_buy_volume(bars.close, bars.high, bars.low, bars.volume)
    bars = BarSeries(
        bars.close,
        bars.high,
        bars.low,
        bars.volume,
        buy_volume=buy_vol,
    )
    brain = brain or Hippocampus()
    ctx = EvalContext(brain=brain, window=window, ticker=ticker)
    state = SimState(equity=[0.0], last_z={})
    trades: list[Trade] = []
    position: Optional[Position] = None
    for i in range(window, len(bars.close) - 1):
        position, state.last_z, trade = _process_bar(i, ctx, bars, position, state)
        if trade is not None:
            trades.append(trade)
    return trades, state.equity
