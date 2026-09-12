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

from collections.abc import Callable
from dataclasses import dataclass, field
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
    SLIPPAGE_BPS,
    TIMEOUT_BARS,
)
from .types import BarSeries, Position, Trade

GATE = Callable[[int, "EvalContext", Any, BarSeries], bool]
ALPHA_EXTRA = Callable[[int, "EvalContext", BarSeries], dict[str, float]]
EXIT_PLAN = Callable[[int, "EvalContext", BarSeries, "Position"], Optional["ExitPlan"]]


@dataclass(slots=True)
class ExitPlan:
    """Phase-8 opt-in: a staged-exit action for the current bar.

    ``scale_out`` — fraction of the ORIGINAL position realized now at
    ``fill_price`` while the runner stays open. ``close_remaining`` closes
    whatever is left after earlier scale-outs; the composite trade then
    equals the realized legs plus the remaining leg at ``fill_price``.
    """

    scale_out: float = 0.5
    fill_price: float = 0.0
    reason: str = "scale_out"
    close_remaining: bool = False


@dataclass(slots=True)
class SimHooks:
    """Phase-7/8 opt-in plumbing for ``simulate_ticker``.

    Shipped decision path is UNCHANGED when all hooks are None:
      * ``times``: per-bar timestamps aligned with ``bars`` (enables the
        gate and session-aware extras).
      * ``gate``: consulted before entry so regime filters (time window,
        RVOL) can block entries.
      * ``extra_alpha``: merged into the cerebellum alpha so additional
        factors (e.g. RS vs SPY) reach the Cortex scorer.
      * ``exit_plan``: consulted before the ATR stop/target/timeout check
        so a staged/partial-exit policy can override the shipped exit path
        (Phase-8 sandbox only; None = shipped path unchanged).
    """

    times: Optional[list[Any]] = None
    gate: Optional[GATE] = None
    extra_alpha: Optional[ALPHA_EXTRA] = None
    exit_plan: Optional[EXIT_PLAN] = None

    def allows(self, i: int, ctx: "EvalContext", bars: BarSeries) -> bool:
        """Gate check: True when no gate hook or the gate passes bar ``i``."""
        if self.gate is None or self.times is None:
            return True
        return bool(self.gate(i, ctx, self.times[i], bars))

    def extra(self, i: int, ctx: "EvalContext", bars: BarSeries) -> dict[str, float]:
        """Extra alpha factors for bar ``i`` (empty when no hook)."""
        if self.extra_alpha is None:
            return {}
        return self.extra_alpha(i, ctx, bars) or {}

    def plan_exit(
        self, i: int, ctx: "EvalContext", bars: BarSeries, pos: "Position"
    ) -> Optional[ExitPlan]:
        """Exit action for bar ``i`` (None when no exit hook is set)."""
        if self.exit_plan is None:
            return None
        return self.exit_plan(i, ctx, bars, pos)


@dataclass(slots=True)
class EvalContext:
    """Per-ticker execution context carried through the bar loop."""

    brain: Hippocampus
    window: int
    ticker: str
    hooks: Optional[SimHooks] = None

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
    """Mutable accumulator shared by enter/exit/close steps.

    ``staged`` records, per open position, the composite pnl_pct already
    banked by earlier scale-outs plus the cumulative fraction realized —
    the ledger Phase-8 exit hooks need for the final composite trade.
    """

    equity: list[float]
    last_z: dict[str, float]
    staged: dict[int, tuple[float, float]] = field(default_factory=dict)


def _adverse_fill(price: float, direction: int) -> float:
    """Adverse-slippage fill: buyers pay up, sellers take less."""
    return price * (1.0 + SLIPPAGE_BPS / 10000.0 * direction)


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
    fill = _adverse_fill(entry.price, d)
    stop, target = _make_position(d, fill, entry.atr)
    return Position(
        ticker=ticker,
        entry_idx=idx,
        entry_price=fill,
        shares=shares,
        direction=d,
        stop_price=stop,
        target_price=target,
        peak_price=fill,
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


def _compute_pnl(pos: Position, exit_price: float, frac: float = 1.0) -> float:
    """Compute P&L percentage after fees for a fraction ``frac`` of position.

    The gross move is scale-invariant; the fixed-fee leg is amortized over
    the leg's own notional so a partial scale-out pays its own fill costs
    (contact with the shipped all-or-nothing fee math at ``frac=1.0``).
    """
    d = pos.direction
    gross = (
        (exit_price - pos.entry_price) / pos.entry_price
        if d > 0
        else (pos.entry_price - exit_price) / pos.entry_price
    )
    notional = pos.entry_price * pos.shares
    if notional <= 0:
        return 0.0
    return frac * gross - 2 * (FIXED_FEE + FEE_RATE * frac * notional) / notional


def _close_position(
    pos: Position,
    exit: ExitContext,
    brain: Optional[Hippocampus],
    state: SimState,
    pnl_pct: Optional[float] = None,
) -> Trade:
    """Close position: P&L, equity update, learning feedback.

    ``pnl_pct`` optionally carries a Phase-8 composite (realized legs +
    remaining leg) instead of a single-fill P&L.
    """
    d = pos.direction
    if pnl_pct is None:
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
    alpha = compute_alpha(
        close=bars.close[i - w : i + 1],
        volume=v_w,
        buy_volume=bv_w,
        bid_sizes=bids,
        ask_sizes=asks,
    )
    if ctx.hooks is not None:
        alpha.update(ctx.hooks.extra(i, ctx, bars))
    return ctx.cortex.evaluate(alpha)


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


def _apply_exit_plan(
    i: int,
    pos: Position,
    ctx: EvalContext,
    state: SimState,
    plan: ExitPlan,
) -> tuple[Optional[Position], Optional[Trade]]:
    """Execute a Phase-8 staged exit plan. Returns (position, trade)."""
    d = pos.direction
    fill = _adverse_fill(plan.fill_price, -d)
    real, closed = state.staged.get(pos.entry_idx, (0.0, 0.0))
    if not plan.close_remaining:
        leg = min(plan.scale_out, 1.0 - closed)
        if leg <= 0.0:
            return pos, None
        state.staged[pos.entry_idx] = (
            real + _compute_pnl(pos, fill, frac=leg),
            closed + leg,
        )
        return pos, None
    remaining = max(0.0, 1.0 - closed)
    composite = real + _compute_pnl(pos, fill, frac=remaining)
    trade = _close_position(
        pos,
        ExitContext(price=fill, idx=i + 1, reason=plan.reason),
        ctx.brain,
        state,
        pnl_pct=composite,
    )
    state.staged.pop(pos.entry_idx, None)
    return None, trade


def _try_exit(
    i: int,
    ctx: EvalContext,
    bars: BarSeries,
    position: Optional[Position],
    state: SimState,
) -> tuple[Optional[Position], Optional[Trade]]:
    """Try to exit an open position. Returns (position, trade)."""
    if position is None or position.entry_idx >= i:
        return position, None
    if ctx.hooks is not None:
        plan = ctx.hooks.plan_exit(i, ctx, bars, position)
        if plan is not None:
            return _apply_exit_plan(i, position, ctx, state, plan)
    exit_r = _check_exit(
        position, float(bars.low[i + 1]), float(bars.high[i + 1]), i + 1
    )
    if not exit_r:
        return position, None
    fill = _adverse_fill(exit_r[0], -position.direction)
    trade = _close_position(
        position,
        ExitContext(price=fill, idx=i + 1, reason=exit_r[1]),
        ctx.brain,
        state,
    )
    return None, trade


def _clock_allows(ctx: EvalContext, i: int, bars: BarSeries) -> bool:
    """Phase-7 opt-in: gate passes only when hooks.gate allows bar ``i``."""
    if ctx.hooks is None:
        return True
    return ctx.hooks.allows(i, ctx, bars)


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
    if (
        position is None
        and thought.direction != 0
        and ctx.brain.check_entry_allowed()
        and _clock_allows(ctx, i, bars)
    ):
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
    position, trade = _try_exit(i, ctx, bars, position, state)
    state.last_z = z_scores
    return position, state.last_z, trade


def simulate_ticker(
    ticker: str,
    bars: BarSeries,
    window: int = EDGE_LOOKBACK,
    brain: Optional[Hippocampus] = None,
    hooks: Optional[SimHooks] = None,
) -> tuple[list[Trade], list[float]]:
    """Run the JULI pipeline bar-by-bar. Returns (trades, equity_curve).

    ``hooks`` (Phase-7 opt-in) can add a per-bar regime gate and extra alpha
    factors without changing the shipped decision path: see ``SimHooks``.
    Default None = shipped behavior unchanged.
    """
    buy_vol = compute_buy_volume(bars.close, bars.high, bars.low, bars.volume)
    bars = BarSeries(
        bars.close,
        bars.high,
        bars.low,
        bars.volume,
        buy_volume=buy_vol,
    )
    brain = brain or Hippocampus()
    ctx = EvalContext(brain=brain, window=window, ticker=ticker, hooks=hooks)
    state = SimState(equity=[0.0], last_z={})
    trades: list[Trade] = []
    position: Optional[Position] = None
    for i in range(window, len(bars.close) - 1):
        position, state.last_z, trade = _process_bar(i, ctx, bars, position, state)
        if trade is not None:
            trades.append(trade)
    return trades, state.equity
