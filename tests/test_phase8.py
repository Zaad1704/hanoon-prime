"""tests/test_phase8.py — Phase 8: sandboxed staged-exit benchmark stack.

Verifies the opt-in ``SimHooks.exit_plan`` plumbing and the staged-exit
policy while keeping the shipped decision path transparent by default:
  * stop-loss and timeout still close the whole remaining position
  * Target-1 (1.5x ATR) scales out ``cfg.scale`` of the position
  * the runner keeps the rest and exits on an ATR trail or trailing VWAP
  * the breakeven floor keeps the runner from giving back Target-1 gains
  * the composite trade bookkeeping splits P&L across realized legs
  * ``run_walk_forward_staged`` mirrors WFA OOS scoring on fixtures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hanoon_prime.hands import (
    EvalContext,
    ExitPlan,
    SimHooks,
    SimState,
    _apply_exit_plan,
    _try_exit,
)
from hanoon_prime.hippocampus import Hippocampus
from hanoon_prime.immune import EDGE_LOOKBACK
from hanoon_prime.phase7 import LeanCfg
from hanoon_prime.phase8 import (
    StageCfg,
    StageRun,
    make_staged_exit,
    run_walk_forward_staged,
)
from hanoon_prime.types import BarSeries, Position

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"
_LONG = 1


def _bars(segments: list[list[float]]) -> BarSeries:
    """Build a BarSeries from [close, high, low, volume] row segments."""
    return BarSeries(
        np.asarray([s[0] for s in segments]),
        np.asarray([s[1] for s in segments]),
        np.asarray([s[2] for s in segments]),
        np.asarray([s[3] for s in segments]),
    )


def _pos() -> Position:
    """A long position with 1.0 ATR (Target-1 sits at +1.5 above entry)."""
    return Position(
        ticker="T",
        entry_idx=1,
        entry_price=100.0,
        shares=100.0,
        direction=_LONG,
        stop_price=98.0,
        target_price=106.0,
        peak_price=100.0,
        score=1.0,
        atr=1.0,
    )


def _ctx(hook) -> EvalContext:
    """EvalContext with the given exit-plan hook attached."""
    return EvalContext(
        brain=Hippocampus(),
        window=EDGE_LOOKBACK,
        ticker="T",
        hooks=SimHooks(exit_plan=hook),
    )


def _filler(n: int) -> list[list[float]]:
    """``n`` benign bars far from every trigger level."""
    return [[100.0, 100.3, 99.7, 1000.0] for _ in range(n)]


def _hold(bars: BarSeries, hook, i_last: int) -> tuple[Position, object]:
    """Drive the exit loop to ``i_last`` and return (position, trade)."""
    ctx = _ctx(hook)
    state = SimState(equity=[0.0], last_z={})
    p: Position = _pos()
    trade = None
    for i in range(1, i_last + 1):
        p, trade = _try_exit(i, ctx, bars, p, state)
    return p, trade


def test_stop_loss_closes_remaining() -> None:
    """A pierced stop closes the entire remaining position immediately."""
    bars = _bars(_filler(4))
    # bar[3] low <= stop 98
    bars_hi = _bars(_filler(3) + [[97.0, 97.5, 96.0, 1000.0]])
    p, trade = _hold(bars_hi, make_staged_exit(), 2)
    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert p is None


def test_timeout_closes_remaining() -> None:
    """A timeout closes the entire remaining position at bar midpoint."""
    # entry_idx=1 and i=999 -> bar 1000 is beyond the 999-bar timeout.
    bars = _bars(_filler(1001))
    p, trade = _hold(bars, make_staged_exit(), 999)
    assert trade is not None
    assert trade.exit_reason == "timeout"
    assert p is None


def test_target1_scales_out_not_full_close() -> None:
    """Target-1 triggers a scale-out (not a full close) at ``cfg.scale``."""
    # bar[3] high >= 101.5 (= entry + 1.5*ATR)
    bars = _bars(_filler(3) + [[103.0, 103.5, 102.5, 1000.0]])
    state = SimState(equity=[0.0], last_z={})
    ctx = _ctx(make_staged_exit(StageCfg(scale=0.25)))
    p = _pos()
    p, trade = _try_exit(2, ctx, bars, p, state)
    assert trade is None  # position stays open
    assert p is not None
    assert abs(state.staged[p.entry_idx][1] - 0.25) < 1e-9  # 25% realized


def test_runner_atr_trail_exits_after_scale() -> None:
    """After Target-1 the runner trails 3 ATR behind its extreme (floored)."""
    # bar[3] triggers Target-1; bar[4] low falls below the trail floor.
    bars = _bars(
        _filler(3)
        + [
            [103.0, 103.5, 102.5, 1000.0],  # Target-1
            [100.5, 100.8, 98.9, 1000.0],  # low 98.9 <= trail floor 100.0
        ]
    )
    p, trade = _hold(bars, make_staged_exit(), 3)
    assert trade is not None
    assert trade.exit_reason == "runner_trail"
    assert abs(trade.exit_price - 100.0) < 0.1  # breakeven floor fill


def test_breakeven_floor_keeps_runner_above_entry() -> None:
    """The breakeven floor stops the runner at entry, never below it."""
    bars = _bars(
        _filler(3)
        + [
            [103.0, 103.5, 102.5, 1000.0],  # Target-1
            [99.5, 99.8, 98.9, 1000.0],  # fades below entry; floor = 100.0
        ]
    )
    p, trade = _hold(bars, make_staged_exit(), 3)
    assert trade is not None
    assert trade.exit_reason == "runner_trail"
    # Breakeven floor = entry 100.0; adverse fill dips it by SLIPPAGE_BPS (5bps).
    assert abs(trade.exit_price - (100.0 * (1.0 - 0.0005))) < 1e-9


def test_vwap_trail_exits_without_breakeven_floor() -> None:
    """Trailing VWAP exits the runner even when no ATR trail would fire."""
    bars = _bars(
        _filler(3)
        + [
            [103.0, 103.5, 102.5, 1000.0],  # Target-1
            [99.5, 99.8, 98.2, 1000.0],  # vwap ~99.2 < entry; low breaks it
        ]
    )
    p, trade = _hold(bars, make_staged_exit(StageCfg(runner_mode="vwap")), 3)
    assert trade is not None
    assert trade.exit_reason == "runner_trail"
    assert trade.exit_price < 100.0  # vwap exit, not breakeven


def test_composite_pnl_combines_realized_legs() -> None:
    """Scale-out P&L plus runner P&L pool into a single composite trade."""
    ctx = _ctx(None)
    state = SimState(equity=[0.0], last_z={})
    p = _pos()

    p, trade = _apply_exit_plan(
        1, p, ctx, state, ExitPlan(scale_out=0.5, fill_price=102.0, reason="target1")
    )
    assert trade is None  # position stays open after the scale-out
    assert abs(state.staged[p.entry_idx][1] - 0.5) < 1e-9

    p, trade = _apply_exit_plan(
        20,
        p,
        ctx,
        state,
        ExitPlan(fill_price=108.0, reason="runner", close_remaining=True),
    )
    assert trade is not None
    assert trade.exit_reason == "runner"
    # 0.5 leg up ~2% + 0.5 leg up ~8% ~ 5%, minus fees/slippage
    assert abs(trade.pnl_pct - 0.05) < 0.005
    assert _pos().entry_idx not in state.staged  # ledger cleaned up on close


def test_default_hooks_transparent_with_exit_none() -> None:
    """SimHooks with exit_plan None keeps the shipped path byte-identical."""
    from hanoon_prime.eyes import load_ohlcv
    from hanoon_prime.hands import simulate_ticker

    data = load_ohlcv(FIXTURES / "AAPL_1min.csv")
    bars = BarSeries(
        np.asarray(data["close"]),
        np.asarray(data["high"]),
        np.asarray(data["low"]),
        np.asarray(data["volume"]),
    )
    plain, _ = simulate_ticker("AAPL", bars, EDGE_LOOKBACK)
    hooked, _ = simulate_ticker("AAPL", bars, EDGE_LOOKBACK, hooks=SimHooks())
    assert [(t.entry_idx, t.exit_idx, t.pnl_pct) for t in plain] == [
        (t.entry_idx, t.exit_idx, t.pnl_pct) for t in hooked
    ]


def test_walk_forward_staged_on_fixtures() -> None:
    """Staged WFA runs against committed fixtures with SPY RS data."""
    from hanoon_prime.eyes import load_ohlcv
    from hanoon_prime.phase7 import load_spy_closes

    spy_closes = load_spy_closes(str(FIXTURES / "SPY_1min.csv"))
    assert spy_closes is not None and len(spy_closes) > 1000
    aapl = load_ohlcv(FIXTURES / "AAPL_1min.csv")
    run = StageRun(lean=LeanCfg(spy_closes=spy_closes, folds=4))
    folds = run_walk_forward_staged("AAPL", aapl, run)
    assert len(folds) == 4
    assert all(0.0 <= len(f.pnl) == f.total_trades for f in folds)
