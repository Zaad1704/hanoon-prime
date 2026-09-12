"""tests/test_phase7.py — Phase 7: sandboxed lean 3-factor benchmark stack.

Verifies the opt-in ``SimHooks`` plumbing is transparent by default
(shipped behavior unchanged) and that the lean stack pieces behave
independently of the shipped organs:
  * RVOL gate blocks low-volume bars, passes high-volume ones
  * session gate respects the ET opening window (09:30–11:00)
  * extra-alpha RS factor appears only when SPY data is supplied
  * ``run_walk_forward_lean`` mirrors WFA OOS scoring on committed fixtures
  * default (hooks=None) simulation equals the plain shipped path
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hanoon_prime.hands import EvalContext, SimHooks, simulate_ticker
from hanoon_prime.hippocampus import Hippocampus
from hanoon_prime.immune import EDGE_LOOKBACK
from hanoon_prime.phase7 import (
    LEAN_WEIGHTS,
    RS_WINDOW,
    LeanCfg,
    load_spy_closes,
    make_lean_brain,
    make_sim_hooks,
    run_walk_forward_lean,
    rvol_ok_at,
    session_ok,
)
from hanoon_prime.types import BarSeries

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def _bars(closes: list[float], volumes: list[float]) -> BarSeries:
    highs = [c * 1.001 for c in closes]
    lows = [c * 0.999 for c in closes]
    return BarSeries(
        np.asarray(closes),
        np.asarray(highs),
        np.asarray(lows),
        np.asarray(volumes),
    )


def test_rvol_ok_at_pass_and_block() -> None:
    """RVOL >= 2x a 20-bar mean passes; flat volume is blocked."""
    vols = [1000.0] * 20 + [10000.0]  # last bar 10x the 20-bar mean
    bars = _bars([100.0] * 21, vols)
    assert rvol_ok_at(bars, 20)
    assert not rvol_ok_at(bars, 10)  # too little history
    bars_flat = _bars([100.0] * 21, [500.0] * 21)
    assert not rvol_ok_at(bars_flat, 20)  # no volume spike


def test_session_ok_window() -> None:
    """Entries allowed only in [09:30, 11:00) ET wall-clock."""
    assert not session_ok("2026-08-27 09:29:00-0400")
    assert session_ok("2026-08-27 09:30:00-0400")
    assert session_ok("2026-08-27 10:59:00-0400")
    assert not session_ok("2026-08-27 11:00:00-0400")
    assert not session_ok("2026-08-27 14:00:00-0400")


def test_hooks_gate_blocks_outside_window_and_low_rvol() -> None:
    """SimHooks gate requires BOTH session window and RVOL spike."""
    import datetime as dt

    base = dt.datetime(2026, 8, 27, 9, 30)
    times = [(base + dt.timedelta(minutes=k)).isoformat() + "-0400" for k in range(60)]
    vols = [1000.0] * 40 + [15000.0] * 20  # spike from bar 40
    bars = _bars([100.0] * 60, vols)
    hooks = make_sim_hooks(ticker_times=times)
    ctx = EvalContext(brain=Hippocampus(), window=EDGE_LOOKBACK, ticker="T")

    # minute 0x = 09:3{0..} → within window; spike starts at bar 40 (10:10)
    assert hooks.allows(45, ctx, bars) is True  # 10:15, high RVOL
    assert hooks.allows(10, ctx, bars) is False  # 09:40 window ok but RVOL < 2
    # By bar 50 the spike has entered the 20-bar mean → RVOL fell back < 2.
    assert hooks.allows(50, ctx, bars) is False  # 10:20, spike now "normal"


def test_hooks_gate_blocks_after_1100() -> None:
    import datetime as dt

    base = dt.datetime(2026, 8, 27, 11, 0)
    times = [(base + dt.timedelta(minutes=k)).isoformat() + "-0400" for k in range(30)]
    vols = [1000.0] * 10 + [15000.0] * 20
    bars = _bars([100.0] * 30, vols)
    hooks = make_sim_hooks(ticker_times=times)
    ctx = EvalContext(brain=Hippocampus(), window=EDGE_LOOKBACK, ticker="T")
    assert hooks.allows(20, ctx, bars) is False  # 11:20 out of window


def test_extra_alpha_injects_rs_only_with_spy() -> None:
    """relative_strength_spy is absent without SPY; present with it."""
    import datetime as dt

    base = dt.datetime(2026, 8, 27, 9, 30)
    times = [(base + dt.timedelta(minutes=k)).isoformat() + "-0400" for k in range(20)]
    bars = _bars([100.0 + 0.1 * k for k in range(20)], [1000.0] * 20)
    ctx = EvalContext(brain=Hippocampus(), window=EDGE_LOOKBACK, ticker="T")

    hooks = make_sim_hooks(ticker_times=times)  # no SPY
    assert hooks.extra(19, ctx, bars) == {}
    # With SPY, bar 19 needs i >= RS_WINDOW and both keys present.
    spy_closes = {
        (base + dt.timedelta(minutes=k)).isoformat(sep=" ", timespec="minutes"): (
            200.0 + 0.2 * k
        )
        for k in range(20)
    }
    hooks2 = make_sim_hooks(ticker_times=times, spy_closes=spy_closes)
    extra = hooks2.extra(19, ctx, bars)
    assert "relative_strength_spy" in extra
    # ticker gained 0.1*4/100 = 0.4%; SPY gained 0.2*4/200 = 0.4% → RS ~ 0
    assert abs(extra["relative_strength_spy"]) < 1e-9


def test_hooks_none_is_transparent() -> None:
    """With hooks=None the shipped decision path is byte-identical."""
    from hanoon_prime.eyes import load_ohlcv

    data = load_ohlcv(FIXTURES / "AAPL_1min.csv")
    bars = BarSeries(
        np.asarray(data["close"]),
        np.asarray(data["high"]),
        np.asarray(data["low"]),
        np.asarray(data["volume"]),
    )
    plain, _ = simulate_ticker("AAPL", bars, EDGE_LOOKBACK)
    with_hooks, _ = simulate_ticker("AAPL", bars, EDGE_LOOKBACK, hooks=SimHooks())
    assert [(t.entry_idx, t.exit_idx, t.pnl_pct) for t in plain] == [
        (t.entry_idx, t.exit_idx, t.pnl_pct) for t in with_hooks
    ]


def test_lean_brain_weights() -> None:
    """Lean brain uses only the sandboxed factor weights."""
    brain = make_lean_brain()
    w = brain.indicator_weights
    assert set(w.keys()) == set(LEAN_WEIGHTS.keys())
    assert "vpin" not in w and "institutional_flow" not in w


def test_walk_forward_lean_on_fixtures() -> None:
    """Lean WFA runs against committed fixtures with SPY RS data."""
    from hanoon_prime.eyes import load_ohlcv

    spy_closes = load_spy_closes(str(FIXTURES / "SPY_1min.csv"))
    assert spy_closes is not None and len(spy_closes) > 1000
    aapl = load_ohlcv(FIXTURES / "AAPL_1min.csv")
    folds = run_walk_forward_lean("AAPL", aapl, LeanCfg(spy_closes=spy_closes, folds=4))
    assert len(folds) == 4
    assert all(0.0 <= len(f.pnl) == f.total_trades for f in folds)


def test_lean_toggles_change_trade_counts() -> None:
    """Gate-off trades more; RS-off scores identically without SPY."""
    from hanoon_prime.eyes import load_ohlcv

    spy_closes = load_spy_closes(str(FIXTURES / "SPY_1min.csv"))
    aapl = load_ohlcv(FIXTURES / "AAPL_1min.csv")
    gated = run_walk_forward_lean("AAPL", aapl, LeanCfg(spy_closes=spy_closes, folds=4))
    ungated = run_walk_forward_lean(
        "AAPL", aapl, LeanCfg(spy_closes=spy_closes, folds=4, use_gate=False)
    )
    gated_n = sum(f.total_trades for f in gated)
    ungated_n = sum(f.total_trades for f in ungated)
    assert ungated_n >= gated_n  # removing the gate can only add entries
    # RS off == no SPY data (extra_alpha silently returns {})
    no_rs = run_walk_forward_lean(
        "AAPL", aapl, LeanCfg(spy_closes=spy_closes, folds=4, use_rs=False)
    )
    no_spy = run_walk_forward_lean("AAPL", aapl, LeanCfg(folds=4))
    assert [(f.total_trades, f.ev_per_trade) for f in no_rs] == [
        (f.total_trades, f.ev_per_trade) for f in no_spy
    ]
