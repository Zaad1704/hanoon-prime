"""hanoon_prime.phase8 — sandboxed staged-exit benchmark stack.

Phase-7's lean stack still loses on 1-min bars (Alpaca 180d: EV ~ -0.116R
OOS with RVOL/session gate + SPY-relative factor). One lever the shipped
protocol never tests is the EXIT side: ATR stop/target exits are
all-or-nothing single fills that force winners to give gains back.

Phase 8 is a **sandbox ablation of the exit policy**: realize a fraction
of the position at a nearby Target-1 (1.5x ATR), move the stop to
breakeven, then let the runner trail (3x ATR behind the favorable extreme)
or a trailing VWAP computed since entry.

Nothing here mutates shipped organs. Exits ride the opt-in ``SimHooks``
``exit_plan`` plumbing on ``hands.simulate_ticker`` (default None = shipped
behavior unchanged); the benchmark compares each arm against the lean
static control on the same panel before any production change.

Tunables live HERE (sandbox), not in immune.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from .hands import EXIT_PLAN, EvalContext, ExitPlan, SimHooks, simulate_ticker
from .immune import EDGE_LOOKBACK, TIMEOUT_BARS
from .phase7 import LeanCfg, _fold_bars, make_lean_brain, make_sim_hooks
from .types import BarSeries, Position
from .wfa import FoldResult, _fold_sharpe, fold_windows

# ── Sandbox tunables (NOT shipped risk config) ────────────────────────────
STAGE_SCALE: float = 0.5  # fraction of the position realized at Target-1
TARGET1_MULT: float = 1.5  # x ATR distance to Target-1
RUNNER_TRAIL_ATR: float = 3.0  # x ATR trail behind the runner's extreme


@dataclass(frozen=True)
class StageCfg:
    """Staged-exit configuration (sandbox-only, never shipped)."""

    scale: float = STAGE_SCALE
    t1_mult: float = TARGET1_MULT
    breakeven: bool = True
    runner_trail_atr: float = RUNNER_TRAIL_ATR
    runner_mode: Literal["atr", "vwap"] = "atr"


DEFAULT_STAGE: StageCfg = StageCfg()


@dataclass(frozen=True)
class StageRun:
    """Lean-stack walk-forward config plus a staged-exit policy."""

    lean: LeanCfg = field(default_factory=LeanCfg)
    stage: StageCfg = DEFAULT_STAGE


DEFAULT_RUN: StageRun = StageRun()


@dataclass
class _Stage:
    """Per-position runner state held by the exit-plan closure."""

    peak: float
    vwap_num: float = 0.0
    vwap_den: float = 0.0
    scaled: bool = False


@dataclass
class _ExitState:
    """Policy plus the mutable per-position runner state."""

    cfg: StageCfg
    stage: dict[int, _Stage] = field(default_factory=dict)


def _touch(st: _Stage, d: int, bars: BarSeries, k: int) -> None:
    """Update the runner extreme and the running VWAP inputs for one bar."""
    ext = float(bars.high[k]) if d > 0 else float(bars.low[k])
    st.peak = max(st.peak, ext) if d > 0 else min(st.peak, ext)
    typical = (float(bars.high[k]) + float(bars.low[k]) + float(bars.close[k])) / 3.0
    st.vwap_num += typical * float(bars.volume[k])
    st.vwap_den += float(bars.volume[k])


def _trail_ref(cfg: StageCfg, st: _Stage, d: int, pos: Position) -> float:
    """Break-even-floored trailing reference the runner exits below/above."""
    if cfg.runner_mode == "vwap":
        ref = st.vwap_num / st.vwap_den if st.vwap_den > 0 else pos.entry_price
    else:
        ref = st.peak - d * cfg.runner_trail_atr * pos.atr
    if not cfg.breakeven:
        return ref
    if d > 0:
        return max(ref, pos.entry_price)
    return min(ref, pos.entry_price)


def _runner_plan(
    st: _ExitState, pos: Position, bars: BarSeries, k: int, run: _Stage
) -> Optional[ExitPlan]:
    """Trailing-runner exit once Target-1 has been scaled out."""
    d = pos.direction
    low = float(bars.low[k])
    high = float(bars.high[k])
    _touch(run, d, bars, k)
    ref = _trail_ref(st.cfg, run, d, pos)
    if (d > 0 and low <= ref) or (d < 0 and high >= ref):
        st.stage.pop(pos.entry_idx, None)
        return ExitPlan(fill_price=ref, reason="runner_trail", close_remaining=True)
    if (d > 0 and high >= pos.target_price) or (d < 0 and low <= pos.target_price):
        st.stage.pop(pos.entry_idx, None)
        return ExitPlan(
            fill_price=pos.target_price, reason="target_hit", close_remaining=True
        )
    return None


def _decide_exit(
    st: _ExitState, i: int, _ctx: EvalContext, bars: BarSeries, pos: Position
) -> Optional[ExitPlan]:
    """Decide one bar's staged exit: stop/timeout, Target-1, runner."""
    k = i + 1
    if k >= len(bars.close):
        return None
    d = pos.direction
    low = float(bars.low[k])
    high = float(bars.high[k])
    if (d > 0 and low <= pos.stop_price) or (d < 0 and high >= pos.stop_price):
        return ExitPlan(
            fill_price=pos.stop_price, reason="stop_loss", close_remaining=True
        )
    if (k - pos.entry_idx) >= TIMEOUT_BARS:
        mid = (high + low) / 2.0
        return ExitPlan(fill_price=mid, reason="timeout", close_remaining=True)
    run = st.stage.setdefault(pos.entry_idx, _Stage(peak=pos.entry_price))
    t1 = pos.entry_price + d * st.cfg.t1_mult * pos.atr
    if not run.scaled:
        if (d > 0 and high >= t1) or (d < 0 and low <= t1):
            run.scaled = True
            run.peak = t1
            return ExitPlan(scale_out=st.cfg.scale, fill_price=t1, reason="target1")
        return None
    return _runner_plan(st, pos, bars, k, run)


def make_staged_exit(cfg: StageCfg = DEFAULT_STAGE) -> EXIT_PLAN:
    """Build a Phase-8 opt-in exit hook for one ticker's bar series.

    The hook decides: stop-loss and timeout first (mirroring shipped
    ``_check_exit``), then a Target-1 scale-out at ``cfg.scale``, then a
    trailing runner exit. All decisions are per-bar and purely a function
    of ``bars``/``pos`` plus closure state keyed by entry index.
    """
    st = _ExitState(cfg=cfg)

    def exit_plan(
        i: int, _ctx: EvalContext, bars: BarSeries, pos: Position
    ) -> Optional[ExitPlan]:
        """Route one bar's exit decision to the staged-exit policy."""
        return _decide_exit(st, i, _ctx, bars, pos)

    return exit_plan


def make_staged_hooks(
    ticker_times: list[str],
    cfg: StageCfg = DEFAULT_STAGE,
    spy_closes: dict[str, float] | None = None,
    use_gate: bool = True,
    use_rs: bool = True,
) -> SimHooks:
    """Lean gate/RS hooks plus a staged-exit ``exit_plan`` hook."""
    hooks = make_sim_hooks(
        ticker_times=ticker_times,
        spy_closes=spy_closes,
        use_gate=use_gate,
        use_rs=use_rs,
    )
    hooks.exit_plan = make_staged_exit(cfg)
    return hooks


def _scored_fold_staged(
    ticker: str,
    data: dict[str, Any],
    run: StageRun,
    window: tuple[int, int, int],
    brain: Any,
) -> FoldResult:
    """One OOS fold through the lean stack + staged exits."""
    j, start, end = window
    warm = max(0, start - EDGE_LOOKBACK)
    hooked_times = list(data["datetime"][warm:end])
    hooks = make_staged_hooks(
        ticker_times=hooked_times,
        cfg=run.stage,
        spy_closes=run.lean.spy_closes,
        use_gate=run.lean.use_gate,
        use_rs=run.lean.use_rs,
    )
    trades, _ = simulate_ticker(
        ticker, _fold_bars(data, warm, end), EDGE_LOOKBACK, brain=brain, hooks=hooks
    )
    scored = [
        t for t in trades if warm + t.exit_idx < end and warm + t.entry_idx >= start
    ]
    pnl = [t.pnl_pct for t in scored]
    return FoldResult(
        fold=j,
        start=start,
        end=end,
        trades=scored,
        ev_per_trade=float(np.mean(pnl)) if pnl else 0.0,
        sharpe=_fold_sharpe(pnl),
        pnl=pnl,
    )


def run_walk_forward_staged(
    ticker: str, data: dict[str, Any], run: StageRun = DEFAULT_RUN
) -> list[FoldResult]:
    """Run WFA on a ticker through the lean stack + staged-exit policy."""
    n = len(data["close"])
    brain = make_lean_brain(run.lean.weights)
    return [
        _scored_fold_staged(ticker, data, run, (j, start, end), brain)
        for j, (start, end) in enumerate(fold_windows(n, run.lean.folds))
    ]


__all__ = [
    "StageCfg",
    "StageRun",
    "DEFAULT_STAGE",
    "STAGE_SCALE",
    "TARGET1_MULT",
    "RUNNER_TRAIL_ATR",
    "make_staged_exit",
    "make_staged_hooks",
    "run_walk_forward_staged",
    "_scored_fold_staged",
]
