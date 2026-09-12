"""hanoon_prime.phase7 — sandboxed lean 3-factor benchmark stack.

Phase-2/4 walk-forward FAILED the shipped 5-factor cocktail. Phase 7 is a
**sandbox ablation**: drop the three contested/horizon-mismatched factors
(vpin, institutional_flow, orderbook_imbalance), keep vwap_deviation +
momentum, add a relative-strength-vs-SPY factor, and gate entries to a
high-RVOL opening window (09:30–11:00 ET).

Nothing here mutates shipped organs. Entries are filtered through the
opt-in ``SimHooks`` plumbing on ``hands.simulate_ticker`` (default None =
shipped behavior unchanged); the benchmark harness compares this stack
against the old baseline side-by-side before any production change.

The gate is pure and time-aware:
  * ``session_ok`` — entry time in [09:30, 11:00) ET
  * ``rvol_ok``   — latest volume >= RVOL_MIN x rolling mean volume
The extra-alpha hook adds ``relative_strength_spy`` = the ticker's 15-bar
return minus SPY's return over the same window (long-side carry).

Tunables live HERE (sandbox), not in immune.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .cortex import Cortex
from .eyes import load_ohlcv
from .hands import EvalContext, SimHooks, simulate_ticker
from .hippocampus import Hippocampus
from .immune import EDGE_LOOKBACK
from .types import BarSeries
from .wfa import FoldResult, _fold_sharpe, fold_windows

# ── Sandbox tunables (NOT shipped risk config) ────────────────────────────
LEAN_WEIGHTS: dict[str, float] = {
    "vwap_deviation": 0.40,
    "momentum": 0.35,
    "relative_strength_spy": 0.25,
}
RVOL_MIN: float = 2.0  # entry needs volume >= 2x its 20-bar mean
RVOL_PERIOD: int = 20
SESSION_START: str = "09:30"
SESSION_END: str = "11:00"
RS_WINDOW: int = 15  # bars used for the SPY-relative 15-min return
SPY_TICKER: str = "SPY"


@dataclass
class LeanCfg:
    """Lean-stack run configuration (sandbox-only, never shipped)."""

    spy_closes: dict[str, float] | None = None
    weights: dict[str, float] = field(default_factory=lambda: dict(LEAN_WEIGHTS))
    use_gate: bool = True
    use_rs: bool = True
    folds: int = 6


def _bar_dt(ts: str) -> datetime:
    """Parse a fixture timestamp (tz-aware) into a naive datetime."""
    return datetime.fromisoformat(ts).replace(tzinfo=None)


def _hhmm_minutes(hhmm: str) -> int:
    """Minutes since midnight for an HH:MM wall-clock string."""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _minutes_of_day(dt: datetime) -> int:
    """Minutes since midnight for a wall-clock session comparison."""
    return dt.hour * 60 + dt.minute


def session_ok(ts: str) -> bool:
    """True when ``ts`` falls inside the modeled opening window (ET)."""
    start = _hhmm_minutes(SESSION_START)
    end = _hhmm_minutes(SESSION_END)
    now = _minutes_of_day(_bar_dt(ts))
    return start <= now < end


def rvol_ok_at(
    bars: BarSeries,
    i: int,
    period: int = RVOL_PERIOD,
    min_rvol: float = RVOL_MIN,
) -> bool:
    """True when volume at bar ``i`` is >= ``min_rvol`` x rolling mean."""
    if i < period:
        return False
    ref = float(np.mean(bars.volume[i - period : i]))
    if ref <= 1e-12:
        return False
    return float(bars.volume[i]) / ref >= min_rvol


def _session_mask(ticker_times: list[str], start: str, end: str) -> np.ndarray:
    """Boolean mask: True for timestamps inside the [start, end) window."""
    start_m = _hhmm_minutes(start)
    end_m = _hhmm_minutes(end)
    n = len(ticker_times)
    is_ok = np.zeros(n, dtype=bool)
    for j, ts in enumerate(ticker_times):
        m = _minutes_of_day(_bar_dt(ts))
        is_ok[j] = start_m <= m < end_m
    return is_ok


def _rs_key(ts: str) -> str:
    """Naive minute key for SPY lookups (``YYYY-MM-DD HH:MM``)."""
    return _bar_dt(ts).isoformat(sep=" ", timespec="minutes")


def _rs_factor(
    i: int,
    spy_closes: dict[str, float] | None,
    ticker_times: list[str],
    bars: BarSeries,
) -> dict[str, float]:
    """Ticker 15-bar return minus SPY's over the same window, as alpha."""
    if spy_closes is None or i < RS_WINDOW:
        return {}
    key = _rs_key(ticker_times[i])
    prev_key = _rs_key(ticker_times[i - RS_WINDOW])
    spy_now = spy_closes.get(key)
    spy_prev = spy_closes.get(prev_key)
    if not spy_now or not spy_prev or spy_prev <= 0:
        return {}
    ticker_ret = float(bars.close[i] / bars.close[i - RS_WINDOW] - 1.0)
    spy_ret = float(spy_now / spy_prev - 1.0)
    return {"relative_strength_spy": float(ticker_ret - spy_ret)}


def _regime_allows(is_ok: np.ndarray, use_gate: bool, i: int, bars: BarSeries) -> bool:
    """True when the regime gate (session x RVOL) allows bar ``i``."""
    if not use_gate:
        return True
    if not (i < len(is_ok) and is_ok[i]):
        return False
    return rvol_ok_at(bars, i)


def make_sim_hooks(
    ticker_times: list[str],
    spy_closes: dict[str, float] | None = None,
    window: tuple[str, str] = (SESSION_START, SESSION_END),
    use_gate: bool = True,
    use_rs: bool = True,
) -> SimHooks:
    """Build the Phase-7 opt-in hooks for one ticker's bar series.

    Args:
        ticker_times: per-bar timestamps aligned with the BarSeries.
        spy_closes: {naive_dt_key: close} lookup for RS vs SPY. When None
            (SPY unavailable), relative_strength_spy is omitted entirely.
        window: (start, end) HH:MM opening entries are allowed.
        use_gate: False disables the regime gate (RVOL x session).
        use_rs: False disables the SPY-relative factor.

    Returns:
        ``SimHooks`` with a gate (RVOL x session) and RS extra-alpha.
    """
    start, end = window
    is_ok = _session_mask(ticker_times, start, end)

    def gate(i: int, ctx: EvalContext, ts: str, bars: BarSeries) -> bool:
        """Entry gate: inside session window AND above the RVOL floor."""
        return _regime_allows(is_ok, use_gate, i, bars)

    def extra_alpha(i: int, ctx: EvalContext, bars: BarSeries) -> dict[str, float]:
        """Inject relative_strength_spy for bar ``i`` when SPY data exists."""
        if not use_rs:
            return {}
        return _rs_factor(i, spy_closes, ticker_times, bars)

    return SimHooks(
        times=ticker_times,
        gate=gate,
        extra_alpha=extra_alpha,
    )


def make_lean_brain(weights: dict[str, float] | None = None) -> Hippocampus:
    """Hippocampus over a lean-weight Cortex (static, learning off)."""
    cortex = Cortex(weights=dict(weights or LEAN_WEIGHTS))
    return Hippocampus(cortex=cortex)


def load_spy_closes(path: str) -> dict[str, float] | None:
    """Load ``{naive minute key: close}`` from an SPY 1-min CSV."""
    data = load_ohlcv(path)
    closes = {}
    for ts, close in zip(data["datetime"], data["close"]):
        key = _bar_dt(ts).isoformat(sep=" ", timespec="minutes")
        closes[key] = float(close)
    return closes or None


def _fold_bars(data: dict[str, Any], warm: int, end: int) -> BarSeries:
    """Slice a BarSeries for one fold (warmup index ``warm`` to ``end``)."""
    s = slice(warm, end)
    return BarSeries(
        np.asarray(data["close"])[s],
        np.asarray(data["high"])[s],
        np.asarray(data["low"])[s],
        np.asarray(data["volume"])[s],
    )


def _scored_fold(
    ticker: str,
    data: dict[str, Any],
    cfg: LeanCfg,
    window: tuple[int, int, int],
    brain: Hippocampus,
) -> FoldResult:
    """One OOS fold through the lean stack; trades must close before ``end``."""
    j, start, end = window
    warm = max(0, start - EDGE_LOOKBACK)
    hooked_times = list(data["datetime"][warm:end])
    hooks = make_sim_hooks(
        ticker_times=hooked_times,
        spy_closes=cfg.spy_closes,
        use_gate=cfg.use_gate,
        use_rs=cfg.use_rs,
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


def run_walk_forward_lean(
    ticker: str,
    data: dict[str, Any],
    cfg: LeanCfg,
) -> list[FoldResult]:
    """Run WFA on a ticker through the lean stack (RVOL+session gate, RS).

    Mirrors ``wfa.run_walk_forward`` but threads the Phase-7 hooks + brain
    through ``simulate_ticker`` so the SAME OOS scoring applies. ``cfg``
    lets the harness ablate gate and RS factor in isolation.
    """
    n = len(data["close"])
    brain = make_lean_brain(cfg.weights)
    return [
        _scored_fold(ticker, data, cfg, (j, start, end), brain)
        for j, (start, end) in enumerate(fold_windows(n, cfg.folds))
    ]


__all__ = [
    "LeanCfg",
    "LEAN_WEIGHTS",
    "RVOL_MIN",
    "RVOL_PERIOD",
    "SESSION_START",
    "SESSION_END",
    "RS_WINDOW",
    "SPY_TICKER",
    "session_ok",
    "rvol_ok_at",
    "make_sim_hooks",
    "make_lean_brain",
    "load_spy_closes",
    "run_walk_forward_lean",
    "_bar_dt",
    "_session_mask",
    "_rs_key",
    "_rs_factor",
    "_scored_fold",
]
