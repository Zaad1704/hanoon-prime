"""hanoon_prime.phase9 — scheduled-earnings catalyst screen (sandbox, Path A).

Phase-9 tests whether an earnings catalyst + heavy PRE-MARKET volume can
turn the lean stack's OOS R-expectancy positive. Entries are restricted to
sessions that are BOTH a scheduled-earnings trading day AND pre-market
RVOL>5 (08:00-09:25 ET, versus the trailing 20 *prior* sessions). Every
funnel input is knowable before 09:30 — no RTH information leaks into the
entry decision.

The stack inside a chosen session is UNCHANGED (Phase-7 lean: vwap +
momentum + RS behind the 09:30-11:00 regime gate). Only the SESSION universe
changes. Earnings dates are scheduled weeks ahead (knowable without
lookahead); a rare reschedule reveals the final date (a documented minor
survivorship residual in this sandbox).

Nothing ships: the bench compares this to ``run_walk_forward_lean`` over the
same panel and reports pooled OOS R-expectancy + deflated edge/verdict.

Session mapping (see ``earnings_session_dates``): an earnings timestamp with
hour < 16:00 ET is a Before-Market-Open report → the SAME trading day; hour
>= 16:00 is After-Market-Close → the NEXT trading session in the panel.

Tunables live HERE (sandbox), not in immune.py.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .eyes import load_ohlcv
from .hands import EvalContext, SimHooks, simulate_ticker
from .immune import EDGE_LOOKBACK
from .phase7 import (
    SESSION_END,
    SESSION_START,
    LeanCfg,
    _rs_factor,
    _session_mask,
    make_lean_brain,
)
from .types import BarSeries
from .wfa import FoldResult, _fold_sharpe, fold_windows

# ── Sandbox tunables (NOT shipped risk config) ────────────────────────────
PREMKT_START: str = "08:00"
PREMKT_END: str = "09:25"
PREMKT_RVOL_MIN: float = 5.0
PREMKT_PERIOD: int = 20
EARN_HOUR_AMC: int = 16  # earnings at/after hour -> next trading session
CATALYST_DEFAULT_LABEL: str = "catalyst"


def _dt(ts: str) -> datetime:
    """Parse a fixture timestamp (tz-aware) into a naive datetime."""
    return datetime.fromisoformat(ts).replace(tzinfo=None)


def _date_key(ts: str) -> str:
    """``YYYY-MM-DD`` wall-clock key for a fixture timestamp."""
    return _dt(ts).isoformat(sep=" ", timespec="seconds")[:10]


def _sorted_date_keys(times: list[str]) -> list[str]:
    """Unique dates in ``times``, in first-seen (chronological) order."""
    return list(dict.fromkeys(_date_key(t) for t in times))


def premkt_flagged_dates(
    premkt_csv: str | Path,
    min_rvol: float = PREMKT_RVOL_MIN,
    period: int = PREMKT_PERIOD,
) -> set[str]:
    """Session dates whose pre-market volume >= ``min_rvol`` x trailing mean.

    The trailing mean uses only PRIOR sessions in strict chronological order
    (period window), so the flag is leak-free at any session. Sessions before
    the first ``period`` observations are never flagged (warmup).
    """
    data = load_ohlcv(str(premkt_csv))
    by_date: dict[str, float] = {}
    for ts, vol in zip(data["datetime"], data["volume"]):
        by_date[_date_key(ts)] = by_date.get(_date_key(ts), 0.0) + float(vol)
    dates = sorted(by_date)
    vols = [by_date[d] for d in dates]
    flags: set[str] = set()
    for k in range(period, len(vols)):
        mean = float(np.mean(vols[k - period : k]))
        if mean > 1e-12 and vols[k] >= min_rvol * mean:
            flags.add(dates[k])
    return flags


def earnings_session_dates(
    earnings_et: list[str],
    rth_dates: list[str],
) -> set[str]:
    """Map earnings timestamps (ET) onto trading sessions inside the panel.

    ``rth_dates`` are the distinct session dates of the RTH panel (needed to
    resolve After-Market-Close earnings to their next trading day). Returns
    only sessions present in ``rth_dates``.
    """
    rth_set = set(rth_dates)
    ordered = sorted(rth_dates)
    out: set[str] = set()
    for iso in earnings_et:
        t = _dt(iso) if "T" in iso else datetime.fromisoformat(iso)
        base = t.date().isoformat()
        candidate = (
            base
            if t.hour < EARN_HOUR_AMC
            else next((d for d in ordered if d > base), None)
        )
        if candidate and candidate in rth_set:
            out.add(candidate)
    return out


def load_earnings(ticker: str, earnings_dir: str | Path) -> list[str]:
    """Earnings timestamps (ET) for one ticker from a research JSON file."""
    path = Path(earnings_dir) / f"{ticker}.json"
    if not path.exists():
        return []
    import json

    payload = json.loads(path.read_text())
    rows = payload.get("earnings_et")
    return list(rows) if rows else []


def _catalyst_mask(times: list[str], allowed_dates: set[str]) -> np.ndarray:
    """Per-bar mask: True when the bar's session date is allowed."""
    n = len(times)
    mask = np.zeros(n, dtype=bool)
    for j, ts in enumerate(times):
        mask[j] = _date_key(ts) in allowed_dates
    return mask


def _catalyst_gate(
    i: int,
    is_ok: np.ndarray,
    c_ok: np.ndarray,
    use_gate: bool,
    bars: BarSeries,
) -> bool:
    """Entry gate: inside session window, above RVOL, on a catalyst day."""
    if not use_gate:
        return bool(i < len(c_ok) and c_ok[i])
    if not (i < len(is_ok) and is_ok[i]):
        return False
    return bool(i < len(c_ok) and c_ok[i]) and _rvol_ok(bars, i)


def make_catalyst_hooks(
    ticker_times: list[str],
    spy_closes: dict[str, float] | None = None,
    window: tuple[str, str] = (SESSION_START, SESSION_END),
    use_gate: bool = True,
    use_rs: bool = True,
    catalyst_ok: Optional[np.ndarray] = None,
) -> SimHooks:
    """Phase-7 lean hooks whose gate ALSO requires a catalyst session bar.

    Args:
        ticker_times: per-bar timestamps aligned with the BarSeries.
        spy_closes: {naive_dt_key: close} lookup for RS vs SPY.
        window: (start, end) HH:MM opening entries are allowed.
        use_gate: False disables the regime gate (RVOL x session).
        use_rs: False disables the SPY-relative factor.
        catalyst_ok: per-bar boolean aligned with ``ticker_times``; when None,
            every bar passes (pure Phase-7 behavior).

    Returns:
        ``SimHooks`` with a composed gate (regime AND catalyst) + RS alpha.
    """
    is_ok = _session_mask(ticker_times, window[0], window[1])
    c_ok = catalyst_ok if catalyst_ok is not None else np.ones(len(ticker_times), bool)

    def gate(i: int, ctx: EvalContext, ts: str, bars: BarSeries) -> bool:
        """Entry gate: regime (session x RVOL) AND catalyst-session mask."""
        return _catalyst_gate(i, is_ok, c_ok, use_gate, bars)

    def extra_alpha(i: int, ctx: EvalContext, bars: BarSeries) -> dict[str, float]:
        """Inject relative_strength_spy for bar ``i`` when SPY data exists."""
        if not use_rs:
            return {}
        return _rs_factor(i, spy_closes, ticker_times, bars)

    return SimHooks(times=ticker_times, gate=gate, extra_alpha=extra_alpha)


def _rvol_ok(bars: BarSeries, i: int) -> bool:
    """True when volume at bar ``i`` >= 2.0 x its 20-bar rolling mean."""
    from .phase7 import RVOL_MIN, RVOL_PERIOD

    if i < RVOL_PERIOD:
        return False
    ref = float(np.mean(bars.volume[i - RVOL_PERIOD : i]))
    if ref <= 1e-12:
        return False
    return float(bars.volume[i]) / ref >= RVOL_MIN


def _fold_bars(data: dict[str, Any], warm: int, end: int) -> BarSeries:
    """Slice a BarSeries for one fold (warmup index ``warm`` to ``end``)."""
    s = slice(warm, end)
    return BarSeries(
        np.asarray(data["close"])[s],
        np.asarray(data["high"])[s],
        np.asarray(data["low"])[s],
        np.asarray(data["volume"])[s],
    )


def _scored_fold_catalyst(
    ticker: str,
    data: dict[str, Any],
    cfg: LeanCfg,
    window: tuple[int, int, int],
    brain: Any,
    full_mask: np.ndarray,
) -> FoldResult:
    """One OOS fold through the catalyst-gated lean stack."""
    j, start, end = window
    warm = max(0, start - EDGE_LOOKBACK)
    hooked_times = list(data["datetime"][warm:end])
    hooks = make_catalyst_hooks(
        ticker_times=hooked_times,
        spy_closes=cfg.spy_closes,
        use_gate=cfg.use_gate,
        use_rs=cfg.use_rs,
        catalyst_ok=full_mask[warm:end],
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


def run_walk_forward_catalyst(
    ticker: str,
    data: dict[str, Any],
    cfg: LeanCfg,
    allowed_dates: set[str],
) -> list[FoldResult]:
    """WFA a ticker where entries are restricted to ``allowed_dates`` sessions.

    Mirrors ``phase7.run_walk_forward_lean`` scoring exactly; the only change
    is the catalyst gate restricting which sessions may produce entries.
    ``allowed_dates`` empty => no entries => empty folds (INSUFFICIENT).
    """
    n = len(data["close"])
    full_mask = _catalyst_mask(list(data["datetime"]), allowed_dates)
    brain = make_lean_brain(cfg.weights)
    return [
        _scored_fold_catalyst(ticker, data, cfg, (j, start, end), brain, full_mask)
        for j, (start, end) in enumerate(fold_windows(n, cfg.folds))
    ]


__all__ = [
    "PREMKT_START",
    "PREMKT_END",
    "PREMKT_RVOL_MIN",
    "PREMKT_PERIOD",
    "EARN_HOUR_AMC",
    "premkt_flagged_dates",
    "earnings_session_dates",
    "load_earnings",
    "make_catalyst_hooks",
    "run_walk_forward_catalyst",
    "_dt",
    "_date_key",
    "_catalyst_mask",
    "_scored_fold_catalyst",
]
