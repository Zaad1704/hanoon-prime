"""hanoon_prime.phase14 — intraday PDH/PDL sweep-and-reclaim (sandbox)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .types import F64Array
from .wfa import FoldResult, fold_windows

ENTRY_FROM = "10:00"  # first allowed entry minute ET
ENTRY_TO = "11:30"  # last allowed entry minute ET
SESSION_CLOSE = "15:50"  # force-flat time ET
SWEEP_MAX_BARS = 10  # reclaim must land within N bars of the sweep
STOP_CUSHION_MULT = 0.25  # stop cushion as fraction of ATR_prev
MIN_RR = 1.0  # min R:R to accept a signal
ATR_PERIOD = 14  # sessions used for ATR_prev
FEE_R = 0.02  # all-in per-trade cost in R units (applied by the bench)
FOLDS = 6


@dataclass
class SweepTrade:
    """One completed sweep-and-reclaim trade (pnl in R multiples)."""

    session: int
    side: int  # +1 long, -1 short (R13: direction as integer)
    entry: float
    stop: float
    target: float
    exit: float
    entry_ts: str
    rvol: float
    r: float


@dataclass
class _Sess:
    data: dict[str, Any]
    ts: list[str]
    pdl: float
    pdh: float
    mid: float
    atr_prev: float
    spy_ok: Callable[[str], bool] | None
    s: int


def _rvol_at(volume: F64Array, i: int) -> float:
    hist = volume[max(0, i - 20) : i]
    if hist.size < 5:
        return 0.0
    med = float(np.median(hist))
    return float(volume[i]) / med if med > 0 else 0.0


_Stats = tuple[F64Array, F64Array, F64Array, F64Array, list[int], F64Array]


def _session_stats(data: dict[str, Any], ts: list[str]) -> _Stats:
    o, h, l, c, starts, last = [], [], [], [], [], ""
    for i, t in enumerate(ts):
        if t[:10] != last:
            o.append(float(data["open"][i]))
            h.append(float(data["high"][i]))
            l.append(float(data["low"][i]))
            c.append(float(data["close"][i]))
            starts.append(i)
            last = t[:10]
        else:
            h[-1] = max(h[-1], float(data["high"][i]))
            l[-1] = min(l[-1], float(data["low"][i]))
            c[-1] = float(data["close"][i])
    h_a, l_a, c_a = np.asarray(h), np.asarray(l), np.asarray(c)
    n = len(h_a)
    tr = np.zeros(n)
    for s in range(1, n):
        tr[s] = max(h_a[s] - l_a[s], abs(h_a[s] - c_a[s - 1]), abs(l_a[s] - c_a[s - 1]))
    atr = np.full(n, np.nan)
    for s in range(ATR_PERIOD + 1, n):
        atr[s] = float(np.mean(tr[s - ATR_PERIOD : s]))
    return np.asarray(o), h_a, l_a, c_a, starts, atr


def _try_entry(
    side: int, ctx: _Sess, i: int, ext: float
) -> tuple[SweepTrade | None, float]:
    e = float(ctx.data["close"][i])
    long = side == 1
    ext = (min if long else max)(
        ext, float((ctx.data["low"] if long else ctx.data["high"])[i])
    )
    neg = -1.0 if long else 1.0
    stop = ext + neg * STOP_CUSHION_MULT * ctx.atr_prev
    num = (ctx.mid - e) if long else (e - ctx.mid)
    den = (e - stop) if long else (stop - e)
    if not ctx.spy_ok is None and not ctx.spy_ok(ctx.ts[i]):
        return None, ext
    if (e < ctx.pdl if long else e > ctx.pdh) or den <= 0 or num / den < MIN_RR:
        return None, ext
    tstamp = ctx.ts[i][:16]
    t = SweepTrade(
        ctx.s, side, e, stop, ctx.mid, 0.0, tstamp, _rvol_at(ctx.data["volume"], i), 0.0
    )
    return t, ext


def _scan_bar(
    ctx: _Sess, i: int, sweep: int, ext: float, sweep_bar: int
) -> tuple[int, float, int, SweepTrade | None]:
    if sweep == 0:
        if ctx.data["low"][i] < ctx.pdl:
            return 1, float(ctx.data["low"][i]), i, None
        if ctx.data["high"][i] > ctx.pdh:
            return -1, float(ctx.data["high"][i]), i, None
        return 0, ext, sweep_bar, None
    if i - sweep_bar > SWEEP_MAX_BARS:
        return 0, ext, sweep_bar, None
    side = 1 if sweep == 1 else -1
    trade, ext = _try_entry(side, ctx, i, ext)
    return sweep, ext, sweep_bar, trade


def _resolve_exit(trade: SweepTrade, c: float, slot: str) -> float | None:
    neg = -1.0 if trade.side == 1 else 1.0
    if neg * c >= neg * trade.stop:
        return c
    if neg * c <= neg * trade.target:
        return trade.target
    return c if slot >= SESSION_CLOSE else None


def _finalize(
    trades: list[SweepTrade], t: SweepTrade, ex: float | None
) -> SweepTrade | None:
    if ex is None:
        return t
    t.exit = ex
    # R multiple: signed exit distance over stop distance (side=+1 long/-1 short)
    t.r = t.side * (ex - t.entry) / abs(t.entry - t.stop)
    trades.append(t)
    return None


def sweep_trades(
    data: dict[str, Any], spy_ok: Callable[[str], bool] | None = None
) -> list[SweepTrade]:
    """Scan every session for sweep-and-reclaim signals; one trade/session max."""
    ts = list(data["datetime"])
    o_s, h_s, l_s, c_s, starts, atr = _session_stats(data, ts)
    ends = starts[1:] + [len(ts)]
    trades: list[SweepTrade] = []

    for s in range(1, len(starts)):
        if not np.isfinite(atr[s]):
            continue
        lo, hi = float(l_s[s - 1]), float(h_s[s - 1])
        ctx = _Sess(data, ts, lo, hi, (lo + hi) / 2.0, float(atr[s]), spy_ok, s)
        sweep, ext, sweep_bar = 0, 0.0, 0
        trade: SweepTrade | None = None
        for i in range(starts[s], ends[s]):
            slot = ts[i][11:16]
            if trade is not None:
                exit_ = _resolve_exit(trade, float(data["close"][i]), slot)
                trade = _finalize(trades, trade, exit_)
                continue
            if slot < ENTRY_FROM or slot >= ENTRY_TO:
                sweep = 0  # outside entry window: clear any stale sweep
            else:
                sweep, ext, sweep_bar, trade = _scan_bar(ctx, i, sweep, ext, sweep_bar)
    return trades


def run_sweep(
    data: dict[str, Any],
    folds: int = FOLDS,
    spy_ok: Callable[[str], bool] | None = None,
) -> list[FoldResult]:
    """SESSION-indexed walk-forward folds over completed sweep trades."""
    n_sess = len({t[:10] for t in data["datetime"]})
    if n_sess <= ATR_PERIOD + folds * 5:
        return []
    trades = sweep_trades(data, spy_ok)
    out: list[FoldResult] = []
    for k, (start, end) in enumerate(fold_windows(n_sess, folds)):
        pnl = [t.r for t in trades if start <= t.session < end]
        out.append(
            FoldResult(
                fold=k,
                start=start,
                end=end,
                trades=pnl,
                ev_per_trade=float(np.mean(pnl)) if pnl else 0.0,
                sharpe=0.0,
                pnl=pnl,
            )
        )
    return out
