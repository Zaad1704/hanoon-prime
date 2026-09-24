#!/usr/bin/env python3
"""scripts/orb_backtest.py - QQQ Trend-ORB (opening-range breakout) backtest.

Replicates the user's Trend-ORB spec against IBKR-fetched 1-min RTH bars
(fetch_ibkr.py / fetch_alpaca.py CSV format, bar stamps = bar-START ET times):

  * Opening range = first 15 RTH bars (09:30..09:44 ET). OR_high =
    max(high), OR_low = min(low).
  * vref = 1.5 * mean(volume of those 15 bars).
  * Signal on any COMPLETED bar 09:45..15:30 (stamps 585..930) with
    close > OR_high and volume >= vref  -> long
    (and the symmetric short for close < OR_low).
  * Entry: market at the NEXT bar's open.
  * Exit:  market at the close of the 15:55 bar (~15:56 ET). Flat before close.
  * One trade per day.

Costs: --cost-bps is applied ONE-WAY per fill (entry + exit), so a round trip
pays 2x. The published edge is execution-sensitive: breakeven slippage in
independent replications was ~2.2c/share; the user's own numbers pay <=3bp and
die at 6bp. Sweep mode runs the documented variant battery (vref multiplier,
signal window cutoff, ATR trailing stop, OR stops, partial take-profit,
day-type filters, relative-volume filter).

Deterministic: pure function of the input CSV (no hash-order dependence).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

OPEN_MIN = 570  # 09:30 ET label
LAST_MIN = 959  # 15:59 ET label
OR_N = 15  # bars in the opening range
SIG_LO = OPEN_MIN + OR_N  # 09:45 (first POST-range completed bar)
SIG_HI = 930  # 15:30 (latest signal bar)
EXIT_MIN = 955  # 15:55 close (~15:56 ET) -> exit fill


def load_bars(
    path: Path,
) -> dict[str, dict[int, tuple[float, float, float, float, int]]]:
    """Keyed by session date -> minute-of-day -> (open, high, low, close, vol)."""
    sessions: dict[str, dict[int, tuple[float, float, float, float, int]]] = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        for _ in range(3):  # skip Price / Ticker / Datetime rows
            next(r)
        for row in r:
            if not row or len(row) < 6:
                continue
            try:
                dt = datetime.fromisoformat(row[0])
            except ValueError:
                continue
            minute = dt.hour * 60 + dt.minute
            if not (OPEN_MIN <= minute <= LAST_MIN):
                continue
            key = f"{dt:%Y-%m-%d}"
            bar = (
                float(row[4]),  # open
                float(row[2]),  # high
                float(row[3]),  # low
                float(row[1]),  # close
                int(row[5]),  # volume
            )
            sessions.setdefault(key, {})[minute] = bar
    return sessions


def day_range(bars: dict[int, tuple[float, float, float, float, int]]) -> float:
    hi = max(b[1] for b in bars.values())
    lo = min(b[2] for b in bars.values())
    return hi - lo


def opening_range(
    bars: dict[int, tuple[float, float, float, float, int]], n: int
) -> tuple[float, float, float] | None:
    """Return (OR_high, OR_low, mean_vol) if all n OR bars are present."""
    need = list(range(OPEN_MIN, OPEN_MIN + n))
    if any(m not in bars for m in need):
        return None
    hs = [bars[m][1] for m in need]
    ls = [bars[m][2] for m in need]
    vs = [bars[m][4] for m in need]
    return max(hs), min(ls), float(np.mean(vs))


@dataclass(frozen=True)
class Levels:
    or_high: float
    or_low: float
    vref: float
    vwap: bool
    long_only: bool = False
    short_only: bool = False


def find_signal(bars: dict, levels: Levels, sig_end: int) -> tuple[int, str] | None:
    """First completed bar in [SIG_LO, sig_end] breaching the range on vref."""
    cap = min(sig_end, EXIT_MIN - 1)
    vwap_series = _vwap_series(bars) if levels.vwap else None
    for m in range(SIG_LO, cap + 1):
        b = bars.get(m)
        if b is None:
            continue
        close, vol = b[3], b[4]
        if vol < levels.vref:
            continue
        vw = vwap_series.get(m) if vwap_series is not None else 0.0
        if (
            close > levels.or_high
            and (not levels.vwap or close > vw)
            and not levels.short_only
        ):
            return m, "L"
        if (
            close < levels.or_low
            and (not levels.vwap or close < vw)
            and not levels.long_only
        ):
            return m, "S"
    return None


def _vwap_series(
    bars: dict[int, tuple[float, float, float, float, int]],
) -> dict[int, float]:
    """Cumulative VWAP starting at the open, keyed by minute."""
    pv = 0.0
    q = 0.0
    out: dict[int, float] = {}
    for m in range(OPEN_MIN, min(max(bars), EXIT_MIN) + 1):
        b = bars.get(m)
        if b is None:
            continue
        o, h, l, c, v = b
        pv += ((h + l + c) / 3.0) * v
        q += v
        out[m] = pv / q if q > 0 else o
    return out


@dataclass(frozen=True)
class Entry:
    entry_min: int
    side: str
    entry_price: float
    or_width: float


@dataclass(frozen=True)
class StopCtx:
    delay_r: float
    sign: float
    or_width: float
    entry_price: float
    trail: float
    cfg: Config


def _stop_update(
    ctx: StopCtx, engaged: bool, peak: float, lo: float, active_stop: float | None
) -> tuple[bool, float | None]:
    """Return (engaged, active_stop) after applying the configured trailing logic."""
    sign = ctx.sign
    if ctx.delay_r > 0.0 and not engaged:
        profit = (peak - ctx.entry_price) if sign > 0 else (ctx.entry_price - lo)
        if profit >= 2.0 * ctx.or_width:
            return True, peak - sign * ctx.or_width
        return engaged, active_stop
    if ctx.delay_r > 0.0:
        cand = peak - sign * ctx.or_width
        return engaged, max(active_stop, cand) if sign > 0 else min(active_stop, cand)
    if ctx.cfg.trail_atr > 0.0 and active_stop is not None:
        cand = peak - sign * ctx.trail
        return engaged, max(active_stop, cand) if sign > 0 else min(active_stop, cand)
    return engaged, active_stop


def apply_exit(
    bars: dict[int, tuple[float, float, float, float, int]],
    e: Entry,
    prior_ranges: list[float],
    cfg: Config,
) -> tuple[float, str, bool]:
    """Return (exit_price, reason, took_partial). Baseline = hold to close."""
    sign = 1.0 if e.side == "L" else -1.0
    atr = float(np.median(prior_ranges)) if len(prior_ranges) >= 5 else e.or_width
    trail = (cfg.trail_atr * atr) if cfg.trail_atr > 0.0 else cfg.stop_or * e.or_width
    stop = e.entry_price - sign * trail if trail > 0.0 else None
    exit_price: float | None = None
    exit_reason = "eod"
    peak = e.entry_price
    tp = (cfg.partial_r * e.or_width) if cfg.partial_r > 0.0 else 0.0
    took_partial = False
    exit_min = cfg.exit_ms[0]

    delay_r = cfg.delayed_trail
    engaged = delay_r <= 0.0
    active_stop = stop
    if delay_r > 0.0:
        active_stop = e.entry_price - sign * 2.0 * e.or_width

    for m in range(e.entry_min + 1, exit_min):
        b = bars.get(m)
        if b is None:
            continue
        hi, lo = b[1], b[2]
        peak = max(peak, hi) if sign > 0 else min(peak, lo)
        engaged, active_stop = _stop_update(
            StopCtx(delay_r, sign, e.or_width, e.entry_price, trail, cfg),
            engaged,
            peak,
            lo,
            active_stop,
        )
        hit = active_stop is not None and (
            (lo <= active_stop) if sign > 0 else (hi >= active_stop)
        )
        if hit:
            return active_stop, "stop", took_partial
        if tp > 0.0 and not took_partial:
            if (hi >= e.entry_price + tp) if sign > 0 else (lo <= e.entry_price - tp):
                took_partial = True
    eb = bars.get(exit_min)
    if eb is None:
        return e.entry_price, "invalid", took_partial
    return eb[3], exit_reason, took_partial


@dataclass
class Priors:
    ranges: list[float]
    orv: list[float]

    def median_range(self) -> float:
        return float(np.median(self.ranges)) if len(self.ranges) >= 5 else 0.0

    def mean_orv(self) -> float:
        return float(np.mean(self.orv[-14:])) if len(self.orv) >= 5 else 0.0

    def record(self, day_rng: float, or_vmean: float) -> None:
        self.ranges.append(day_rng)
        self.orv.append(or_vmean)


@dataclass(frozen=True)
class Config:
    cost_bps: float = 3.0
    vref_mult: float = 1.5
    sig_end: int = SIG_HI
    exit_ms: tuple[int, int] = (EXIT_MIN,)  # exit at close of first min in range
    trail_atr: float = 0.0
    stop_or: float = 0.0
    partial_r: float = 0.0
    day_filter: str = ""
    rv_filter: float = 0.0
    vwap_filter: bool = False
    or_n: int = OR_N
    delayed_trail: float = 0.0  # start trailing after +r*or_width, trail 1R behind


def run_strategy(
    sessions: dict[str, dict[int, tuple[float, float, float, float, int]]],
    cfg: Config,
) -> list[dict[str, Any]]:
    """Chronological single-pass simulation; returns per-trade dicts."""
    priors = Priors([], [])
    trades: list[dict[str, Any]] = []
    for key in sorted(sessions):
        bars = sessions[key]
        orb = opening_range(bars, cfg.or_n)
        if orb is None:
            continue
        or_high, or_low, or_vmean = orb
        or_width = or_high - or_low

        d = day_range(bars)
        med = priors.median_range()
        if cfg.day_filter == "expansion" and med > 0 and or_width <= med:
            priors.record(d, or_vmean)
            continue
        if cfg.day_filter == "contraction" and med > 0 and or_width >= med:
            priors.record(d, or_vmean)
            continue
        if cfg.rv_filter > 0.0:
            base = priors.mean_orv()
            if base <= 0 or or_vmean / base < cfg.rv_filter:
                priors.record(d, or_vmean)
                continue

        sig = find_signal(
            bars,
            Levels(or_high, or_low, cfg.vref_mult * or_vmean, cfg.vwap_filter),
            cfg.sig_end,
        )
        if sig is None:
            priors.record(day_range(bars), or_vmean)
            continue
        sig_min, side = sig
        entry_bar = bars.get(sig_min + 1)
        if entry_bar is None or sig_min + 1 >= cfg.exit_ms[0]:
            priors.record(day_range(bars), or_vmean)
            continue
        entry_price = entry_bar[0]
        exit_price, reason, took_partial = apply_exit(
            bars, Entry(sig_min + 1, side, entry_price, or_width), priors.ranges, cfg
        )
        gross = sign_pnl(side, entry_price, exit_price)
        if took_partial and cfg.partial_r > 0.0:
            tp = (
                entry_price + cfg.partial_r * or_width
                if side == "L"
                else entry_price - cfg.partial_r * or_width
            )
            first = 0.5 * sign_pnl(side, entry_price, tp)
            rest = 0.5 * sign_pnl(side, entry_price, exit_price)
            gross = first + rest
        cost = cfg.cost_bps / 1e4 * (entry_price + exit_price)
        net = gross - cost
        trades.append(
            {
                "date": key,
                "side": side,
                "sig_min": sig_min,
                "entry": round(entry_price, 4),
                "exit": round(exit_price, 4),
                "gross": round(gross, 6),
                "cost": round(cost, 6),
                "net": round(net, 6),
                "or_width": round(or_width, 4),
                "reason": reason,
            }
        )
        priors.record(day_range(bars), or_vmean)
    return trades


def sign_pnl(side: str, entry: float, exit: float) -> float:
    return (exit - entry) if side == "L" else (entry - exit)


def summarize(trades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"n": 0}
    nets = np.array([t["net"] for t in trades])
    gross = np.array([t["gross"] for t in trades])
    longs = np.array([t["net"] for t in trades if t["side"] == "L"])
    shorts = np.array([t["net"] for t in trades if t["side"] == "S"])
    ows = np.array([t["or_width"] for t in trades])
    return {
        "n": n,
        "n_long": int(len(longs)),
        "n_short": int(len(shorts)),
        "win_rate": float(np.mean(nets > 0)),
        "gross_sum": float(gross.sum()),
        "cost_sum": float(sum(t["cost"] for t in trades)),
        "net_sum": float(nets.sum()),
        "mean_net": float(nets.mean()),
        "median_net": float(np.median(nets)),
        "std_net": float(nets.std()),
        "ev_ratio": float(nets.mean() / np.mean(ows)) if ows.size else 0.0,
        "long_mean": float(longs.mean()) if longs.size else 0.0,
        "short_mean": float(shorts.mean()) if shorts.size else 0.0,
    }


def report(summary: dict[str, Any], label: str) -> str:
    if summary["n"] == 0:
        return f"{label:<28s} n=0"
    return (
        f"{label:<28s} n={summary['n']:6d} win={summary['win_rate']:5.1%} "
        f"meanNet=${summary['mean_net']:+8.4f} sumNet=${summary['net_sum']:+10.1f} "
        f"evRatio={summary['ev_ratio']:+6.3f} L={summary['long_mean']:+7.4f} "
        f"S={summary['short_mean']:+7.4f}"
    )


def _run_cfg(sessions: dict, cfg: Config, label: str) -> str:
    return report(summarize(run_strategy(sessions, cfg)), label)


def run_sweep(sessions: dict) -> int:
    print("\n-- baseline (user spec, 3bp one-way) --")
    print(_run_cfg(sessions, Config(), "user-spec 15m ORB"))
    print("\n-- cost sensitivity (user spec, one-way bps) --")
    for bp in (0.0, 1.0, 3.0, 6.0, 10.0):
        print(_run_cfg(sessions, Config(cost_bps=bp), f"cost {bp:g}bp"))
    print("\n-- vref multiplier --")
    for vm in (1.0, 1.25, 1.5, 2.0, 3.0):
        print(_run_cfg(sessions, Config(vref_mult=vm), f"vref x{vm}"))
    print("\n-- signal window cutoff (15:30 = all day) --")
    for se in (930, 840, 720, 660, 600):
        print(
            _run_cfg(sessions, Config(sig_end=se), f"signals until {se // 60:02d}:00")
        )
    print("\n-- exit-time cutoff (hold to close = 15:55) --")
    for e in (955, 900, 840, 720, 660):
        print(_run_cfg(sessions, Config(exit_ms=(e,)), f"exit at {e // 60:02d}:00"))
    print("\n-- VWAP alignment filter --")
    print(_run_cfg(sessions, Config(vwap_filter=True), "vwap-aligned entries"))
    _sweep_management(sessions)
    print("\n-- day-type filters (OR width vs prior-20d median) --")
    for df in ("expansion", "contraction"):
        print(_run_cfg(sessions, Config(day_filter=df), f"{df} day only"))
    print("\n-- relative-volume filter (OR vol / prior-14d OR vol) --")
    for rv in (0.5, 1.0, 1.5):
        print(_run_cfg(sessions, Config(rv_filter=rv), f"RV >= {rv:g}"))
    print("\n-- OR window length (bars) --")
    for n in (5, 10, 15, 30):
        print(_run_cfg(sessions, Config(or_n=n), f"{n}-min OR"))
    print("\n-- delayed trailing (trade after +2R, trail 1R) --")
    print(_run_cfg(sessions, Config(delayed_trail=1.0), "delayed-trail 1R"))
    return 0


def _sweep_management(sessions: dict) -> None:
    print("\n-- ATR trailing stop (x prior-20d median range) --")
    for k in (0.5, 1.0, 2.0, 3.0):
        print(_run_cfg(sessions, Config(trail_atr=k), f"trail ATR x{k}"))
    print("\n-- OR-width stop (no trail) --")
    for k in (0.5, 1.0, 1.5, 2.0):
        print(_run_cfg(sessions, Config(stop_or=k), f"stop OR x{k}"))
    print("\n-- partial take-profit (half at +r*ORw) --")
    for r in (0.5, 1.0, 1.5, 2.0):
        print(_run_cfg(sessions, Config(partial_r=r), f"partial +{r:g}R"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="QQQ Trend-ORB backtest")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--cost-bps", type=float, default=3.0)
    ap.add_argument("--vref-mult", type=float, default=1.5)
    ap.add_argument("--sig-end", type=int, default=SIG_HI)
    ap.add_argument("--trail-atr", type=float, default=0.0)
    ap.add_argument("--stop-or", type=float, default=0.0)
    ap.add_argument("--partial-r", type=float, default=0.0)
    ap.add_argument(
        "--day-filter", choices=["", "expansion", "contraction"], default=""
    )
    ap.add_argument("--rv-filter", type=float, default=0.0)
    ap.add_argument("--exit-min", type=int, default=EXIT_MIN)
    ap.add_argument("--vwap-filter", action="store_true")
    ap.add_argument("--or-n", type=int, default=OR_N)
    ap.add_argument("--delayed-trail", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--verify", type=Path, default=None, help="json evidence")
    args = ap.parse_args(argv)

    sessions = load_bars(args.csv)
    if not sessions:
        print("ERROR: no sessions parsed", file=sys.stderr)
        return 1
    print(f"sessions={len(sessions)} bars={sum(len(v) for v in sessions.values())}")
    if args.sweep:
        return run_sweep(sessions)

    cfg = Config(
        cost_bps=args.cost_bps,
        vref_mult=args.vref_mult,
        sig_end=args.sig_end,
        exit_ms=(args.exit_min,),
        trail_atr=args.trail_atr,
        stop_or=args.stop_or,
        partial_r=args.partial_r,
        day_filter=args.day_filter,
        rv_filter=args.rv_filter,
        vwap_filter=args.vwap_filter,
        or_n=args.or_n,
        delayed_trail=args.delayed_trail,
    )
    trades = run_strategy(sessions, cfg)
    print(report(summarize(trades), "result"))
    if trades:
        _print_split(trades)
        _print_years(trades)
        out = Path(args.csv).with_suffix(".trades.json")
        out.write_text(json.dumps(trades, indent=1))
        print(f"\nwrote {out}")
        if args.verify is not None:
            args.verify.write_text(json.dumps(summarize(trades), indent=1))
            print(f"wrote {args.verify}")
    return 0


def _print_split(trades: list[dict[str, Any]]) -> None:
    is_lo, is_hi = "2023-06-01", "2025-09-01"
    first, last = trades[0]["date"], trades[-1]["date"]
    if not (first < is_lo < is_hi <= last):
        return
    is_t = [t for t in trades if is_lo <= t["date"] < is_hi]
    oos_t = [t for t in trades if t["date"] >= is_hi]
    pre_t = [t for t in trades if t["date"] < is_lo]
    print()
    print(report(summarize(pre_t), f"pre-IS (< {is_lo})"))
    print(report(summarize(is_t), f"IS {is_lo}..{is_hi}"))
    print(report(summarize(oos_t), f"OOS >= {is_hi}"))


def _print_years(trades: list[dict[str, Any]]) -> None:
    by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        by_year[t["date"][:4]].append(t)
    print("\n-- per calendar year --")
    for year in sorted(by_year):
        print(report(summarize(by_year[year]), year))


if __name__ == "__main__":
    sys.exit(main())
