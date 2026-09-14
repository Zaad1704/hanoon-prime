#!/usr/bin/env python3
"""scripts/phase14_bench.py — Phase-14 PDH/PDL sweep-and-reclaim bench.

Usage:
  python scripts/phase14_bench.py --data-dir data/research/alpaca_180d \
      --output reports/phase14_bench_alpaca.json
  python scripts/phase14_bench.py --data-dir data/research/ibkr_5y_rth \
      --output reports/phase14_bench_ibkr5y.json

Runs the pre-registered strategy (protocols/sweep_reclaim_protocol.md) —
reclaim fade off the prior session's high/low, 10:00–11:30 entry window,
session-mid target, sweep-extreme stop, force-flat 15:50 — through the SAME
WFA verdict / deflation / PBO surface as phases 7-13. Fold indices are in
SESSION space (each ticker ~125 sessions on alpaca_180d, ~1250 on ibkr_5y).

Pre-registered diagnostics run FIRST and are REPORTED (not gating):
  1. RVOL decomposition: does the edge survive only on RVOL>1.2 reclaim
     bars? (the story's signature hypothesis)
  2. long/short split EV (symmetry check)
  3. before/after 10:30 entry split (front-loading of the morning window)
Exit code 0 = report written (the WF verdict is the gate, not this script).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hanoon_prime.backtest import _discover_tickers
from hanoon_prime.eyes import load_ohlcv
from hanoon_prime.phase14 import (
    ENTRY_FROM,
    ENTRY_TO,
    FEE_R,
    FOLDS,
    MIN_RR,
    SESSION_CLOSE,
    STOP_CUSHION_MULT,
    SWEEP_MAX_BARS,
    SweepTrade,
    run_sweep,
    sweep_trades,
)
from hanoon_prime.wfa import MIN_TRADES, _pooled_returns, verdicts

DATA_DIR = "data/research/alpaca_180d"
RVOL_SPLIT = 1.2  # pre-registered RVOL hypothesis threshold


def _pool(pnl: list[float]) -> dict[str, float]:
    if not pnl:
        return {"ev": 0.0, "wr": 0.0, "trades": 0.0, "rr": 0.0, "n_win": 0, "n_loss": 0}
    r = np.asarray(pnl, dtype=float)
    wins = r[r > 0]
    losses = r[r <= 0]
    aw = float(np.mean(wins)) if wins.size else 0.0
    al = abs(float(np.mean(losses))) if losses.size else 0.0
    rr = aw / al if al > 0 else 0.0
    wr = float(np.mean(r > 0))
    return {
        "ev": wr * rr - (1.0 - wr) if al > 0 else float(wr),
        "wr": wr,
        "trades": float(len(r)),
        "rr": rr,
        "n_win": int(wins.size),
        "n_loss": int(losses.size),
    }


def _flat_pnl(results: dict[str, list[Any]]) -> np.ndarray:
    return np.asarray(
        [p for folds in results.values() for f in folds for p in f.pnl],
        dtype=float,
    )


def _rvol_decomp(trades: list[SweepTrade]) -> dict[str, float]:
    """Split by the reclaim-bar RVOL captured at signal time (protocol §7)."""
    h, lo = [], []
    for t in trades:
        (h if t.rvol >= RVOL_SPLIT else lo).append(t.r - FEE_R)
    hp, lp = _pool(h), _pool(lo)
    return {
        "rvol_gt_1_2_ev": hp["ev"],
        "rvol_gt_1_2_trades": hp["trades"],
        "rvol_le_1_2_ev": lp["ev"],
        "rvol_le_1_2_trades": lp["trades"],
    }


def _half_diag(trades: list[SweepTrade]) -> dict[str, float]:
    """Long vs short and pre/post 10:30 splits (net of FEE_R)."""
    long_, short, early, late = [], [], [], []
    for t in trades:
        (long_ if t.side == "long" else short).append(t.r - FEE_R)
        (early if t.entry_ts[11:16] < "10:30" else late).append(t.r - FEE_R)
    return {
        "long_ev": _pool(long_)["ev"],
        "long_trades": _pool(long_)["trades"],
        "short_ev": _pool(short)["ev"],
        "short_trades": _pool(short)["trades"],
        "pre_1030_ev": _pool(early)["ev"],
        "pre_1030_trades": _pool(early)["trades"],
        "post_1030_ev": _pool(late)["ev"],
        "post_1030_trades": _pool(late)["trades"],
    }


def _load_all(data_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Load every non-SPY CSV; (ticker order, {ticker: data}). Has at least 1."""
    out: dict[str, Any] = {}
    for t in _discover_tickers(data_dir):
        if t == "SPY":
            continue
        try:
            out[t] = load_ohlcv(data_dir / f"{t}_1min.csv")
        except ValueError:
            continue
    return list(out.keys()), out


def _build_spy_veto(data_dir: Path):
    """spy_ok callable (long: SPY close > its session open; short: below), or
    None when no SPY file exists (veto off, flagged in the report)."""
    spy_file = data_dir / "SPY_1min.csv"
    if not spy_file.exists():
        return None
    spy = load_ohlcv(spy_file)
    opens: dict[str, float] = {}
    for i in range(len(spy["datetime"])):
        opens.setdefault(spy["datetime"][i][:10], float(spy["open"][i]))
    closes: dict[str, float] = {}
    for i in range(len(spy["datetime"])):
        closes[spy["datetime"][i]] = float(spy["close"][i])

    def spy_ok(ts: str) -> bool:
        if ts not in closes or ts[:10] not in opens:
            return True
        return closes[ts] > opens[ts[:10]]

    return spy_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 14 sweep-and-reclaim bench")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    tickers, all_data = _load_all(data_dir)
    if not tickers:
        print("no tickers found")
        return 2

    spy_ok = _build_spy_veto(data_dir)
    print(f"spy_veto={'on' if spy_ok is not None else 'off'}")

    # ── Pre-registered diagnostics (computed on the FULL sample first) ──
    all_trades: list[SweepTrade] = []
    for t in tickers:
        all_trades += sweep_trades(all_data[t], spy_ok)
    rvol = _rvol_decomp(all_trades)
    half = _half_diag(all_trades)
    print(
        f"rvol_decomp  RVOL>1.2 EV={rvol['rvol_gt_1_2_ev']:+.3f}R({int(rvol['rvol_gt_1_2_trades'])}t) "
        f"RVOL<=1.2 EV={rvol['rvol_le_1_2_ev']:+.3f}R({int(rvol['rvol_le_1_2_trades'])}t)"
    )
    print(
        f"long_short   long EV={half['long_ev']:+.3f}R({int(half['long_trades'])}t) "
        f"short EV={half['short_ev']:+.3f}R({int(half['short_trades'])}t)"
    )
    print(
        f"time_split   pre-10:30 EV={half['pre_1030_ev']:+.3f}R({int(half['pre_1030_trades'])}t) "
        f"post-10:30 EV={half['post_1030_ev']:+.3f}R({int(half['post_1030_trades'])}t)"
    )

    results: dict[str, list[Any]] = {}
    for t in tickers:
        folds = run_sweep(all_data[t], folds=FOLDS, spy_ok=spy_ok)
        if any(f.trades for f in folds):
            results[t] = folds

    if not results:
        print("no ticker produced any OOS sweep trade — INSUFFICIENT")
        return 2

    pnl = _flat_pnl(results)
    pool = _pool(list(pnl))
    net_pnl = np.asarray([p - FEE_R for p in pnl], dtype=float)
    net = _pool(list(net_pnl))
    adm = _pool(list(_pooled_returns(results)))
    v = verdicts(results)
    print(
        f"sweep       EV={pool['ev']:+.3f}R netEV={net['ev']:+.3f}R WR={pool['wr']:+.1%} "
        f"rr={pool['rr']:.2f} trades={int(pool['trades'])} verdict={v.verdict} "
        f"[adm EV={adm['ev']:+.3f}R/{int(adm['trades'])}t]"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "data": str(data_dir),
            "pre_registered": {
                "entry_from": ENTRY_FROM,
                "entry_to": ENTRY_TO,
                "session_close": SESSION_CLOSE,
                "sweep_max_bars": SWEEP_MAX_BARS,
                "stop_cushion_mult": STOP_CUSHION_MULT,
                "min_rr": MIN_RR,
                "fee_r": FEE_R,
                "frozen": "protocols/sweep_reclaim_protocol.md §6",
            },
            "spy_veto": spy_ok is not None,
            "min_trades_for_admissible": MIN_TRADES,
            "diagnostics": {
                "rvol": rvol,
                "long_short": half,
            },
            "pool": {
                k: (round(v_, 4) if isinstance(v_, float) else v_)
                for k, v_ in pool.items()
            },
            "net_pool": {
                k: (round(v_, 4) if isinstance(v_, float) else v_)
                for k, v_ in net.items()
            },
            "wfa": {
                "verdict": v.verdict,
                "pooled_sharpe": round(float(v.pooled_sharpe), 4),
                "deflated_edge": round(float(v.deflated_edge), 4),
                "pbo": round(float(v.pbo), 4),
                "admissible": len(v.admissible_tickers),
                "adm_ev": round(adm["ev"], 4),
                "adm_trades": round(adm["trades"], 1),
                "detail": v.detail,
            },
        }
        out.write_text(json.dumps(blob, indent=2) + "\n")
        md = out.with_suffix(".md")
        md.write_text(_render_md(pool, net, adm, v, rvol, half))
        print(f"\nBenchmark report written to {out} / {md}")
    return 0


def _render_md(
    pool: dict[str, float],
    net: dict[str, float],
    adm: dict[str, float],
    v: Any,
    rvol: dict[str, float],
    half: dict[str, float],
) -> str:
    return "\n".join(
        [
            "# Phase 14 — PDH/PDL Sweep-and-Reclaim (sandbox)",
            "",
            "Fade the 1-min wick through the prior session's High/Low that",
            "reclaims inside within 10 bars, entered at the reclaim close,",
            "stop beyond the sweep extreme, target the session Mid.",
            "Same WFA verdict/deflation/PBO surface as phases 7-13; folds",
            "indexed in sessions.",
            "",
            "## Pre-registered diagnostics (reported, not gating)",
            "",
            f"**RVOL decomposition:** RVOL>1.2 EV = {rvol['rvol_gt_1_2_ev']:+.3f}R "
            f"({int(rvol['rvol_gt_1_2_trades'])}t) vs RVOL<=1.2 EV = "
            f"{rvol['rvol_le_1_2_ev']:+.3f}R ({int(rvol['rvol_le_1_2_trades'])}t) — "
            "evidence surviving only the >1.2 bucket would confirm the story.",
            "",
            f"**Long/Short:** long EV = {half['long_ev']:+.3f}R "
            f"({int(half['long_trades'])}t) vs short EV = {half['short_ev']:+.3f}R "
            f"({int(half['short_trades'])}t).",
            "",
            f"**Time split:** pre-10:30 EV = {half['pre_1030_ev']:+.3f}R "
            f"({int(half['pre_1030_trades'])}t) vs post-10:30 EV = "
            f"{half['post_1030_ev']:+.3f}R ({int(half['post_1030_trades'])}t).",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| pooled EV (R) | {pool['ev']:+.3f} |",
            f"| net EV (R) | {net['ev']:+.3f} |",
            f"| pooled WR | {pool['wr']:.1%} |",
            f"| pooled R:R | {pool['rr']:.2f} |",
            f"| pool trades | {int(pool['trades'])} |",
            f"| admissible EV (R) | {adm['ev']:+.3f} |",
            f"| admissible trades | {int(adm['trades'])} |",
            f"| pooled Sharpe | {v.pooled_sharpe:+.3f} |",
            f"| deflated edge | {v.deflated_edge:+.3f} |",
            f"| PBO | {v.pbo:.2f} |",
            f"| verdict | {v.verdict} |",
            "",
            "## Interpretation",
            "",
            "**verdict = PASS** only if deflated edge > 0 AND pooled Sharpe > 0",
            "with ≥ MIN_TRADES admissible tickers. PBO < 0.5 is an analyst",
            "override criterion, not an engine gate. PASS here does NOT arm",
            "live capital — MicroLiveGuard still requires a paper run (§14).",
            "",
            f"Detail: {v.detail}",
            "",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
