#!/usr/bin/env python3
"""scripts/phase9_bench.py — Phase-9 Path A catalyst-screen benchmark.

Usage:
  python scripts/phase9_bench.py --rth-dir data/research/alpaca_180d \
      --premkt-dir data/research/alpaca_180d_pre \
      --earnings-dir data/research/earnings \
      --output reports/phase9_bench_alpaca.json

Four arms over the SAME WFA OOS scoring (same panel, same deflation):

  control       lean static, every session      (phase7 reference: -0.116R)
  catalyst      earnings-day AND pre-market RVOL>5 sessions only
  earnings_only any scheduled-earnings session   (isolates the RVOL filter)
  rvol_only     pre-market RVOL>5 sessions       (isolates the calendar)

All funnel inputs (earnings date, pre-market volume) are knowable before
09:30 ET, so no RTH information leaks into the entry decision. Exit code 0 =
report written. The comparison is INFORMATIONAL; the sandbox answer is
NO-GO unless the catalyst arm beats control on pooled + admissible EV with a
positive edge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from phase8_bench import _pool, _pool_admissible

from hanoon_prime.backtest import _discover_tickers
from hanoon_prime.eyes import load_ohlcv
from hanoon_prime.phase7 import (
    LEAN_WEIGHTS,
    LeanCfg,
    load_spy_closes,
    run_walk_forward_lean,
)
from hanoon_prime.phase9 import (
    earnings_session_dates,
    load_earnings,
    premkt_flagged_dates,
    run_walk_forward_catalyst,
)
from hanoon_prime.wfa import MIN_TRADES, _pooled_returns, verdicts

VARIANTS = ("control", "catalyst", "earnings_only", "rvol_only")


def _per_ticker(
    ticker: str, rth_dir: Path, premkt_dir: Path, earnings_dir: Path
) -> dict[str, Any] | None:
    """Load one ticker's data + its three allowed-session sets (or None)."""
    path = rth_dir / f"{ticker}_1min.csv"
    if not path.exists():
        return None
    try:
        data = load_ohlcv(path)
    except ValueError:
        return None
    if len(data["close"]) < 80:
        return None
    from hanoon_prime.phase9 import _sorted_date_keys

    rth_sorted = _sorted_date_keys(list(data["datetime"]))
    earnings_set = earnings_session_dates(
        load_earnings(ticker, earnings_dir), rth_sorted
    )
    premkt_csv = premkt_dir / f"{ticker}_1min.csv"
    rvol_set = premkt_flagged_dates(premkt_csv) if premkt_csv.exists() else set()
    return {
        "data": data,
        "earnings": earnings_set,
        "rvol": rvol_set,
        "catalyst": earnings_set & rvol_set,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 9 catalyst-screen benchmark")
    parser.add_argument("--rth-dir", required=True)
    parser.add_argument("--premkt-dir", required=True)
    parser.add_argument("--earnings-dir", required=True)
    parser.add_argument("--tickers", default="ALL")
    parser.add_argument("--spy", default="SPY_1min.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    rth_dir, premkt_dir = Path(args.rth_dir), Path(args.premkt_dir)
    earnings_dir = Path(args.earnings_dir)
    if args.tickers.upper() == "ALL":
        tickers = [t for t in _discover_tickers(rth_dir) if t != "SPY"]
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]

    spy = load_spy_closes(str(rth_dir / args.spy))
    catalog: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        entry = _per_ticker(ticker, rth_dir, premkt_dir, earnings_dir)
        if entry is not None:
            catalog[ticker] = entry
            entry["sessions"] = {
                "earnings": len(entry["earnings"]),
                "rvol": len(entry["rvol"]),
                "catalyst": len(entry["catalyst"]),
            }

    pools: dict[str, dict[str, float]] = {}
    stats: dict[str, dict[str, Any]] = {}
    run_map = {
        "control": lambda d, dates: run_walk_forward_lean(
            d["tag"], d["data"], LeanCfg(spy_closes=spy)
        ),
        "catalyst": lambda d, dates: run_walk_forward_catalyst(
            d["tag"], d["data"], LeanCfg(spy_closes=spy), dates
        ),
        "earnings_only": lambda d, dates: run_walk_forward_catalyst(
            d["tag"], d["data"], LeanCfg(spy_closes=spy), dates
        ),
        "rvol_only": lambda d, dates: run_walk_forward_catalyst(
            d["tag"], d["data"], LeanCfg(spy_closes=spy), dates
        ),
    }
    date_map = {
        "catalyst": lambda d: d["catalyst"],
        "earnings_only": lambda d: d["earnings"],
        "rvol_only": lambda d: d["rvol"],
    }

    for name in VARIANTS:
        results: dict[str, list[Any]] = {}
        for ticker, d in catalog.items():
            d["tag"] = ticker
            dates = date_map[name](d) if name in date_map else None
            results[ticker] = run_map[name](d, dates)
        pools[name] = _pool(results)
        adm = _pool_admissible(results)
        v = verdicts(results)
        sessions = (
            0
            if name == "control"
            else sum(len(date_map[name](d)) for d in catalog.values())
        )
        contributing = (
            len(catalog)
            if name == "control"
            else sum(1 for d in catalog.values() if date_map[name](d))
        )
        stats[name] = {
            "verdict": v.verdict,
            "pooled_sharpe": round(float(v.pooled_sharpe), 4),
            "deflated_edge": round(float(v.deflated_edge), 4),
            "pbo": round(float(v.pbo), 4),
            "admissible": len(v.admissible_tickers),
            "adm_ev": round(adm["ev"], 4),
            "adm_wr": round(adm["wr"], 4),
            "adm_trades": round(adm["trades"], 1),
            "flagged_sessions": sessions if name != "control" else None,
            "contributing_tickers": contributing,
            "detail": v.detail,
        }
        p = pools[name]
        sess = "" if sessions == 0 else f" sessions={sessions}/{contributing}tkr"
        print(
            f"{name:14s} EV={p['ev']:+.3f}R WR={p['wr']:.1%} R:R={p['rr']:.2f} "
            f"trades={int(p['trades']):4d}{sess} verdict={v.verdict} "
            f"[adm EV={adm['ev']:+.3f}R/{int(adm['trades'])}t]"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "rth": str(rth_dir),
            "premkt": str(premkt_dir),
            "earnings": str(earnings_dir),
            "min_trades_for_admissible": MIN_TRADES,
            "weights": LEAN_WEIGHTS,
            "pools": {
                k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in pools.items()
            },
            "catalog_sessions": {t: d["sessions"] for t, d in catalog.items()},
            "wfa": stats,
        }
        out.write_text(json.dumps(blob, indent=2) + "\n")
        out.with_suffix(".md").write_text(_render_md(pools, stats))
        print(f"\nBenchmark report written to {out} / {out.with_suffix('.md')}")
    return 0


def _render_md(
    pools: dict[str, dict[str, float]], stats: dict[str, dict[str, Any]]
) -> str:
    lines = [
        "# Phase 9 — Catalyst-Screen Benchmark (Path A, sandbox)",
        "",
        "Earnings-day x pre-market volume screen over the same 180-day",
        "Alpaca panel used by Phases 7-8. One fold protocol, same deflation,",
        "only the per-session entry universe changes.",
        "",
        "| variant | EV(R) | WR | R:R | trades | flagged sess | pooled SR | defl. edge | PBO | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in VARIANTS:
        p = pools[name]
        s = stats[name]
        sess = "—" if name == "control" else f"{s['flagged_sessions']}"
        lines.append(
            f"| {name} | {p['ev']:+.3f} | {p['wr']:.1%} | {p['rr']:.2f} "
            f"| {int(p['trades'])} | {sess} | {s['pooled_sharpe']:+.3f} "
            f"| {s['deflated_edge']:+.3f} | {s['pbo']:.2f} | {s['verdict']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "**control**      — lean stack, every session (Phase-7 reference).",
        "**catalyst**     — scheduled-earnings trading day AND pre-market RVOL>5.",
        "**earnings_only**— any scheduled-earnings session (isolates pre-market 5x).",
        "**rvol_only**    — pre-market RVOL>5 any day (isolates the calendar).",
        "",
        "WIN criterion: ``catalyst`` must beat ``control`` on pooled EV AND",
        "admissible EV AND carry a positive deflated edge. Any verdict short of",
        "that (negatives included) keeps the sandbox answer NO-GO.",
        "",
    ]
    lines += _render_findings(pools, stats)
    lines += [
        "",
        f"Admissible floor: {MIN_TRADES} OOS trades/ticker. Weights: `{LEAN_WEIGHTS}`.",
    ]
    return "\n".join(lines) + "\n"


def _render_findings(
    pools: dict[str, dict[str, float]], stats: dict[str, dict[str, Any]]
) -> list[str]:
    ctrl = pools["control"]
    ctrl_adm = stats["control"]["adm_ev"]
    out = [
        "## Findings (computed from the run)",
        "",
        f"- Control (every session): pooled EV {ctrl['ev']:+.3f}R, WR {ctrl['wr']:.1%}, "
        f"R:R {ctrl['rr']:.2f}, {int(ctrl['trades'])} trades; admissible EV {ctrl_adm:+.3f}R.",
    ]
    for name in ("catalyst", "earnings_only", "rvol_only"):
        p = pools[name]
        s = stats[name]
        delta = p["ev"] - ctrl["ev"]
        adm_delta = s["adm_ev"] - ctrl_adm
        adm_sign = "above" if adm_delta > 0 else "below"
        if s["adm_trades"] > 0:
            adm_note = (
                f"admissible EV {s['adm_ev']:+.3f}R ({adm_sign} control by "
                f"{adm_delta:+.3f}R)"
            )
        else:
            adm_note = "no admissible subset (0 trades clears MIN_TRADES)"
        out.append(
            f"- {name}: EV {p['ev']:+.3f}R (vs control {delta:+.3f}R); WR {p['wr']:.1%} "
            f"(vs {ctrl['wr']:.1%}); R:R {p['rr']:.2f}; {int(p['trades'])} trades "
            f"across {s['flagged_sessions']} flagged sessions; {adm_note}; "
            f"verdict {s['verdict']}."
        )
    cat = pools["catalyst"]
    cs = stats["catalyst"]
    clean = bool(
        cs["flagged_sessions"] > 0
        and cat["ev"] > ctrl["ev"]
        and cat["ev"] > 0.0
        and cs["adm_ev"] > ctrl_adm
    )
    if clean:
        headline = (
            "**Conclusion:** the catalyst screen beats control on pooled OOS EV with a "
            "positive edge — directional support for the earnings catalyst thesis. A "
            "protocol GO still needs a PASS (30+ OOS trades/ticker, deflation bar); the "
            f"verdict table shows {cs['verdict']} on this panel."
        )
    else:
        headline = (
            "**Conclusion:** the catalyst screen did NOT overcome the negative OOS "
            "expectancy on this panel — sandbox answer remains NO-GO. With only ~2 "
            "earnings sessions per ticker per 180d, the catalyst universe is structurally "
            "too thin for the 30-trade floor; the number to fix is panel depth "
            "(years, not months), not the screen."
        )
    out += ["", headline]
    return out


if __name__ == "__main__":
    sys.exit(main())
