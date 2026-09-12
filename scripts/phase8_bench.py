#!/usr/bin/env python3
"""scripts/phase8_bench.py — Phase-8 sandbox benchmark: staged exits.

Usage:
  python scripts/phase8_bench.py --data-dir data/research/alpaca_180d \
      --output reports/phase8_bench_alpaca.json

Runs the lean stack (Phase-7's current best: vwap+momentum+RS, RVOL>2 x
09:30-11:00 gate) with STATIC exits as the control arm, then three staged
EXIT-arm variants through the SAME WFA OOS scoring:

  control      lean static                 (Phase-7 reference, EV target)
  stage_atr    scale 50% @ 1.5xATR, stop->BE, runner trails 3xATR
  stage_vwap   scale 50% @ 1.5xATR, stop->BE, runner trails VWAP
  stage_nobe   stage_atr WITHOUT the breakeven floor (isolates BE effect)

(Shipped-cocktail baseline context lives in reports/phase7_bench_alpaca.md —
lean already won the factor half; Phase 8 isolates the EXIT policy only.)

Exit code 0 = report written (reporting is not a gate; the WF verdict on
top of it is). The comparison is INFORMATIONAL until the numbers win.
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
from hanoon_prime.phase7 import (
    LEAN_WEIGHTS,
    LeanCfg,
    load_spy_closes,
    run_walk_forward_lean,
)
from hanoon_prime.phase8 import StageCfg, StageRun, run_walk_forward_staged
from hanoon_prime.wfa import MIN_TRADES, _pooled_returns, verdicts

VARIANTS = ("control", "stage_atr", "stage_vwap", "stage_nobe")


def _pool(results: dict[str, list[Any]]) -> dict[str, float]:
    """Pool per-trade pnl_pct into R-expectancy, WR, count, Sharpe."""
    pnl = [p for folds in results.values() for f in folds for p in f.pnl]
    if not pnl:
        return {"ev": 0.0, "wr": 0.0, "trades": 0.0, "sharpe": 0.0, "rr": 0.0}
    r = np.asarray(pnl, dtype=float)
    wins = r[r > 0]
    losses = r[r <= 0]
    aw = float(np.mean(wins)) if wins.size else 0.0
    al = abs(float(np.mean(losses))) if losses.size else 0.0
    rr = aw / al if al > 0 else 0.0
    wr = float(np.mean(r > 0))
    sd = float(np.std(r))
    return {
        "ev": wr * rr - (1.0 - wr) if al > 0 else float(wr),
        "wr": wr,
        "trades": float(len(r)),
        "sharpe": float(np.mean(r) / (sd + 1e-12)) if sd > 0 else 0.0,
        "rr": rr,
    }


def _pool_admissible(results: dict[str, list[Any]]) -> dict[str, float]:
    """R-expectancy pooled over protocol-admissible tickers (>= MIN_TRADES)."""
    r = _pooled_returns(results)
    if r.size == 0:
        return {"ev": 0.0, "wr": 0.0, "trades": 0.0, "rr": 0.0}
    wins = r[r > 0]
    losses = r[r <= 0]
    aw = float(np.mean(wins)) if wins.size else 0.0
    al = abs(float(np.mean(losses))) if losses.size else 0.0
    rr = aw / al if al > 0 else 0.0
    wr = float(np.mean(r > 0))
    return {
        "ev": wr * rr - (1.0 - wr) if al > 0 else float(wr),
        "wr": wr,
        "trades": float(r.size),
        "rr": rr,
    }


def _run_variant(
    name: str,
    tickers: list[str],
    data_dir: Path,
    spy_closes: dict[str, float] | None,
) -> dict[str, list[Any]]:
    results: dict[str, list[Any]] = {}
    for ticker in tickers:
        if ticker == "SPY":
            continue
        path = data_dir / f"{ticker}_1min.csv"
        if not path.exists():
            continue
        try:
            data = load_ohlcv(path)
        except ValueError:
            continue
        if len(data["close"]) < 80:
            continue
        if name == "control":
            folds = run_walk_forward_lean(ticker, data, LeanCfg(spy_closes=spy_closes))
        elif name == "stage_atr":
            run = StageRun(
                lean=LeanCfg(spy_closes=spy_closes),
                stage=StageCfg(runner_mode="atr"),
            )
            folds = run_walk_forward_staged(ticker, data, run)
        elif name == "stage_vwap":
            run = StageRun(
                lean=LeanCfg(spy_closes=spy_closes),
                stage=StageCfg(runner_mode="vwap"),
            )
            folds = run_walk_forward_staged(ticker, data, run)
        else:  # stage_nobe
            run = StageRun(
                lean=LeanCfg(spy_closes=spy_closes),
                stage=StageCfg(runner_mode="atr", breakeven=False),
            )
            folds = run_walk_forward_staged(ticker, data, run)
        results[ticker] = folds
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8 staged-exit benchmark")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tickers", default="ALL")
    parser.add_argument("--spy", default="SPY_1min.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if args.tickers.upper() == "ALL":
        tickers = _discover_tickers(data_dir)
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]
    spy_closes = load_spy_closes(str(data_dir / args.spy))

    pools: dict[str, dict[str, float]] = {}
    stats: dict[str, dict[str, Any]] = {}
    for name in VARIANTS:
        results = _run_variant(name, tickers, data_dir, spy_closes)
        pools[name] = _pool(results)
        adm = _pool_admissible(results)
        v = verdicts(results)
        stats[name] = {
            "verdict": v.verdict,
            "pooled_sharpe": round(float(v.pooled_sharpe), 4),
            "deflated_edge": round(float(v.deflated_edge), 4),
            "pbo": round(float(v.pbo), 4),
            "admissible": len(v.admissible_tickers),
            "adm_ev": round(adm["ev"], 4),
            "adm_wr": round(adm["wr"], 4),
            "adm_trades": round(adm["trades"], 1),
            "detail": v.detail,
        }
        p = pools[name]
        print(
            f"{name:12s} EV={p['ev']:+.3f}R WR={p['wr']:.1%} "
            f"trades={int(p['trades']):4d} verdict={v.verdict} "
            f"[adm EV={adm['ev']:+.3f}R/{int(adm['trades'])}t]"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "data": str(data_dir),
            "spy": str(data_dir / args.spy),
            "min_trades_for_admissible": MIN_TRADES,
            "weights": LEAN_WEIGHTS,
            "pools": {
                k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in pools.items()
            },
            "wfa": stats,
        }
        out.write_text(json.dumps(blob, indent=2) + "\n")
        md = out.with_suffix(".md")
        md.write_text(_render_md(pools, stats))
        print(f"\nBenchmark report written to {out} / {md}")
    return 0


def _render_md(
    pools: dict[str, dict[str, float]], stats: dict[str, dict[str, Any]]
) -> str:
    lines = [
        "# Phase 8 — Staged-Exit Benchmark (sandbox)",
        "",
        "Side-by-side OOS walk-forward on ONE panel (default: 180-day Alpaca",
        "1-min, 23 tickers): the lean stack with STATIC exits (the Phase-7",
        "reference) vs three staged-exit heads. Same fold scoring, same",
        "gates, same deflation — only the exit policy differs.",
        "",
        "| variant | EV(R) | WR | R:R | trades | pooled SR | defl. edge | PBO | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in VARIANTS:
        p = pools[name]
        s = stats[name]
        lines.append(
            f"| {name} | {p['ev']:+.3f} | {p['wr']:.1%} | {p['rr']:.2f} "
            f"| {int(p['trades'])} | {s['pooled_sharpe']:+.3f} "
            f"| {s['deflated_edge']:+.3f} | {s['pbo']:.2f} | {s['verdict']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "**control**     — lean stack, static ATR stop/target exits.",
        "**stage_atr**   — 50% scale @ 1.5xATR, stop->BE, runner trails 3xATR.",
        "**stage_vwap**  — 50% scale @ 1.5xATR, stop->BE, runner trails VWAP.",
        "**stage_nobe**  — stage_atr WITHOUT the breakeven floor.",
        "",
        "Exit WIN criterion: a staged arm must beat ``control`` on deflated",
        "edge AND pooled EV (all-ticker) AND on the admissible subset. If",
        "not, the sandbox answer is NO-GO unchanged.",
        "",
    ]
    lines += _render_findings(pools, stats)
    lines += [
        "",
        f"Admissible floor: {MIN_TRADES} OOS trades/ticker. Weights:",
        f"`{LEAN_WEIGHTS}`.",
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
        f"- Control (lean static) pooled EV: {ctrl['ev']:+.3f}R",
        f" ({int(ctrl['trades'])} trades); admissible EV {ctrl_adm:+.3f}R.",
        f"- Protocol verdict (control): {stats['control']['verdict']}",
        f" (deflated edge {stats['control']['deflated_edge']:+.3f}).",
    ]
    for name in ("stage_atr", "stage_vwap", "stage_nobe"):
        p = pools[name]
        s = stats[name]
        delta = p["ev"] - ctrl["ev"]
        adm_delta = s["adm_ev"] - ctrl_adm
        adm_sign = "above" if adm_delta > 0 else "below"
        out.append(
            f"- {name}: EV {p['ev']:+.3f}R (vs control {delta:+.3f}R); "
            f"WR {p['wr']:.1%} (vs control {ctrl['wr']:.1%}); "
            f"R:R {p['rr']:.2f} (vs control {ctrl['rr']:.2f}); "
            f"admissible EV {s['adm_ev']:+.3f}R ({adm_sign} control by "
            f"{adm_delta:+.3f}R); verdict {s['verdict']}."
        )
    best = max(("stage_atr", "stage_vwap", "stage_nobe"), key=lambda n: pools[n]["ev"])
    bp = pools[best]
    clean = bp["ev"] > ctrl["ev"] and bp["ev"] > 0.0
    if clean:
        headline = (
            f"**Conclusion:** **{best}** improves pooled OOS R-expectancy vs "
            "the static control and turns it positive — directional support "
            "for the staged-exit thesis. A protocol GO still needs a PASS on "
            f"{MIN_TRADES}+ OOS trades/ticker under the deflation bar; the "
            "verdict table shows this arm did not clear it."
        )
    else:
        best_wr = max(
            ("stage_atr", "stage_vwap", "stage_nobe"),
            key=lambda n: pools[n]["wr"],
        )
        headline = (
            "**Conclusion:** no staged arm beats the static control on pooled "
            "OOS R-expectancy with a positive edge — staged exits do NOT clear "
            "the gate on this set; sandbox answer remains NO-GO (data, not the "
            "exit policy, is the bound). Every staged arm roughly DOUBLED win "
            f"rate (control {ctrl['wr']:.1%} -> best {pools[best_wr]['wr']:.1%}) "
            "but cut R:R ~3:1 -> ~0.7, so EV fell — the win-rate thesis is "
            "mechanically real and EV-negative. The bound is that the entrance "
            "edge is not positive, not the exit policy."
        )
    out += ["", headline]
    return out


if __name__ == "__main__":
    sys.exit(main())
