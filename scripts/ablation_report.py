#!/usr/bin/env python3
"""scripts/ablation_report.py — run the Phase-3 ablation matrix and emit a report.

Usage:
  python scripts/ablation_report.py --data-dir data/fixtures \
      --output reports/phase3_ablation.json

Writes JSON + adjacent Markdown. Exit code is 0 (report generation is not a
gate; the R2/WF gates judge profitability — this just answers "which organ
carries the edge?").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hanoon_prime.ablation import (
    baseline_variant,
    build_variants,
    contribution,
    pool,
    rank_factors,
    run_variant,
)
from hanoon_prime.backtest import _discover_tickers


def _render_md(pools: dict[str, dict[str, float]], data_dir: str) -> str:
    base = pools["baseline"]
    ranked = rank_factors(pools)
    lines = [
        "# Phase 3 — Per-Organ Ablation Report",
        "",
        f"Data: `{data_dir}` · pool = mean over tickers with trades",
        f"Baseline: EV/trade `{base['ev_mean']:+.3f}R`, "
        f"trades `{int(base['n_trades'])}`, "
        f"WR `{base['win_rate']:.1%}`, R:R `{base['realized_rr']:.2f}`",
        "",
        "Marginal contribution = baseline_ev − variant_ev.",
        "**Positive** = component carries edge (removing it hurt).",
        "**Negative** = component harmful (removing it helped).",
        "**~0** = inert.",
        "",
        "## Factor contribution ranking (cerebellum alpha)",
        "",
        "| factor | contribution (R) | off-EV | trades |",
        "|---|---|---|---|",
    ]
    for name, contrib in ranked:
        p = pools[name]
        lines.append(
            f"| {name.removeprefix('off_')} | {contrib:+.3f} "
            f"| {p['ev_mean']:+.3f} | {int(p['n_trades'])} |"
        )
    if not ranked:
        lines.append("| _(no factors ranked)_ | | | |")
    lines += [
        "",
        "## Organ toggles (all variants vs baseline)",
        "",
        "| variant | EV/trade | ΔEV | trades | WR | R:R | sharp | DD | ret% |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, p in pools.items():
        delta: str = "—"
        if name != "baseline":
            delta = f"{contribution(p, base):+.3f}"
        lines.append(
            f"| {name} | {p['ev_mean']:+.3f} | {delta} "
            f"| {int(p['n_trades'])} | {p['win_rate']:.1%} "
            f"| {p['realized_rr']:.2f} | {p['sharpe']:.2f} "
            f"| {p['drawdown']:.2%} | {p['return_pct']:+.1f}% |"
        )
    lines += [
        "",
        "## Caveat",
        "",
        "Committed fixtures span only ~5 trading days (~1,900 bars / ticker).",
        "Per-ticker trade counts are low; these deltas rank *direction of*",
        "marginal contribution, they do not reach Phase-2 statistical",
        "significance on their own. The WF gate remains the verdict on",
        "whether any variant is *actually* better.",
        "",
        "Sharpe is a mean across per-ticker sharpe (clipped to ±20; near-flat",
        "equity curves blow up otherwise). Drawdown is mean per-ticker max",
        "drawdown in per-share terms, so it can exceed 100% (sizing scaled up",
        "through a negative equity drift) and is *not* a portfolio drawdown.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 ablation matrix")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tickers", default="ALL")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if args.tickers.upper() == "ALL":
        tickers = _discover_tickers(data_dir)
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]

    variants = [baseline_variant()] + build_variants()
    pools: dict[str, dict[str, float]] = {}
    for v in variants:
        results = run_variant(v, tickers, data_dir)
        pools[v.name] = pool(results)
        p = pools[v.name]
        print(
            f"{v.name:18s} EV={p['ev_mean']:+.3f}R trades={int(p['n_trades']):4d} "
            f"WR={p['win_rate']:.1%} R:R={p['realized_rr']:.2f}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "data_dir": str(data_dir),
            "pools": pools,
            "factor_ranking": [list(kv) for kv in rank_factors(pools)],
        }
        out.write_text(json.dumps(blob, indent=2) + "\n")
        md = out.with_suffix(".md")
        md.write_text(_render_md(pools, str(data_dir)))
        print(f"Ablation report written to {out} / {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
