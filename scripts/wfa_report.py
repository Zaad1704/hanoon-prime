#!/usr/bin/env python3
"""scripts/wfa_report.py — run Phase-2 walk-forward and emit reports.

Usage:
  python scripts/wfa_report.py --data-dir data/fixtures \
      --output reports/phase2_wfa.json

Writes JSON + adjacent Markdown. Exit code:
  0 → universe PASS
  1 → universe FAIL or INSUFFICIENT (blocking, honest)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hanoon_prime.backtest import _discover_tickers
from hanoon_prime.wfa import (
    DEFAULT_FOLDS,
    MIN_TRADES,
    run_wfa_universe,
    serialize,
    verdicts,
)


def _render_md(verdict: object, universe: dict) -> str:
    """Human-readable Markdown report from the serialized verdict."""
    v = verdict
    lines = [
        "# Phase 2 — Walk-Forward Validation Report",
        "",
        f"Universe verdict: **{v['verdict']}**",
        f"- Pooled OOS Sharpe: `{v['pooled_sharpe']}`",
        f"- Deflated edge: `{v['deflated_edge']}`",
        f"- PBO (prob. backtest overfitting): `{v['pbo']}`",
        f"- Min OOS trades floor: `{MIN_TRADES}`",
        "",
        f"Detail: {v['detail']}",
        "",
        "## Per-ticker OOS (admissible only)",
        "",
        "| ticker | oos trades | EV/trade | sharpe | verdict |",
        "|---|---|---|---|---|",
    ]
    for t in v["admissible_tickers"]:
        lines.append(
            f"| {t['ticker']} | {t['oos_trades']} | {t['ev_per_trade']} "
            f"| {t['sharpe']} | {t['verdict']} |"
        )
    if not v["admissible_tickers"]:
        lines.append("| _(none met the min-trade floor — insufficient data)_ |")
    lines += ["", "## Per-ticker raw OOS trade counts"]
    for ticker, folds in sorted(universe.items()):
        n = sum(w.total_trades for w in folds)
        lines.append(f"- {ticker}: {n} OOS trades")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 walk-forward validation")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tickers", default="ALL")
    parser.add_argument("--output", default=None)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.tickers.upper() == "ALL":
        tickers = _discover_tickers(data_dir)
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]

    universe = run_wfa_universe(tickers, data_dir, folds=args.folds)
    verdict = verdicts(universe)
    blob = serialize(verdict)
    blob["ticker_count"] = len(universe)
    blob["fold_config"] = {"folds": args.folds, "min_trades": MIN_TRADES}

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        md = out.with_suffix(".md")
        md.write_text(_render_md(blob, universe))
        print(f"WFA report written to {out} / {md}")

    print(
        f"WFA verdict={verdict.verdict} ticks={len(universe)} "
        f"pooled_sr={verdict.pooled_sharpe:.3f} "
        f"deflated={verdict.deflated_edge:.3f} pbo={verdict.pbo:.2f}"
    )
    print(verdict.detail)

    return 0 if verdict.verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
