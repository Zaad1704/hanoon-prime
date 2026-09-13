#!/usr/bin/env python3
"""scripts/phase12_bench.py — Phase-12 daily cross-sectional momentum bench.

Usage:
  python scripts/phase12_bench.py --data-dir data/research/alpaca_180d \
      --output reports/phase12_bench_alpaca.json

Runs the pre-registered §12.3 strategy (long top-7 / short bottom-7 by
trailing 5-session return, dollar-neutral, held session O→C) through the
SAME WFA verdict / deflation / PBO surface as phases 7-9, so the comparison
table stays apples-to-apples. Fold indices are in SESSION space (each ticker
~125 O→C observations → clears MIN_TRADES=30 trivially).

Pre-registered fail-fast gates (§12.4) run FIRST:
  1. decile monotonicity of O→C return vs signal   -> NO-GO if non-monotone
  2. gross vs net pooled EV                        -> record sign-flip if any
If either fails the bench still writes the report but carries the flag.
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
from hanoon_prime.phase12 import LOOKBACK, K, monotonicity_profile, run_spread
from hanoon_prime.wfa import MIN_TRADES, _pooled_returns, verdicts

DATA_DIR = "data/research/alpaca_180d"


def _pool(pnl: list[float]) -> dict[str, float]:
    if not pnl:
        return {"ev": 0.0, "wr": 0.0, "trades": 0.0, "rr": 0.0, "fees": 0.0}
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
        "fees": 0.0,
    }


def _flat_pnl(results: dict[str, list[Any]]) -> np.ndarray:
    return np.asarray(
        [p for folds in results.values() for f in folds for p in f.pnl],
        dtype=float,
    )


def _monotonicity_diag(
    tickers: list[str], data_dir: str
) -> tuple[dict[str, float], bool]:
    """Fail-fast §12.4.1: is decile O→C return monotone in the traded signal?

    Uses the cross-sectional per-session decile construction (see
    ``phase12.monotonicity_profile``). PASS requires strictly increasing
    d1..d10 (ties allowed within ±0.5bp noise).
    """
    prof = monotonicity_profile(tickers, data_dir)
    if len(prof) < 3:
        return prof, False
    means = [prof[f"d{i}"] for i in range(1, len(prof) + 1)]
    return prof, all(means[i + 1] > means[i] - 5e-5 for i in range(len(means) - 1))


def _gross_vs_net_diag(
    results: dict[str, list[Any]],
) -> tuple[dict[str, float], dict[str, float], bool]:
    """Fail-fast §12.4.2: gross (no fee) vs net EV on the pooled spread pnl.

    phase12 emits GROSS O→C pnl; net applies the §7 fee ledger (0.01%/leg +
    $0.01/trade). Returns (gross_pool, net_pool, sign_flip).
    """
    pnl = _flat_pnl(results)
    gross = _pool(list(pnl))
    net_pnl = np.asarray([p - 1e-4 for p in pnl], dtype=float)  # FEE_RATE 1bp/leg
    net = _pool(list(net_pnl))
    return gross, net, gross["ev"] > 0 and net["ev"] <= 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12 daily cross-sectional bench")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    tickers = [t for t in _discover_tickers(data_dir) if t != "SPY"]
    if not tickers:
        print("no tickers found")
        return 2

    # ── Phase 12.4 fail-fast diagnostics (pre-registered, run BEFORE bench) ──
    prof, mono_ok = _monotonicity_diag(tickers, str(data_dir))
    print(
        "monotonicity "
        + " ".join(
            f"d{i}={prof.get(f'd{i}', 0.0):+.4f}" for i in range(1, len(prof) + 1)
        )
    )
    print(f"monotone_decreasing_profile={'ok' if mono_ok else 'NO-GO'}")

    results = run_spread(tickers, str(data_dir))
    if not results:
        print("no ticker produced any OOS spread trade — INSUFFICIENT")
        return 2

    gross, net, flip = _gross_vs_net_diag(results)
    print(
        f"gross_vs_net gross EV={gross['ev']:+.3f}R net EV={net['ev']:+.3f}R "
        f"{'SIGN-FLIP' if flip else 'ok'}"
    )

    pnl = _flat_pnl(results)
    pool = _pool(list(pnl))
    adm = _pool(list(_pooled_returns(results)))
    v = verdicts(results)
    print(
        f"spread  EV={pool['ev']:+.3f}R WR={pool['wr']:.1%} rr={pool['rr']:.2f} "
        f"trades={int(pool['trades']):d} verdict={v.verdict} "
        f"[adm EV={adm['ev']:+.3f}R/{int(adm['trades'])}t]"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "data": str(data_dir),
            "pre_registered": {
                "lookback": LOOKBACK,
                "k": K,
                "hold": "session O->C",
                "frozen": "no parameter search; K=7 lookback=5 only",
            },
            "min_trades_for_admissible": MIN_TRADES,
            "fail_fast": {
                "monotonicity": {
                    "profile": {k: round(v, 4) for k, v in prof.items()},
                    "ok": mono_ok,
                },
                "gross_vs_net": {
                    "gross_ev": round(gross["ev"], 4),
                    "net_ev": round(net["ev"], 4),
                    "sign_flip": flip,
                },
            },
            "pool": {k: round(v, 4) for k, v in pool.items()},
            "wfa": {
                "verdict": v.verdict,
                "pooled_sharpe": round(float(v.pooled_sharpe), 4),
                "deflated_edge": round(float(v.deflated_edge), 4),
                "pbo": round(float(v.pbo), 4),
                "admissible": len(v.admissible_tickers),
                "adm_ev": round(adm["ev"], 4),
                "adm_wr": round(adm["wr"], 4),
                "adm_trades": round(adm["trades"], 1),
                "detail": v.detail,
            },
        }
        out.write_text(json.dumps(blob, indent=2) + "\n")
        md = out.with_suffix(".md")
        md.write_text(_render_md(pool, adm, v, prof, mono_ok, gross, net, flip))
        print(f"\nBenchmark report written to {out} / {md}")
    return 0


def _render_md(
    pool: dict[str, float],
    adm: dict[str, float],
    v: Any,
    prof: dict[str, float],
    mono_ok: bool,
    gross: dict[str, float],
    net: dict[str, float],
    flip: bool,
) -> str:
    mono_line = " ".join(
        f"d{i}={prof.get(f'd{i}', 0.0):+.4f}" for i in range(1, len(prof) + 1)
    )
    return "\n".join(
        [
            "# Phase 12 — Daily Cross-Sectional Momentum (sandbox)",
            "",
            "Long top-7 / short bottom-7 by trailing 5-session return,",
            "dollar-neutral, held session open→close. Same WFA verdict/",
            "deflation/PBO surface as phases 7-9; folds indexed in sessions.",
            "",
            "## Phase 12.4 fail-fast diagnostics (pre-registered)",
            "",
            f"**Monotonicity (§12.4.1):** {'PASS' if mono_ok else 'NO-GO'}",
            "",
            f"```",
            mono_line,
            f"```",
            "",
            f"**Gross vs Net (§12.4.2):** gross EV = {gross['ev']:+.3f}R, "
            f"net EV = {net['ev']:+.3f}R — "
            f"{'SIGN-FLIP' if flip else 'no sign flip'}",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| pooled EV (R) | {pool['ev']:+.3f} |",
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
            "live capital — MicroLiveGuard still requires a paper run (see §12.7).",
            "",
            f"Detail: {v.detail}",
            "",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
