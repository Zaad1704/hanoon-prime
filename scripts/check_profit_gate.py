#!/usr/bin/env python3
"""scripts/check_profit_gate.py — Enforce R2: fail if any ticker shows negative EV.

Usage: python scripts/check_profit_gate.py <metrics_dir>/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_profit_gate.py <metrics_dir>")
        return 2

    metrics_dir = Path(sys.argv[1])
    if not metrics_dir.exists():
        print(f"WARNING: metrics dir {metrics_dir} does not exist yet — skipping gate")
        print("PROFIT GATE: PENDING (no metrics yet)")
        return 0

    failures = []
    for f in sorted(metrics_dir.glob("*.json")):
        data = json.loads(f.read_text())
        ticker = data.get("ticker", f.stem)
        expectancy = data.get("ev_per_trade", None)
        if expectancy is None:
            expectancy = data.get("expectancy", 0.0)

        if expectancy is not None and expectancy <= 0:
            failures.append(f"  {ticker}: EV/trade = {expectancy:.4f} (must be > 0)")

    if failures:
        print("R2 VIOLATION — profitability gate failed for these tickers:")
        for f in failures:
            print(f)
        print("\n💀 The backtest shows negative expectancy. Fix the alpha,")
        print("   not the risk management.")
        return 1

    print("R2 OK — all tickers show positive expectancy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
