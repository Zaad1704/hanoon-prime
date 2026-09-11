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
    if not metrics_dir.exists() or not any(metrics_dir.glob("*.json")):
        print(f"R2 VIOLATION — no backtest metrics found in {metrics_dir}")
        print("   The backtest ran on zero data (gate is vacuous). Fix the pipeline,")
        print("   not the risk management.")
        return 1

    failures = []
    total_trades = 0
    for f in sorted(metrics_dir.glob("*.json")):
        data = json.loads(f.read_text())
        ticker = data.get("ticker", f.stem)
        expectancy = data.get("ev_per_trade", None)
        if expectancy is None:
            expectancy = data.get("expectancy", 0.0)
        trades = int(data.get("total_trades", 0) or 0)
        total_trades += trades

        if trades == 0:
            failures.append(
                f"  {ticker}: 0 trades — insufficient data to prove profitability"
            )
        elif expectancy <= 0:
            failures.append(f"  {ticker}: EV/trade = {expectancy:.4f} (must be > 0)")

    if total_trades == 0:
        failures.append("  (all tickers combined) 0 trades total")

    if failures:
        print("R2 VIOLATION — profitability gate failed:")
        for f in failures:
            print(f)
        print("\n💀 The backtest shows no positive expectancy. Fix the alpha,")
        print("   not the risk management.")
        return 1

    print("R2 OK — all tickers show positive expectancy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
