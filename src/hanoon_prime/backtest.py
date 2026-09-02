"""hanoon_prime.backtest — runs the ACTUAL brain pipeline on ACTUAL historical data.

Top-level orchestrator. The heavy simulation logic lives in _sim.

CI uses this to enforce R2: no merges if expectancy < 0 on any ticker.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .brain import Brain
from ._sim import simulate_ticker
from ._metrics import calculate_metrics
from .data import load_ohlcv


def backtest_ticker(
    ticker: str,
    close, high, low, volume,
    window: int = 30,
    output_dir: Path | None = None,
) -> dict:
    """Backtest the full JULI pipeline on one ticker's 1-min bars.

    Returns metrics dict (also writes JSON if output_dir given).
    """
    trades, equity_curve = simulate_ticker(ticker, close, high, low, volume, window)
    metrics = calculate_metrics(ticker, trades, equity_curve)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{ticker}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def run_backtest(
    tickers: list[str],
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    window: int = 30,
) -> tuple[dict[str, dict], list[str]]:
    """Run backtest on all tickers. Returns (results, errors)."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else None
    results: dict[str, dict] = {}
    errors: list[str] = []

    for ticker in tickers:
        csv_path = data_dir / f"{ticker}_1min.csv"
        if not csv_path.exists():
            print(f"SKIP: {csv_path} not found")
            errors.append(f"{ticker}: no data file")
            continue

        data = load_ohlcv(csv_path)
        close = data["close"]

        if len(close) < window + 10:
            print(f"SKIP: {ticker} — only {len(close)} bars")
            errors.append(f"{ticker}: insufficient bars ({len(close)})")
            continue

        metrics = backtest_ticker(
            ticker, close, data["high"], data["low"], data["volume"],
            window=window, output_dir=output_dir,
        )
        results[ticker] = metrics

        status = "✅" if metrics["ev_per_trade"] > 0 else "❌"
        tr = f"{metrics['total_trades']} trades"
        wr = f"WR={metrics.get('win_rate', 0):.1%}" if metrics["total_trades"] else "WR=NA"
        ev = f"EV={metrics['ev_per_trade']:.3f}R" if metrics["total_trades"] else "EV=0"
        print(f"  {ticker:8s} {status}  {tr}  {wr}  {ev}  R:R={metrics.get('realized_rr', 0):.2f}")

    return results, errors


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="HANOON PRIME backtest")
    parser.add_argument("--tickers", required=True, help="Comma-separated (or 'ALL')")
    parser.add_argument("--data-dir", required=True, help="Directory with *_1min.csv files")
    parser.add_argument("--output", default=None, help="Output dir for metrics JSON")
    parser.add_argument("--window", type=int, default=30, help="Lookback window (default 30)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.tickers.upper() == "ALL":
        tickers = sorted(f.stem.replace("_1min", "") for f in data_dir.glob("*_1min.csv"))
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]

    print(f"Running backtest on {len(tickers)} tickers from {data_dir}...")
    print(f"  Window: {args.window} bars")
    print()

    results, _ = run_backtest(tickers, data_dir, args.output, args.window)
    profitable = sum(1 for m in results.values() if m["ev_per_trade"] > 0)
    total_trades = sum(m["total_trades"] for m in results.values())
    print(f"\n{'='*60}")
    print(f"RESULTS: {profitable}/{len(results)} tickers profitable | {total_trades} trades")
    print(f"{'='*60}")
    unprofitable = [t for t, m in results.items()
                    if m["ev_per_trade"] <= 0 and m["total_trades"] > 0]
    if unprofitable:
        print(f"\n💀 PROFITABILITY GATE FAILED: {unprofitable}")
        return 1
    print("\n✅ All tickers pass profitability gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
