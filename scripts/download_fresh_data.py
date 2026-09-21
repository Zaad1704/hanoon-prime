#!/usr/bin/env python3
"""Download fresh 1-minute OHLCV data from yfinance for all tickers.

Usage:
    python scripts/download_fresh_data.py --output-dir data/market_data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yfinance as yf


def get_ticker_list() -> list[str]:
    """Return the canonical 254-ticker list from market_data."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "market_data"
    if not data_dir.exists():
        return []
    return sorted(p.stem.replace("_1min", "") for p in data_dir.glob("*_1min.csv"))


def download_ticker(ticker: str, output_dir: Path) -> bool:
    """Download 7 days of 1-minute data for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="7d", interval="1m")
        if df.empty or len(df) < 10:
            return False
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index.name = "Datetime"
        out_path = output_dir / f"{ticker}_1min.csv"
        df.to_csv(out_path)
        return True
    except Exception as exc:
        print(f"  FAIL {ticker}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "market_data"),
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--tickers",
        default="ALL",
        help="Comma-separated tickers or ALL",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tickers.upper() == "ALL":
        tickers = get_ticker_list()
        if not tickers:
            print("No existing tickers found. Provide --tickers.", file=sys.stderr)
            return 1
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]

    print(f"Downloading 1-minute data for {len(tickers)} tickers...")
    ok, fail = 0, 0
    for i, ticker in enumerate(tickers):
        success = download_ticker(ticker, output_dir)
        if success:
            ok += 1
        else:
            fail += 1
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)} ({ok} ok, {fail} fail)")
        time.sleep(0.15)  # rate limit

    print(f"Done: {ok} downloaded, {fail} failed out of {len(tickers)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
