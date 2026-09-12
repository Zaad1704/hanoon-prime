#!/usr/bin/env python3
"""scripts/fetch_yfinance.py — fetch 1-min OHLCV research data from yfinance.

Downloads RTH 1-minute OHLCV for a universe (default: all committed fixture
tickers plus SPY) into a research directory as eyes-compatible CSVs, so the
whole pipeline (eyes.load_ohlcv / wfa / phase7 bench) consumes it unchanged.

Why this exists: the committed fixtures cover ~5 trading days — too thin for
WFA (MIN_TRADES=30). IB Gateway historical pulls are slow; yfinance gives the
last ~30 calendar days of 1-minute bars quickly. This is a RESEARCH set only:
it never replaces data/fixtures, and nothing ships off it. Use it to probe
whether the Phase-7 lean stack flips positive on deeper data.

Yahoo limits 1-minute granularity to ~8 calendar days per request, so the
fetch walks BACK from today in weekly chunks and concatenates them (each
calendar day earlier than ``--days`` is dropped). ``--pause`` is the delay
inserted after every request to stay under Yahoo's rate limit.

Caveats (documented deliberately):
  * Yahoo intraday can contain gaps, duplicate bars, or half-day sessions;
    duplicates are dropped and RTH bars are filtered to 09:30-15:59 to match
    the fixture convention.
  * Close is auto-adjusted for splits/dividends (factor-consistent, which is
    what return-based factors need; absolute price level differs from raw).
  * Production decisions still require the protocol-gated Phase 4 panel.

Usage:
  python scripts/fetch_yfinance.py --out-dir data/research/yfinance_1m
  python scripts/fetch_yfinance.py --tickers AAPL,SPY --days 30
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from hanoon_prime.backtest import _discover_tickers

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
_STEP_DAYS = 7  # stays under Yahoo's ~8-day 1m granularity cap


def _ts_str(obj: Any) -> str:
    """Format a yfinance index label into the fixture timestamp string."""
    return pd.Timestamp(obj).isoformat(sep=" ")


def _fetch_chunk(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch one [start, end) RTH 1-min chunk, deduped and sorted."""
    data = yf.Ticker(ticker).history(
        start=start, end=end, interval="1m", prepost=False, auto_adjust=True
    )
    if data is None or data.empty:
        return pd.DataFrame()
    data = data[~data.index.duplicated(keep="last")]
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    data = data.sort_index()
    return data.between_time("09:30", "15:59")


def _fetch_one(ticker: str, days: int, pause: float) -> pd.DataFrame:
    """Walk back from today in weekly chunks and concatenate the sessions."""
    frames: list[pd.DataFrame] = []
    now = datetime.now()
    end = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    for _ in range(0, days, _STEP_DAYS):
        start = end - timedelta(days=_STEP_DAYS)
        chunk = _fetch_chunk(ticker, start, end)
        if not chunk.empty:
            frames.append(chunk)
        time.sleep(pause)
        end = start
    if not frames:
        raise RuntimeError("no 1-min data returned")
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _session_count(index: pd.DatetimeIndex) -> int:
    """Number of distinct trading dates in the index."""
    return len({ts.strftime("%Y-%m-%d") for ts in index})


def _write_csv(path: Path, ticker: str, data: pd.DataFrame) -> None:
    """Write an eyes-compatible CSV mirroring the committed fixture format."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Price", "Close", "High", "Low", "Open", "Volume"])
        writer.writerow([f"Ticker,{ticker},{ticker},{ticker},{ticker},{ticker}"])
        writer.writerow(["Datetime,,,,,"])
        for ts, row in data.iterrows():
            writer.writerow(
                [
                    _ts_str(ts),
                    repr(float(row["Close"])),
                    repr(float(row["High"])),
                    repr(float(row["Low"])),
                    repr(float(row["Open"])),
                    int(round(float(row["Volume"]))),
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="fetch 1-min research CSVs from yfinance"
    )
    parser.add_argument("--out-dir", default="data/research/yfinance_1m")
    parser.add_argument("--tickers", default="ALL")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--pause", type=float, default=1.5)
    args = parser.parse_args(argv)

    if args.tickers.upper() == "ALL":
        tickers = _discover_tickers(_FIXTURE_DIR)
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fetched: list[tuple[str, int, int]] = []
    failed: list[tuple[str, str]] = []
    for ticker in tickers:
        try:
            data = _fetch_one(ticker, args.days, args.pause)
            _write_csv(out_dir / f"{ticker}_1min.csv", ticker, data)
            sessions = _session_count(data.index)
            fetched.append((ticker, len(data), sessions))
            print(f"{ticker:6s} bars={len(data):6d} sessions={sessions:3d}")
        except Exception as exc:
            failed.append((ticker, str(exc)))
            print(f"{ticker:6s} SKIP ({exc})")
        time.sleep(args.pause)

    if failed:
        print("\n--- failed ---")
        for ticker, reason in failed:
            print(f"{ticker:6s} {reason}")

    if not fetched:
        print("ERROR: nothing fetched", file=sys.stderr)
        return 1
    print("\nFetched", len(fetched), "tickers into", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
