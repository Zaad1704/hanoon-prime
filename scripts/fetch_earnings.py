#!/usr/bin/env python3
"""scripts/fetch_earnings.py — yfinance earnings dates for the research universe.

Pulls scheduled earnings timestamps (Eastern) for every universe ticker and
writes ``data/research/earnings/{TICKER}.json`` as:

  {"ticker": "AAPL", "earnings_et": ["2025-06-12T17:00:00-04:00", ...]}

Phase 9 (catalyst screen) maps each timestamp to a TRADING session:
  * hour < 16:00 ET  -> the SAME calendar day (Before-Market-Open report)
  * hour >= 16:00 ET -> the next trading session present in the RTH panel
                        (After-Market-Close report drives the next morning)

Earnings DATES are announced weeks in advance, so they are knowable at each
session without lookahead; a (rare) reschedule shows the final date and is a
documented minor survivorship residual in the sandbox.

This is a RESEARCH artifact (network + vendor schedule). It never ships off
data/research and nothing in the live path depends on it.

Usage:
  python scripts/fetch_earnings.py --out-dir data/research/earnings
  python scripts/fetch_earnings.py --tickers AAPL,SPY --out-dir /tmp/earn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

from hanoon_prime.backtest import _discover_tickers

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
_ET = ZoneInfo("America/New_York")


def _earnings_rows(ticker: str, limit: int) -> list[str]:
    """ISO (Eastern) timestamps of scheduled earnings for ``ticker``."""
    cal = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    out = []
    for ts in cal.index:
        if hasattr(ts, "tzinfo") and getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(_ET)
        out.append(ts.astimezone(_ET).isoformat())
    return out


def _cutoff_iso(days_back: int) -> str:
    """Date-only ISO cutoff so recent earnings can be filtered lexically."""
    return (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fetch yfinance earnings dates")
    parser.add_argument("--out-dir", default="data/research/earnings")
    parser.add_argument("--tickers", default="ALL")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--days-back", type=int, default=400)
    parser.add_argument("--pause", type=float, default=0.25)
    args = parser.parse_args(argv)

    if args.tickers.upper() == "ALL":
        tickers = _discover_tickers(_FIXTURE_DIR)
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    failed: list[tuple[str, str]] = []
    cutoff = _cutoff_iso(args.days_back)
    for ticker in tickers:
        try:
            rows = [r for r in _earnings_rows(ticker, args.limit) if r >= cutoff]
            payload = {"ticker": ticker, "earnings_et": rows}
            (out_dir / f"{ticker}.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
            written.append(ticker)
            print(f"{ticker:6s} earnings={len(rows):3d}")
        except Exception as exc:  # network/vendor parse failures -> skip, report
            failed.append((ticker, str(exc)))
            print(f"{ticker:6s} SKIP ({exc})")
        time.sleep(args.pause)

    if failed:
        print("\n--- failed ---")
        for ticker, reason in failed:
            print(f"{ticker:6s} {reason}")

    if not written:
        print("ERROR: nothing fetched", file=sys.stderr)
        return 1
    print(f"\nWrote {len(written)} earnings calendars into {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
