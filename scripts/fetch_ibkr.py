#!/usr/bin/env python3
"""scripts/fetch_ibkr.py — multi-year 1-min research CSVs from the IB Gateway.

Phase-10 depth source. Pages the TWS API backward month-by-month for 1-minute
TRADES bars with useRTH=False (04:00-19:59 ET coverage) and writes
eyes-compatible CSVs in EXACTLY the format fetch_alpaca.py emits, so the whole
pipeline (eyes.load_ohlcv / wfa / phase7-9 benches) consumes them unchanged:

  Price,Close,High,Low,Open,Volume
  Ticker,AAPL,AAPL,AAPL,AAPL,AAPL
  Datetime,,,,
  2021-01-04 09:30:00-05:00,<c>,<h>,<l>,<o>,<v>

``--window`` selects which ET session slice is kept (mirrors fetch_alpaca):
  * ``rth``         — 09:30-15:59 ET (strategy inputs + SPY closes)
  * ``pre-market``  — 08:00-09:25 ET (Phase-9 catalyst screen: pre-market RVOL)
  * ``post-market`` — 16:00-19:59 ET

Primary purpose: remove the Phase 9 NO-GO — the 180d Alpaca panel holds only ~2
earnings sessions/ticker; YEARS of pre-market + RTH history is required for the
30-trade admissible floor. This is a RESEARCH set; nothing ships off it.

Prerequisites / notes (probe-verified 2026-09-12):
  * IB Gateway running on 127.0.0.1:4002 (paper), API + historical downloads
    enabled. No API key needed (local gateway auth), unlike Alpaca/Polygon.
  * --client-id must not collide with a running bot (bot uses 1).
  * One month of 1-min bars returns in ~40-60s; pacing (60 req / 10 min) is
    NOT binding at that rate. ~2y x 23 tickers ≈ 1.5-2.5h per window.
  * Empty/errored pages (Error 162, connectivity blips) are retried with
    backoff — the probe hit exactly this mid-fetch.
  * IBKR volume is consolidated-tape shares (vs Alpaca free tier = IEX only);
    sparse names (VALE, SNOW) carry full 90-min pre-market blocks here.

Usage:
  python scripts/fetch_ibkr.py --years 2 --window rth
  python scripts/fetch_ibkr.py --years 2 --window pre-market \\
      --out-dir data/research/ibkr_2y_pre
  python scripts/fetch_ibkr.py --tickers VALE,SNOW --window pre-market --months 3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import IB, Stock

from hanoon_prime.backtest import _discover_tickers

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
_ET = ZoneInfo("America/New_York")
_UTC = timezone.utc
HOST, PORT, CLIENT_ID = "127.0.0.1", 4002, 6

_SESS = {
    "rth": (570, 959),  # 09:30-15:59 ET
    "pre-market": (480, 569),  # 08:00-09:25 ET (fetch_alpaca parity)
    "post-market": (960, 1199),  # 16:00-19:59 ET
}
_SLICE = {"rth": "rth", "pre-market": "premarket", "post-market": "postmarket"}
_MAX_ATTEMPTS = 5


def _et_minute(bar: object) -> int:
    """ET wall-clock minute-of-day for an IB bar."""
    t = bar.date.astimezone(_ET)
    return t.hour * 60 + t.minute


def _utc_key(bar: object) -> str:
    """UTC instant key for cross-page dedupe (bars can straddle a month edge)."""
    return bar.date.astimezone(_UTC).isoformat()


def _fmt_et(bar: object) -> str:
    """Fixture-style ET timestamp (``YYYY-MM-DD HH:MM:SS-04:00``)."""
    return bar.date.astimezone(_ET).isoformat(sep=" ", timespec="seconds")


def _one_request(ib: IB, contract: Stock, end_dt: datetime, timeout: float) -> list:
    """One 1-month 1-min page ending at ``end_dt`` (US/Eastern)."""
    end_s = end_dt.strftime("%Y%m%d %H:%M:%S US/Eastern")
    return ib.reqHistoricalData(
        contract,
        end_s,
        "1 M",
        "1 min",
        "TRADES",
        False,  # useRTH=False -> keep 04:00-19:59 ET extended sessions
        1,
        False,
        [],
        timeout=timeout,
    )


def _fetch_month(
    ib: IB, contract: Stock, end_dt: datetime, timeout: float, pause: float
) -> list:
    """Page one month with retry-on-empty (Error 162 / connectivity blips)."""
    last_err = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            bars = _one_request(ib, contract, end_dt, timeout)
            if bars:
                return bars
            last_err = "empty response"
        except Exception as exc:  # transient gateway errors -> retry with backoff
            last_err = str(exc)
        time.sleep(pause * attempt)
    raise RuntimeError(f"{contract.symbol} page ended {end_dt}: {last_err}")


def _fetch_ticker(
    ib: IB,
    stack: Stock,
    end: datetime,
    start: datetime,
    window: str,
    timeout: float,
    pause: float,
    max_pages: int,
) -> tuple[list[object], int]:
    """All 1-min bars in (start, end] within ``window`` via month paging."""
    seen: dict[str, object] = {}
    months, cursor = 0, end
    while cursor > start and months < max_pages:
        bars = _fetch_month(ib, stack, cursor, timeout, pause)
        months += 1
        lo, hi = _SESS[window]
        for b in bars:
            if lo <= _et_minute(b) <= hi:
                seen[_utc_key(b)] = b
        cursor = min(b.date for b in bars) - timedelta(minutes=5)
    ordered = sorted(seen.values(), key=lambda b: _utc_key(b))
    return ordered, months


def _write_csv(path: Path, symbol: str, bars: list[object]) -> None:
    """Write an eyes-compatible CSV in the fetch_alpaca.py format."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Price", "Close", "High", "Low", "Open", "Volume"])
        writer.writerow(["Ticker", symbol, symbol, symbol, symbol, symbol])
        writer.writerow(["Datetime", "", "", "", "", ""])
        for b in bars:
            writer.writerow(
                [
                    _fmt_et(b),
                    repr(float(b.close)),
                    repr(float(b.high)),
                    repr(float(b.low)),
                    repr(float(b.open)),
                    int(round(float(b.volume))),
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="fetch multi-year 1-min research CSVs from the IB Gateway"
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tickers", default="ALL")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--months", type=int, default=0, help="cap pages (tests)")
    parser.add_argument("--window", choices=sorted(_SESS), default="rth")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--client-id", type=int, default=CLIENT_ID)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.tickers.upper() == "ALL":
        tickers = _discover_tickers(_FIXTURE_DIR)
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(f"data/research/ibkr_{args.years}y_{_SLICE[args.window]}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now(_UTC)
    start = end - timedelta(days=365 * args.years)
    max_pages = args.months or (12 * args.years)

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
    except Exception as exc:
        print(f"ERROR connect {args.host}:{args.port} clientId={args.client_id}: {exc}")
        print("  -> ensure IB Gateway is running (paper port 4002 by default)")
        return 1
    print(
        f"CONNECTED {args.host}:{args.port} clientId={args.client_id} "
        f"accounts={list(ib.managedAccounts())}"
    )

    fetched: list[tuple[str, int, int]] = []
    failed: list[tuple[str, str]] = []
    for ticker in tickers:
        try:
            stack = Stock(ticker, "SMART", "USD")
            ib.qualifyContracts(stack)
            if not stack:
                raise RuntimeError("contract not found")
            filled, months = _fetch_ticker(
                ib,
                stack,
                end,
                start,
                args.window,
                args.timeout,
                args.pause,
                max_pages,
            )
            if not filled:
                raise RuntimeError(f"no {args.window} bars returned")
            _write_csv(out_dir / f"{ticker}_1min.csv", ticker, filled)
            sessions = len({_fmt_et(b)[:10] for b in filled})
            fetched.append((ticker, len(filled), sessions))
            print(
                f"{ticker:6s} {args.window:11s} bars={len(filled):7d} "
                f"sessions={sessions:4d} pages={months:3d}"
            )
        except Exception as exc:
            failed.append((ticker, str(exc)))
            print(f"{ticker:6s} SKIP ({exc})")

    ib.disconnect()
    if failed:
        print("\n--- failed ---")
        for ticker, reason in failed:
            print(f"{ticker:6s} {reason}")
    if not fetched:
        print("ERROR: nothing fetched", file=sys.stderr)
        return 1
    print(f"\nFetched {len(fetched)} tickers into {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
