#!/usr/bin/env python3
"""scripts/fetch_alpaca.py — 1-min OHLCV research data from the Alpaca data API.

Pulls RTH 1-minute bars from the Alpaca v2 stocks/bars endpoint and writes
them as eyes-compatible CSVs (same format as the committed fixtures), so the
whole pipeline (eyes.load_ohlcv / wfa / phase7 bench) consumes them unchanged.

Primary purpose: acquire 60-180+ trading days of intraday history for the
Phase-7 lean-stack probe — the committed fixtures cover only ~5 days, too thin
for WFA's MIN_TRADES=30 floor. This is a RESEARCH set: it never replaces
data/fixtures and nothing ships off it. Production decisions still require the
protocol-gated Phase 4 panel.

Credentials are read first from the environment, then from the repo's
gitignored ``.env`` file (``scripts/`` loads it automatically, so the typical
invocation is just ``python scripts/fetch_alpaca.py``):

  ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY   (or short ALPACA_KEY / ALPACA_SECRET)

Alpaca data API uses the SAME credentials as the trading dashboard; a paper
account's keys work for the public data endpoint.

Notes (deliberately documented):
  * Bars come back in UTC; they are converted to America/New_York wall time so
    the 09:30-11:00 ET session gate sees the correct hours, and are filtered to
    09:30-15:59 to match the fixture convention (matches yfinance fetcher too).
  * The free data feed is IEX: volume is IEX-exchange only, not the consolidated
    tape, and unsupported/illiquid names may return empty. Acceptable for a
    relative benchmark; the definitive Phase-4 panel should use SIP or the
    live IB feed.

Usage:
  ALPACA_API_KEY_ID=... ALPACA_API_SECRET_KEY=... python scripts/fetch_alpaca.py \\
      --out-dir data/research/alpaca_180d --days 180
  python scripts/fetch_alpaca.py --tickers AAPL,SPY --days 60 \\
      --out-dir data/research/alpaca_60d
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from hanoon_prime.backtest import _discover_tickers

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
_BASE = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
_ET = ZoneInfo("America/New_York")
_MARKET_OPEN = "09:30"
_MARKET_CLOSE = "15:59"


@dataclass
class ApiCfg:
    """Alpaca data-API configuration + auth (kept out of per-symbol fns)."""

    key: str
    secret: str
    feed: str
    limit: int
    pause: float


_env_precedence = (
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "ALPACA_KEY",
    "ALPACA_SECRET",
)


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from the repo's gitignored .env (never override)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        var, _, value = line.partition("=")
        var = var.strip()
        if var and value and var not in os.environ:
            os.environ[var] = value.strip().strip('"')


def _env_key() -> tuple[str, str]:
    """Return (key_id, secret) from environment or the repo .env file."""
    _load_dotenv()
    key = os.environ.get(_env_precedence[0]) or os.environ.get(_env_precedence[2])
    secret = os.environ.get(_env_precedence[1]) or os.environ.get(_env_precedence[3])
    if not key or not secret:
        raise RuntimeError(
            "set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (or ALPACA_KEY / "
            "ALPACA_SECRET) in the environment or the repo .env file"
        )
    return key, secret


def _request(url: str, key: str, secret: str) -> dict[str, Any]:
    """One authenticated GET; raise on non-200 with the API error text."""
    req = urllib.request.Request(
        url,
        headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:200]}") from exc


def _to_et(ts: str) -> datetime:
    """Parse an Alpaca UTC timestamp and shift it to America/New_York."""
    return datetime.fromisoformat(ts).astimezone(_ET)


def _ts_str(ts: datetime) -> str:
    """Fixture-style timestamp string (``YYYY-MM-DD HH:MM:SS-04:00``)."""
    return ts.isoformat(sep=" ", timespec="seconds")


def _rth_bars(bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only regular-hours bars (09:30-15:59 ET) and sort by time."""
    out = []
    for b in bars:
        t = _to_et(b["t"])
        if _MARKET_OPEN <= t.strftime("%H:%M") <= _MARKET_CLOSE:
            out.append(b)
    out.sort(key=lambda b: b["t"])
    return out


def _fetch_symbol(
    symbol: str, cfg: ApiCfg, start: str, end: str
) -> list[dict[str, Any]]:
    """Fetch *all* 1-min bars for a symbol by walking page tokens."""
    bars: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, str] = {
            "timeframe": "1Min",
            "limit": str(cfg.limit),
            "feed": cfg.feed,
            "start": start,
            "end": end,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        url = _BASE.format(symbol=symbol) + "?" + urllib.parse.urlencode(params)
        body = _request(url, cfg.key, cfg.secret)
        chunk = body.get("bars") or []
        bars.extend(chunk)
        page_token = body.get("next_page_token")
        if not page_token or not chunk:
            break
        time.sleep(cfg.pause)
    return _rth_bars(bars)


def _session_count(bars: list[dict[str, Any]]) -> int:
    """Number of distinct trading dates (ET) present in the bars."""
    return len({_to_et(b["t"]).strftime("%Y-%m-%d") for b in bars})


def _write_csv(path: Path, symbol: str, bars: list[dict[str, Any]]) -> None:
    """Write an eyes-compatible CSV mirroring the committed fixture format."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Price", "Close", "High", "Low", "Open", "Volume"])
        writer.writerow(["Ticker", symbol, symbol, symbol, symbol, symbol])
        writer.writerow(["Datetime", "", "", "", "", ""])
        for b in bars:
            writer.writerow(
                [
                    _ts_str(_to_et(b["t"])),
                    repr(float(b["c"])),
                    repr(float(b["h"])),
                    repr(float(b["l"])),
                    repr(float(b["o"])),
                    int(round(float(b["v"]))),
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="fetch 1-min research CSVs from Alpaca"
    )
    parser.add_argument("--out-dir", default="data/research/alpaca_1m")
    parser.add_argument("--tickers", default="ALL")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--pause", type=float, default=0.35)
    args = parser.parse_args(argv)

    if args.tickers.upper() == "ALL":
        tickers = _discover_tickers(_FIXTURE_DIR)
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    try:
        key, secret = _env_key()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    start_s = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_s = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fetched: list[tuple[str, int, int]] = []
    failed: list[tuple[str, str]] = []
    cfg = ApiCfg(
        key=key, secret=secret, feed=args.feed, limit=args.limit, pause=args.pause
    )
    for ticker in tickers:
        try:
            bars = _fetch_symbol(ticker, cfg, start_s, end_s)
            if not bars:
                raise RuntimeError("no RTH bars returned")
            _write_csv(out_dir / f"{ticker}_1min.csv", ticker, bars)
            sessions = _session_count(bars)
            fetched.append((ticker, len(bars), sessions))
            print(f"{ticker:6s} bars={len(bars):6d} sessions={sessions:3d}")
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
