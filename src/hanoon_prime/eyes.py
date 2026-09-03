"""hanoon_prime.eyes — data ingestion from 1-minute OHLCV CSV files.

The ONLY data path in this system. Reads CSVs with the format:
    Price,Close,High,Low,Open,Volume     <-- header
    Ticker,AAPL,AAPL,AAPL,AAPL,AAPL       <-- ticker row
    Datetime,,,,,                         <-- datetime header
    2026-07-06 09:30:00,...,2848644       <-- data rows

The brain never touches a file path — it only sees numpy arrays.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

# Expected column order in data rows (after headers are skipped)
# CSV: datetime, close, high, low, open, volume
_COL_CLOSE: int = 1
_COL_HIGH: int = 2
_COL_LOW: int = 3
_COL_OPEN: int = 4
_COL_VOLUME: int = 5


def _is_header_row(row: list[str]) -> bool:
    """Detect header/metadata rows to skip."""
    if len(row) < 2:
        return True
    first = row[0].strip().lower()
    if first in ("datetime", "date", "time", "ticker", "price"):
        return True
    try:
        if not first or not first[0].isdigit():
            return True
    except (TypeError,):
        return True
    try:
        float(row[_COL_CLOSE])
        return False
    except (ValueError, IndexError):
        return True


def _parse_csv_row(row: list[str]) -> tuple[float, ...] | None:
    """Parse a data row into (datetime, open, high, low, close, volume)."""
    if len(row) < 6:
        return None
    try:
        return (
            float(row[_COL_OPEN]),
            float(row[_COL_HIGH]),
            float(row[_COL_LOW]),
            float(row[_COL_CLOSE]),
            float(row[_COL_VOLUME]),
        )
    except (ValueError, IndexError):
        return None


def _read_csv(path: str | Path) -> dict[str, list[Any]]:
    """Read CSV, skipping header rows."""
    dt: list[str] = []
    o, h, l, c, v = [], [], [], [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if _is_header_row(row):
                continue
            parsed = _parse_csv_row(row)
            if parsed is None:
                continue
            dt.append(row[0])
            o.append(parsed[0])
            h.append(parsed[1])
            l.append(parsed[2])
            c.append(parsed[3])
            v.append(parsed[4])
    return {"datetime": dt, "open": o, "high": h, "low": l, "close": c, "volume": v}


def load_ohlcv(path: str | Path) -> dict[str, Any]:
    """Load a 1-minute OHLCV CSV file.

    Returns dict with keys: datetime, open, high, low, close, volume.
    All values except datetime are numpy float arrays.
    """
    cols = _read_csv(path)
    if not cols["close"]:
        raise ValueError(f"No data rows in {path}")
    return {
        "datetime": cols["datetime"],
        "open": np.array(cols["open"]),
        "high": np.array(cols["high"]),
        "low": np.array(cols["low"]),
        "close": np.array(cols["close"]),
        "volume": np.array(cols["volume"]),
    }


def compute_buy_volume(
    close: Any,
    high: Any,
    low: Any,
    volume: Any,
) -> Any:
    """Estimate buy-side volume from OHLCV.

    buy_fraction = clamp((close - low) / (high - low + 1e-12), 0, 1)
    If close is near high, most volume was buying; near low, mostly selling.
    """
    rng = np.maximum(high - low, 1e-12)
    buy_frac = np.clip((close - low) / rng, 0.0, 1.0)
    return volume * buy_frac


def estimate_bid_ask(volume: Any, buy_volume: Any) -> tuple[Any, Any]:
    """Estimate aggregate bid/ask sizes from buy/sell volume.

    Returns (bid_sizes, ask_sizes) arrays for orderbook_imbalance.
    Uses the last 10 bars' cumulative buy/sell split.
    """
    sell_volume = np.maximum(volume - buy_volume, 0)
    total = np.maximum(volume, 1e-12)
    n = min(10, len(total))
    bid_size = float(np.sum(buy_volume[-n:]) / n / np.mean(total[-n:]))
    ask_size = float(np.sum(sell_volume[-n:]) / n / np.mean(total[-n:]))
    return np.array([bid_size]), np.array([ask_size])


def rolling_atr(
    high: Any,
    low: Any,
    close: Any,
    period: int = 14,
) -> float:
    """Compute ATR (Average True Range) over the last *period* bars."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if len(h) < 2:
        return 0.0
    tr1 = h[1:] - l[1:]
    tr2 = np.abs(h[1:] - c[:-1])
    tr3 = np.abs(l[1:] - c[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    n = min(period, len(tr))
    return float(np.mean(tr[-n:]))


def compute_vwap(close: Any, volume: Any) -> float:
    """Compute volume-weighted average price over the window."""
    v = np.asarray(volume, dtype=float)
    c = np.asarray(close, dtype=float)
    total_v = float(np.sum(v))
    if total_v <= 0:
        return float(c[-1]) if len(c) else 0.0
    return float(np.sum(c * v) / total_v)
