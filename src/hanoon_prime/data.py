"""hanoon_prime.data — data loading from 1-minute OHLCV CSV files.

The ONLY data path in this system. Reads CSVs with the format:
    Price,Close,High,Low,Open,Volume     <-- header
    Ticker,AAPL,AAPL,AAPL,AAPL,AAPL       <-- ticker row
    Datetime,,,,,                         <-- datetime header
    2026-07-06 09:30:00,...,2848644       <-- data rows

IB → JULI happens via this loader (for historical/backtest) or via the
live feed wrapper (for paper/live trading). The brain itself never
touches a file path — it only sees numpy arrays.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Expected column order in data rows (after headers are skipped)
# CSV: datetime, close, high, low, open, volume
_COL_CLOSE = 1
_COL_HIGH = 2
_COL_LOW = 3
_COL_OPEN = 4
_COL_VOLUME = 5


def _is_header_row(row: list[str]) -> bool:
    """Detect header/metadata rows to skip."""
    if len(row) < 2:
        return True
    first = row[0].strip().lower()
    if first in ("datetime", "date", "time", "ticker", "price"):
        return True
    # Check if first data field is non-numeric (not a timestamp)
    try:
        # Timestamps start with a digit or '2'
        if first and first[0].isdigit():
            return False
    except (IndexError, AttributeError):
        pass
    # If close column isn't parseable as float, skip
    try:
        float(row[_COL_CLOSE])
        return False
    except (ValueError, IndexError):
        return True


def _parse_csv_row(row: list[str]) -> tuple | None:
    """Parse a data row into (datetime, open, high, low, close, volume)."""
    if len(row) < 6:
        return None
    try:
        return (row[0], float(row[_COL_OPEN]), float(row[_COL_HIGH]),
                float(row[_COL_LOW]), float(row[_COL_CLOSE]),
                float(row[_COL_VOLUME]))
    except (ValueError, IndexError):
        return None


def _read_csv(path) -> dict[str, list]:
    """Read CSV, skipping header rows. Returns column lists."""
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
            dt.append(parsed[0])
            o.append(parsed[1])
            h.append(parsed[2])
            l.append(parsed[3])
            c.append(parsed[4])
            v.append(parsed[5])
    return {"datetime": dt, "open": o, "high": h, "low": l, "close": c, "volume": v}


def load_ohlcv(path: str | Path) -> dict[str, NDArray]:
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
    close: NDArray, high: NDArray, low: NDArray, volume: NDArray
) -> NDArray:
    """Estimate buy-side volume from OHLCV.

    buy_fraction = clamp((close - low) / (high - low + 1e-12), 0, 1)
    This is a standard retail proxy: if close is near high, most
    volume was buying; if near low, most was selling.
    """
    rng = np.maximum(high - low, 1e-12)
    buy_frac = np.clip((close - low) / rng, 0.0, 1.0)
    return volume * buy_frac


def estimate_bid_ask(
    volume: NDArray, buy_volume: NDArray
) -> tuple[NDArray, NDArray]:
    """Estimate aggregate bid/ask sizes from buy/sell volume.

    Returns (bid_sizes, ask_sizes) as arrays for the orderbook_imbalance
    indicator. Uses the last bar's cumulative buy/sell split.
    """
    sell_volume = np.maximum(volume - buy_volume, 0)
    total = np.maximum(volume, 1e-12)
    bid_size = float(np.sum(buy_volume[-10:]) / 10.0 / np.mean(total[-10:]))
    ask_size = float(np.sum(sell_volume[-10:]) / 10.0 / np.mean(total[-10:]))
    return np.array([bid_size]), np.array([ask_size])
