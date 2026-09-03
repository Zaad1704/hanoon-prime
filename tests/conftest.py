"""tests/conftest.py — shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market_data"


@pytest.fixture
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture
def sample_tickers() -> list[str]:
    """A subset of tickers for fast backtesting."""
    candidates = ["AAPL", "MSFT", "SPY", "TSLA", "NVDA", "AMD", "GOOGL", "QQQ"]
    available = []
    for t in candidates:
        if (DATA_DIR / f"{t}_1min.csv").exists():
            available.append(t)
    return available if available else ["SPY"]


@pytest.fixture
def sample_data(sample_tickers):
    """Load data for the first available ticker."""
    from hanoon_prime.eyes import load_ohlcv

    ticker = sample_tickers[0]
    return ticker, load_ohlcv(DATA_DIR / f"{ticker}_1min.csv")
