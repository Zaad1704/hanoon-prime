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


@pytest.fixture(autouse=True)
def _hermetic_learning_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the strategy organs' persisted state at a per-test directory.

    Without this, one test's bandit/meta/regime updates leak through the
    shared runtime/*.json files into later tests (nondeterministic
    horizon overrides). Each test gets fresh learning state. The leaf
    modules bind their path constants at import time, so patch the leaf
    module globals directly.
    """
    monkeypatch.setattr(
        "hanoon_prime.brain.meta_label.META_FILE", tmp_path / "meta.json"
    )
    monkeypatch.setattr(
        "hanoon_prime.brain.horizon_bandit.BANDIT_FILE", tmp_path / "bandit.json"
    )
    monkeypatch.setattr(
        "hanoon_prime.brain.regime_weights.REGIME_FILE", tmp_path / "regime.json"
    )
    # Redirect JuliMemory (the brain's persistent learning state) to the
    # per-test directory too — without this, trade-close learning writes
    # SMOKE/test episodes into the production runtime/juli_state.json.
    monkeypatch.setenv("HANOO_MEMORY_FILE", str(tmp_path / "juli_state.json"))


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
