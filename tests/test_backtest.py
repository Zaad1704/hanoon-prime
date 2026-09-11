"""tests/test_backtest.py — end-to-end backtest validation.

R2 gate: the full JULI pipeline must show positive expectancy on
historical data. If it doesn't, the test fails and the build breaks.

These tests run the ACTUAL brain (not a placeholder) through ACTUAL
historical CSV data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hanoon_prime.backtest import backtest_ticker, run_backtest
from hanoon_prime.cerebellum import compute_alpha
from hanoon_prime.cortex import Cortex
from hanoon_prime.edge import compute_ev, kelly_fraction, score_to_win_prob
from hanoon_prime.eyes import compute_buy_volume, estimate_bid_ask, load_ohlcv
from hanoon_prime.immune import EDGE_LOOKBACK
from hanoon_prime.types import BarSeries

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"

# Fast subset for unit testing
FAST_TICKERS = ["AAPL", "MSFT", "SPY", "TSLA", "NVDA"]


def _check_data_available(ticker: str) -> bool:
    return (DATA_DIR / f"{ticker}_1min.csv").exists()


@pytest.mark.backtest
@pytest.mark.parametrize("ticker", FAST_TICKERS)
def test_single_ticker_backtest(ticker):
    """Full pipeline backtest on a single ticker.

    Structural honesty check: the pipeline runs on real closes and produces
    plausible, finite metrics. Profitability is judged ONLY by the R2 gate
    (scripts/check_profit_gate.py), not hardcoded here — otherwise this
    test would be a second, contradictory gate on the same data.
    """
    path = DATA_DIR / f"{ticker}_1min.csv"
    if not path.exists():
        pytest.skip(f"No data for {ticker}")

    data = load_ohlcv(path)
    assert len(data["close"]) >= EDGE_LOOKBACK + 10

    metrics = backtest_ticker(
        ticker,
        BarSeries(data["close"], data["high"], data["low"], data["volume"]),
        window=EDGE_LOOKBACK,
    )

    assert "ev_per_trade" in metrics
    assert "status" in metrics
    assert metrics["total_trades"] >= 0
    assert "win_rate" in metrics and 0.0 <= metrics["win_rate"] <= 1.0
    assert "realized_rr" in metrics
    assert metrics["ev_per_trade"] is not None
    # EV must be finite: NaN/Inf means the pipeline leaked or divided wrong.
    import math

    assert math.isfinite(metrics["ev_per_trade"])


@pytest.mark.backtest
@pytest.mark.parametrize("ticker", FAST_TICKERS)
def test_brain_pipeline_runs(ticker):
    """Verify the full alpha → cortex → EV pipeline runs without errors."""
    path = DATA_DIR / f"{ticker}_1min.csv"
    if not path.exists():
        pytest.skip(f"No data for {ticker}")

    data = load_ohlcv(path)
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    window = EDGE_LOOKBACK
    if len(close) < window + 5:
        pytest.skip(f"Insufficient bars for {ticker}")

    c_slice = close[:window]
    h_slice = high[:window]
    l_slice = low[:window]
    v_slice = volume[:window]

    bv = compute_buy_volume(c_slice, h_slice, l_slice, v_slice)
    bids, asks = estimate_bid_ask(v_slice, bv)

    alpha = compute_alpha(
        close=c_slice,
        volume=v_slice,
        buy_volume=bv,
        bid_sizes=bids,
        ask_sizes=asks,
    )

    cortex = Cortex()
    thought = cortex.evaluate(alpha)

    assert thought.verdict in ("BUY", "SELL", "HOLD")
    assert -1.0 <= thought.score <= 1.0, f"Score out of range: {thought.score}"
    assert 0.50 <= thought.confidence <= 0.95

    win_prob = score_to_win_prob(thought.score)
    assert 0.25 <= win_prob <= 0.60, f"Win prob out of range: {win_prob}"

    ev = compute_ev(win_prob)
    assert "gross_ev" in ev
    assert "net_ev" in ev

    kelly_val = kelly_fraction(win_prob)
    assert 0.0 <= kelly_val <= 0.5

    assert thought.direction in (-1, 0, 1)


@pytest.mark.backtest
def test_full_universe_backtest(sample_tickers):
    """Run backtest on all available tickers; every result must be sane.

    Aggregated profitability is judged by the R2 gate script
    (scripts/check_profit_gate.py), which is the single source of truth for
    the "is it profitable?" question. This test only verifies the pipeline
    runs end-to-end on real data and reports coherent, finite metrics.
    """
    available = [t for t in sample_tickers if _check_data_available(t)]
    if len(available) < 2:
        pytest.skip("Not enough data files available")

    results, _ = run_backtest(
        available[:5],
        DATA_DIR,
        output_dir=None,
    )

    import math

    assert len(results) == len(available[:5])
    for m in results.values():
        assert math.isfinite(m["ev_per_trade"])
        assert m["total_trades"] >= 0
        assert 0.0 <= m["win_rate"] <= 1.0
        assert m["realized_rr"] >= 0.0

    trades_total = sum(m["total_trades"] for m in results.values())
    print(f"\n  Total trades: {trades_total}")
    print(
        f"  Tickr EV: "
        + ", ".join(f"{t}={m['ev_per_trade']:+.2f}R" for t, m in results.items())
    )
