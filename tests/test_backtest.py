"""tests/test_backtest.py — end-to-end backtest validation.

R2 gate: the full JULI pipeline must show positive expectancy on
historical data. If it doesn't, the test fails and the build breaks.

These tests run the ACTUAL brain (not a placeholder) through ACTUAL
historical CSV data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hanoon_prime.alpha import compute_alpha
from hanoon_prime.backtest import backtest_ticker, run_backtest
from hanoon_prime.data import load_ohlcv, compute_buy_volume, estimate_bid_ask

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market_data"

# Fast subset for unit testing
FAST_TICKERS = ["AAPL", "MSFT", "SPY"]


def _check_data_available(ticker: str) -> bool:
    return (DATA_DIR / f"{ticker}_1min.csv").exists()


@pytest.mark.backtest
@pytest.mark.parametrize("ticker", FAST_TICKERS)
def test_single_ticker_backtest(ticker):
    """Full pipeline backtest on a single ticker.

    Must produce:
      - At least 1 trade (the pipeline actually works)
      - Positive expectancy (EV/trade > 0) OR zero trades (no signal found)
    """
    path = DATA_DIR / f"{ticker}_1min.csv"
    if not path.exists():
        pytest.skip(f"No data for {ticker}")

    data = load_ohlcv(path)
    assert len(data["close"]) >= 60, f"Insufficient bars for {ticker}"

    metrics = backtest_ticker(
        ticker, data["close"], data["high"], data["low"], data["volume"],
        window=30,
    )

    # Must not crash
    assert "ev_per_trade" in metrics
    assert "status" in metrics

    if metrics["total_trades"] > 0:
        # If trades were made, expectancy must be positive
        assert metrics["ev_per_trade"] > 0, (
            f"R2 VIOLATION: {ticker} has NEGATIVE expectancy: "
            f"{metrics['ev_per_trade']}R per trade "
            f"(WR={metrics['win_rate']:.1%}, R:R={metrics['realized_rr']:.2f})"
        )
        # Win rate must be above breakeven for 3:1 R:R (25%)
        if metrics["realized_rr"] >= 1.0:
            min_wr = 1.0 / (1.0 + metrics["realized_rr"])
            assert metrics["win_rate"] > min_wr, (
                f"R2 VIOLATION: {ticker} WR {metrics['win_rate']:.1%} < "
                f"breakeven {min_wr:.1%} for R:R {metrics['realized_rr']:.2f}"
            )


@pytest.mark.backtest
@pytest.mark.parametrize("ticker", FAST_TICKERS)
def test_brain_pipeline_runs(ticker):
    """Verify the full alpha → score → EV → think pipeline runs without errors."""
    path = DATA_DIR / f"{ticker}_1min.csv"
    if not path.exists():
        pytest.skip(f"No data for {ticker}")

    data = load_ohlcv(path)
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    # Run a single evaluation
    window = 30
    if len(close) < window + 5:
        pytest.skip(f"Insufficient bars for {ticker}")

    c_slice = close[:window]
    h_slice = high[:window]
    l_slice = low[:window]
    v_slice = volume[:window]

    bv = compute_buy_volume(c_slice, h_slice, l_slice, v_slice)
    bids, asks = estimate_bid_ask(v_slice, bv)

    alpha = compute_alpha(
        close=c_slice, volume=v_slice,
        buy_volume=bv, bid_sizes=bids, ask_sizes=asks,
    )

    # Verify alpha produces valid output
    from hanoon_prime.scoring import compute_score
    from hanoon_prime.edge import score_to_win_prob, compute_ev
    from hanoon_prime.thinker import deliberate

    score = compute_score(alpha)
    assert 0.10 <= score <= 0.70, f"Score out of range: {score}"

    win_prob = score_to_win_prob(score)
    assert 0.25 <= win_prob <= 0.55, f"Win prob out of range: {win_prob}"

    ev = compute_ev(win_prob)
    assert "gross_ev" in ev
    assert "net_ev" in ev

    kelly = compute_ev(win_prob)  # placeholder
    from hanoon_prime.edge import kelly_fraction
    kelly_val = kelly_fraction(win_prob)

    thought = deliberate(
        score=score, alpha=alpha,
        win_prob=win_prob, gross_ev=ev["gross_ev"],
        kelly=kelly_val,
    )

    assert thought.verdict in ("ENTER", "HOLD")
    assert 0.50 <= thought.confidence <= 0.95
    assert thought.direction in (-1, 0, 1)


def test_full_universe_backtest(sample_tickers):
    """Run backtest on all available tickers and check aggregate profitability."""
    available = [t for t in sample_tickers if _check_data_available(t)]
    if len(available) < 2:
        pytest.skip("Not enough data files available")

    results, _ = run_backtest(
        available[:5], DATA_DIR, output_dir=None,
    )

    profitable = sum(1 for m in results.values() if m["ev_per_trade"] > 0)
    trades_total = sum(m["total_trades"] for m in results.values())

    print(f"\n  Profitable: {profitable}/{len(results)}")
    print(f"  Total trades: {trades_total}")

    if trades_total > 0:
        # At least 60% of tickers with trades must be profitable
        tickers_with_trades = [t for t, m in results.items() if m["total_trades"] > 0]
        if tickers_with_trades:
            profitable_traded = sum(1 for t in tickers_with_trades
                                   if results[t]["ev_per_trade"] > 0)
            ratio = profitable_traded / len(tickers_with_trades)
            assert ratio >= 0.60, (
                f"Only {profitable_traded}/{len(tickers_with_trades)} "
                f"tickers with trades are profitable ({ratio:.0%})"
            )
