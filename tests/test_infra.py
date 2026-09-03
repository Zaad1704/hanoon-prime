"""tests/test_infra.py — infrastructure, CLI, and error-handling coverage.

Covers backtest.py CLI paths, calibrate.py helpers, validator.py
permutation edge tests, and error handling for corrupt/missing data.
Runs on FAST tickers only for speed.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hanoon_prime.backtest import (
    _discover_tickers,
    _print_results,
    _try_backtest_ticker,
    backtest_ticker,
    run_backtest,
)
from hanoon_prime.calibrate import (
    Calibration,
    _all_tickers,
    _check_profitability,
    calibrate,
)
from hanoon_prime.eyes import load_ohlcv
from hanoon_prime.immune import EDGE_LOOKBACK
from hanoon_prime.validator import (
    calibrate_weights,
    evaluate_indicator_edge,
    pooled_signals,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market_data"
FAST_TICKERS = ["AAPL", "MSFT", "SPY", "TSLA", "NVDA"]


def _skip_if_no_data():
    if not (DATA_DIR / "AAPL_1min.csv").exists():
        pytest.skip("No data files available")


@pytest.fixture
def aapl_data():
    _skip_if_no_data()
    return load_ohlcv(DATA_DIR / "AAPL_1min.csv")


# ── backtest.py ──────────────────────────────────────────────────────


def test_discover_tickers_finds_all():
    if not DATA_DIR.exists():
        pytest.skip("No data directory")
    result = _discover_tickers(DATA_DIR)
    assert isinstance(result, list)
    assert len(result) > 0
    # Should include FAST tickers
    for t in FAST_TICKERS:
        assert t in result


def test_backtest_ticker_with_output(aapl_data, tmp_path):
    _skip_if_no_data()
    metrics = backtest_ticker(
        "AAPL",
        aapl_data["close"],
        aapl_data["high"],
        aapl_data["low"],
        aapl_data["volume"],
        window=EDGE_LOOKBACK,
        output_dir=tmp_path,
    )
    json_path = tmp_path / "AAPL.json"
    assert json_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["ev_per_trade"] == metrics["ev_per_trade"]
    assert loaded["status"] == metrics["status"]


def test_backtest_ticker_no_output(aapl_data):
    _skip_if_no_data()
    metrics = backtest_ticker(
        "AAPL",
        aapl_data["close"],
        aapl_data["high"],
        aapl_data["low"],
        aapl_data["volume"],
        window=EDGE_LOOKBACK,
    )
    assert "ev_per_trade" in metrics
    assert "total_trades" in metrics


def test_try_backtest_ticker_missing_file(tmp_path):
    errors: list[str] = []
    result = _try_backtest_ticker("NONEXIST", tmp_path, EDGE_LOOKBACK, None, errors)
    assert result is None
    assert any("no data file" in e for e in errors)


def test_try_backtest_ticker_corrupt_file(tmp_path):
    csv_path = tmp_path / "BAD_1min.csv"
    csv_path.write_text("garbage,not,valid\n" * 5)
    errors: list[str] = []
    result = _try_backtest_ticker("BAD", tmp_path, EDGE_LOOKBACK, None, errors)
    assert result is None
    assert any("corrupt" in e for e in errors)


def test_try_backtest_ticker_insufficient_bars(tmp_path):
    csv_path = tmp_path / "SHORT_1min.csv"
    csv_path.write_text("date,open,high,low,close,volume\n")
    for i in range(EDGE_LOOKBACK + 5):
        csv_path.write_text(
            csv_path.read_text() + f"2020-01-01 {i:02d}:00:00,{100.0+i},{100.0+i+0.1},"
            f"{100.0+i},{100.0+i},{1000+i}\n"
        )
    errors: list[str] = []
    result = _try_backtest_ticker("SHORT", tmp_path, EDGE_LOOKBACK, None, errors)
    assert result is None
    assert any("insufficient" in e for e in errors)


def test_print_results_all_profitable():
    results = {
        "AAPL": {"ev_per_trade": 0.5, "total_trades": 10},
        "MSFT": {"ev_per_trade": 0.3, "total_trades": 5},
    }
    code = _print_results(results)
    assert code == 0


def test_print_results_has_unprofitable():
    results = {
        "AAPL": {"ev_per_trade": -0.5, "total_trades": 10},
        "MSFT": {"ev_per_trade": 0.3, "total_trades": 5},
    }
    code = _print_results(results)
    assert code == 1


def test_print_results_empty():
    code = _print_results({})
    assert code == 0


def test_run_backtest_with_errors(tmp_path):
    results, errors = run_backtest(
        ["NONEXIST"],
        DATA_DIR,
        output_dir=None,
    )
    assert results == {}
    assert len(errors) > 0


def test_run_backtest_fast_tickers():
    _skip_if_no_data()
    available = [t for t in FAST_TICKERS if (DATA_DIR / f"{t}_1min.csv").exists()]
    if len(available) < 2:
        pytest.skip("Not enough data files")
    results, _ = run_backtest(available, DATA_DIR, output_dir=None)
    assert len(results) == len(available)
    for ticker in available:
        assert ticker in results
        assert "ev_per_trade" in results[ticker]
        assert (
            results[ticker]["ev_per_trade"] > 0
        ), f"{ticker} not profitable: {results[ticker]}"


# ── calibrate.py ─────────────────────────────────────────────────────


def test_calibration_to_dict():
    cal = Calibration(
        weights={"momentum": 0.5},
        indicator_corrs={"momentum": 0.03},
        indicator_pvalues={"momentum": 0.01},
        indicator_significant={"momentum": True},
        confidence=0.75,
        n_indicators_significant=1,
    )
    d = cal.to_dict()
    assert d["weights"] == {"momentum": 0.5}
    assert d["confidence"] == 0.75
    assert d["n_indicators_significant"] == 1
    assert "indicator_corrs" in d
    assert "indicator_pvalues" in d


def test_calibration_defaults():
    cal = Calibration()
    assert cal.weights == {}
    assert cal.confidence == 0.5
    assert cal.n_indicators_significant == 0


def test_all_tickers_finds_valid():
    _skip_if_no_data()
    result = _all_tickers(DATA_DIR)
    assert isinstance(result, list)
    assert len(result) > 0
    for t in FAST_TICKERS:
        assert t in result


def test_all_tickers_excludes_corrupt(tmp_path):
    # Write a valid CSV
    good = tmp_path / "GOOD_1min.csv"
    good.write_text(
        "date,open,high,low,close,volume,buy_volume\n"
        + "".join(
            f"2020-01-01 {i:02d}:00:00,{100},{101},{99},{100},{1000},{500}\n"
            for i in range(100)
        )
    )
    # Write a corrupt CSV
    bad = tmp_path / "BAD_1min.csv"
    bad.write_text("garbage")
    result = _all_tickers(tmp_path)
    assert "GOOD" in result
    assert "BAD" not in result


def test_check_profitability(aapl_data):
    _skip_if_no_data()
    profitable, total = _check_profitability(["AAPL"], DATA_DIR)
    assert total >= 1
    assert profitable >= 0
    assert profitable <= total


def test_check_profitability_skips_missing(tmp_path):
    profitable, total = _check_profitability(["NOPE"], tmp_path)
    assert total == 0
    assert profitable == 0


def test_calibrate_with_fast_tickers():
    _skip_if_no_data()
    available = [t for t in FAST_TICKERS if (DATA_DIR / f"{t}_1min.csv").exists()]
    if len(available) < 2:
        pytest.skip("Not enough data")
    cal = calibrate(available, DATA_DIR)
    assert isinstance(cal.weights, dict)
    assert len(cal.weights) == 5
    assert 0.0 <= cal.confidence <= 1.0
    assert cal.n_indicators_significant >= 0


def test_calibrate_empty_tickers(tmp_path):
    """With no tickers and empty data dir, calibrate returns defaults."""
    cal = calibrate([], tmp_path)
    assert isinstance(cal.weights, dict)
    assert len(cal.weights) == 5  # default weights from calibrate_weights
    assert cal.confidence == 0.0  # 0 indicators + 0 profitable → 0.0


@patch("sys.argv", ["calibrate", "--data-dir", str(DATA_DIR), "--tickers", "AAPL"])
def test_calibrate_main_cli():
    _skip_if_no_data()
    from hanoon_prime.calibrate import main as cal_main

    code = cal_main()
    assert code in (0, 1)  # depends on confidence threshold


# ── validator.py ─────────────────────────────────────────────────────


def test_evaluate_indicator_edge_fast():
    _skip_if_no_data()
    available = [t for t in FAST_TICKERS if (DATA_DIR / f"{t}_1min.csv").exists()]
    if len(available) < 2:
        pytest.skip("Not enough data")
    result = evaluate_indicator_edge(available, DATA_DIR, n_perm=50)
    assert isinstance(result, dict)
    assert len(result) == 5
    for name in [
        "vpin",
        "orderbook_imbalance",
        "institutional_flow",
        "momentum",
        "vwap_deviation",
    ]:
        assert name in result
        assert "pvalue" in result[name]
        assert "corr" in result[name]


def test_calibrate_weights_from_edge():
    edge = {
        "vpin": {"corr": 0.03, "significant": True, "pvalue": 0.01, "n_samples": 300},
        "orderbook_imbalance": {
            "corr": 0.001,
            "significant": False,
            "pvalue": 0.5,
            "n_samples": 300,
        },
        "institutional_flow": {
            "corr": 0.02,
            "significant": True,
            "pvalue": 0.02,
            "n_samples": 300,
        },
        "momentum": {
            "corr": 0.015,
            "significant": True,
            "pvalue": 0.03,
            "n_samples": 300,
        },
        "vwap_deviation": {
            "corr": 0.012,
            "significant": True,
            "pvalue": 0.04,
            "n_samples": 300,
        },
    }
    weights = calibrate_weights(edge)
    assert len(weights) == 5
    assert sum(abs(v) for v in weights.values()) > 0
    # Significant indicators should have non-floor weights
    assert abs(weights["vpin"]) > 0.01


def test_calibrate_weights_all_insufficient():
    edge = {
        "vpin": {"corr": 0.0, "significant": False, "pvalue": 0.5, "n_samples": 50},
        "orderbook_imbalance": {
            "corr": 0.0,
            "significant": False,
            "pvalue": 0.5,
            "n_samples": 50,
        },
        "institutional_flow": {
            "corr": 0.0,
            "significant": False,
            "pvalue": 0.5,
            "n_samples": 50,
        },
        "momentum": {"corr": 0.0, "significant": False, "pvalue": 0.5, "n_samples": 50},
        "vwap_deviation": {
            "corr": 0.0,
            "significant": False,
            "pvalue": 0.5,
            "n_samples": 50,
        },
    }
    weights = calibrate_weights(edge)
    assert len(weights) == 5
    expected = 0.20  # equal weights when all are floor
    for v in weights.values():
        assert abs(v - expected) < 0.001


def test_pooled_signals_returns_all_indicators():
    _skip_if_no_data()
    available = [t for t in FAST_TICKERS if (DATA_DIR / f"{t}_1min.csv").exists()]
    if not available:
        pytest.skip("No data")
    pooled, returns = pooled_signals(available[:3], DATA_DIR)
    assert isinstance(pooled, dict)
    assert len(pooled) == 5
    assert len(returns) > 0
    assert len(returns.shape) == 1


def test_pooled_signals_skips_missing(tmp_path):
    """pooled_signals should skip tickers that don't exist."""
    pooled, returns = pooled_signals(["NOPE"], tmp_path)
    assert isinstance(pooled, dict)
    assert len(pooled) == 5
    assert len(returns) == 0
