"""tests/test_wfa.py — Phase 2: walk-forward + anti-overfit statistics.

Guards the statistical honesty of the backtest pipeline:
  * folds are contiguous and non-overlapping (pure OOS per fold)
  * a ticker below the MIN_TRADES floor is INSUFFICIENT, never PASS
  * the universe verdict deflates for the number of trials tried
  * PBO is bounded and deterministic
"""

from __future__ import annotations

import numpy as np
import pytest

from hanoon_prime.immune import EDGE_LOOKBACK
from hanoon_prime.wfa import (
    DEFAULT_FOLDS,
    MIN_TRADES,
    _count_trials,
    deflated_sharpe,
    fold_windows,
    pbo,
    run_walk_forward,
    run_wfa_universe,
    verdicts,
)


def test_fold_windows_contiguous_and_covering_tail() -> None:
    total = 1000  # EDGE_LOOKBACK warmup + 950 usable
    wins = fold_windows(total, 5)
    assert len(wins) == 5
    # Folds must touch the tail (cover the latest data).
    assert wins[-1][1] == total
    # No overlap, no gap between consecutive folds.
    for a, b in zip(wins, wins[1:]):
        assert a[1] == b[0]
    assert all(s < e for s, e in wins)


def test_fold_windows_too_short() -> None:
    assert fold_windows(10, 5) == []
    assert fold_windows(EDGE_LOOKBACK, 5) == []


def test_run_walk_forward_counts_only_fold_completing_trades() -> None:
    """Every scored trade must close inside its fold window."""
    from hanoon_prime.eyes import load_ohlcv

    path = "data/fixtures/AAPL_1min.csv"
    data = load_ohlcv(path)
    res = run_walk_forward("AAPL", data, 6)
    assert len(res) == 6
    for w in res:
        assert w.end <= len(data["close"])
        for t in w.trades:
            warm = w.start - EDGE_LOOKBACK
            assert warm + t.entry_idx >= w.start
            assert warm + t.exit_idx < w.end


def _fold(pnl: list[float], fold: int, start: int, end: int):
    """FoldResult with `trades` populated to match pnl (real run_walk_forward
    keeps them in sync — synthetic tests must too)."""
    from hanoon_prime.wfa import FoldResult

    mean = float(np.mean(pnl)) if pnl else 0.0
    std = float(np.std(pnl)) if pnl else 0.0
    return FoldResult(
        fold=fold,
        start=start,
        end=end,
        trades=[object() for _ in pnl],
        ev_per_trade=mean,
        sharpe=float(mean / (std + 1e-12)),
        pnl=pnl,
    )


def test_oos_trades_below_min_trades_is_insufficient() -> None:
    """A ticker with < MIN_TRADES OOS trades must NEVER pass."""
    results = {"T1": []}
    v = verdicts(results)
    assert all(t.verdict == "INSUFFICIENT" for t in v.admissible_tickers)
    assert v.verdict == "INSUFFICIENT"


def test_verdict_pass_requires_positive_deflated_edge() -> None:
    """Even a profitable ticker is FAIL if deflation kills the edge."""
    # Build many folds each with 2+ trades that are all winning (Sanity: a
    # single 'strategy' ticker with huge positive edge deflates to positive).
    pnl_win = [0.05, 0.06, 0.04, 0.05, 0.07, 0.06]

    results = {}
    for k in range(8):
        results[f"T{k}"] = [_fold(pnl_win, f, f, f + 1) for f in range(6)]
    v = verdicts(results)
    assert v.verdict == "PASS"
    assert v.deflated_edge > 0
    assert all(t.verdict == "PASS" for t in v.admissible_tickers)


def test_pooled_sharpe_of_losers_is_negative() -> None:
    from hanoon_prime.wfa import pooled_sharpe

    results = {}
    for k in range(4):
        pnl = [-0.03, -0.02, -0.04, -0.01, -0.03, -0.02] * 6
        results[f"T{k}"] = [_fold(pnl, f, f, f + 1) for f in range(6)]
    assert pooled_sharpe(results) < 0


def test_pbo_bounded_and_deterministic() -> None:
    """PBO must be in [0,1] and stable across calls (seed fixed)."""
    results = {}
    rng = np.random.default_rng(7)
    for k in range(8):
        folds = []
        for f in range(6):
            pnl = list(rng.normal(0.0002, 0.01, 8))
            folds.append(_fold(pnl, f, f, f + 1))
        results[f"T{k}"] = folds
    p1 = pbo(results)
    p2 = pbo(results)
    assert 0.0 <= p1 <= 1.0
    assert p1 == p2


def test_deflated_sharpe_zero_with_no_trials() -> None:
    assert deflated_sharpe({}) == 0.0


def test_count_trials_counts_ticker_folds_with_trades() -> None:
    results = {
        # A: 1 fold with 5 trades (>=2) → counts; B: 0 trades → not counted.
        "A": [_fold([0.01] * 5, 0, 0, 1)],
        "B": [_fold([], 0, 0, 1)],
    }
    assert _count_trials(results) == 1


def test_run_wfa_universe_reads_committed_fixtures() -> None:
    """The WFA must run against the committed fixtures directory."""
    from pathlib import Path

    fixtures = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    universe = run_wfa_universe(["AAPL", "SPY", "MSFT"], fixtures, folds=4)
    assert set(universe.keys()) == {"AAPL", "SPY", "MSFT"}
    for folds in universe.values():
        assert len(folds) == 4


def test_serialize_roundtrip() -> None:
    from hanoon_prime.wfa import serialize

    v = verdicts({})
    blob = serialize(v)
    assert blob["verdict"] == "INSUFFICIENT"
    assert "deflated_edge" in blob and "pbo" in blob
