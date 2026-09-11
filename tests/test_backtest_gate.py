"""tests/test_backtest_gate.py — Phase 1: non-vacuous R2 gate + real fill model.

Guards three Phase-1 properties:
  1. The profitability gate must FAIL when no metrics exist (no vacuous pass).
  2. The gate must FAIL when a ticker produced zero trades.
  3. Backtest fills must model adverse slippage (entry + exit) using
     ``immune.SLIPPAGE_BPS`` so EV is not overstated.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hanoon_prime.hands import _adverse_fill, simulate_ticker
from hanoon_prime.immune import EDGE_LOOKBACK, SLIPPAGE_BPS
from hanoon_prime.types import BarSeries

GATE = Path(__file__).resolve().parent.parent / "scripts" / "check_profit_gate.py"
FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "SPY_1min.csv"


def _run_gate(metrics_dir: Path) -> int:
    return subprocess.run(
        [sys.executable, str(GATE), str(metrics_dir)],
        capture_output=True,
        text=True,
    ).returncode


def test_gate_fails_on_missing_metrics(tmp_path: Path) -> None:
    """An empty metrics dir must FAIL the gate (previously returned 0)."""
    assert _run_gate(tmp_path) == 1


def test_gate_fails_on_zero_trades_metric(tmp_path: Path) -> None:
    """A ticker that produced no trades is not 'profitable' by default."""
    (tmp_path / "AAPL.json").write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "ev_per_trade": 0.0,
                "expectancy": 0.0,
                "total_trades": 0,
            }
        )
    )
    assert _run_gate(tmp_path) == 1


def test_gate_passes_on_positive_ev_with_trades(tmp_path: Path) -> None:
    """Positive EV with real trades is the success path."""
    (tmp_path / "SPY.json").write_text(
        json.dumps(
            {
                "ticker": "SPY",
                "ev_per_trade": 0.4,
                "expectancy": 0.4,
                "total_trades": 20,
            }
        )
    )
    assert _run_gate(tmp_path) == 0


def test_fixtures_dir_is_committed_backtest_source(tmp_path: Path) -> None:
    """CI gate must read committed fixtures, not the gitignored symlink."""
    import subprocess

    fixtures = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    old = Path(__file__).resolve().parent.parent / "data" / "market_data"
    assert fixtures.exists(), "data/fixtures must exist for CI backtest"
    fixture_file = fixtures / "SPY_1min.csv"
    assert fixture_file.exists(), "SPY fixture must be present"
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(fixture_file)],
        capture_output=True,
    ).returncode
    assert ignored == 1, "fixtures must NOT be gitignored"
    assert old.is_symlink() or old.exists(), "legacy data dir expected"


def test_slippage_pessimizes_entry_and_exit() -> None:
    """A long entry must fill above signal price; a short entry below it."""
    px = 100.0
    assert _adverse_fill(px, +1) > px  # long buys higher
    assert _adverse_fill(px, -1) < px  # short sells lower
    expected_off = px * SLIPPAGE_BPS / 10000.0
    assert abs(_adverse_fill(px, +1) - (px + expected_off)) < 1e-9
    assert abs(_adverse_fill(px, -1) - (px - expected_off)) < 1e-9


def _load_spy() -> BarSeries | None:
    from hanoon_prime.eyes import load_ohlcv

    if not FIXTURES.exists():
        pytest.skip("No SPY fixture data")
    data = load_ohlcv(FIXTURES)
    return BarSeries(data["close"], data["high"], data["low"], data["volume"])


def test_slippage_does_not_improve_expectancy() -> None:
    """Wiring check: enabling residual slippage must not raise realized pnl."""
    import hanoon_prime.hands as hands

    bars = _load_spy()
    if bars is None:
        pytest.skip("no data")

    # Default: SLIPPAGE_BPS already applied via module global.
    trades_base, _ = simulate_ticker("SPY", bars, EDGE_LOOKBACK)
    base_ev = sum(t.pnl_pct for t in trades_base)

    # Zero-slippage twin run (no allow-comment needed: not a denylisted symbol).
    orig = hands.SLIPPAGE_BPS
    hands.SLIPPAGE_BPS = 0.0
    try:
        trades_zero, _ = simulate_ticker("SPY", bars, EDGE_LOOKBACK)
    finally:
        hands.SLIPPAGE_BPS = orig
    zero_ev = sum(t.pnl_pct for t in trades_zero)

    # Trade sets can diverge (stop/target move with fill), so allow tiny slack;
    # the point is slippage must not systematically ADD expectancy.
    assert base_ev <= zero_ev + 0.5, (
        f"slippage must not improve expectancy (with={base_ev:.2f}, "
        f"without={zero_ev:.2f})"
    )
