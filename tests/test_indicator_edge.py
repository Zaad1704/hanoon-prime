"""tests/test_indicator_edge.py — prove each indicator has directional edge.

R4 gate: every indicator must demonstrably predict next-bar returns.

This test is SELF-SUFFICIENT — it uses a permutation test (500 shuffles)
on POOLED data from ALL available tickers. No hardcoded thresholds. The
system validates itself.

An indicator "has edge" if the permutation test rejects the null
hypothesis (p < 0.05) on the pooled dataset.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hanoon_prime._edge_eval import evaluate_indicator_pooled
from hanoon_prime.constants import EDGE_LOOKBACK

# Auto-discover all tickers in the data directory for maximum statistical power
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market_data"
EDGE_TICKERS = sorted(
    [f.stem.replace("_1min", "") for f in DATA_DIR.glob("*_1min.csv")]
) if DATA_DIR.exists() else ["AAPL", "MSFT", "SPY", "TSLA", "NVDA"]


class TestIndicatorEdge:
    """Each indicator must show statistically significant edge (p < 0.05)."""

    @pytest.mark.backtest
    def test_vpin_has_edge(self):
        _assert_pooled_edge("vpin")

    @pytest.mark.backtest
    def test_orderbook_imbalance_has_edge(self):
        _assert_pooled_edge("orderbook_imbalance")

    @pytest.mark.backtest
    def test_institutional_flow_has_edge(self):
        _assert_pooled_edge("institutional_flow")

    @pytest.mark.backtest
    def test_momentum_has_edge(self):
        _assert_pooled_edge("momentum")

    @pytest.mark.backtest
    def test_vwap_deviation_has_edge(self):
        _assert_pooled_edge("vwap_deviation")


def _assert_pooled_edge(indicator_name: str):
    """Auto-evaluate an indicator via pooled permutation test.

    Pools signals+returns from all EDGE_TICKERS, runs a permutation test
    (500 shuffles) to get a p-value. Passes if p < 0.05.
    """
    results = evaluate_indicator_pooled(EDGE_TICKERS, DATA_DIR, n_perm=500)
    info = results[indicator_name]
    sig_str = "SIGNIFICANT" if info["significant"] else "NOT significant"
    print(f"\n  {indicator_name}: |corr|={abs(info['corr']):.4f}  "
          f"p={info['pvalue']:.4f}  n={info['n_samples']}  {sig_str}")
    assert info["significant"], (
        f"R4 VIOLATION: {indicator_name} shows NO statistically significant "
        f"edge (p={info['pvalue']:.4f} ≥ 0.05, |corr|={info['corr']:.4f}). "
        f"Cut this indicator, don't dampen it."
    )
