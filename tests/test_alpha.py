"""tests/test_alpha.py — unit tests for each of the 5 indicators.

Every indicator must be:
  - Correctly normalized (returns a float in its expected range)
  - Directionally correct (positive signal → bullish, negative → bearish)
  - Bounded (never NaN, never infinity)
"""
from __future__ import annotations

import numpy as np
import pytest

from hanoon_prime.alpha import (
    INDICATOR_NAMES,
    compute_alpha,
    compute_institutional_flow,
    compute_momentum,
    compute_orderbook_imbalance,
    compute_vpin,
    compute_vwap_deviation,
)


class TestIndicatorContracts:
    """Verify each indicator has correct range and semantics."""

    def test_indicator_count(self):
        assert len(INDICATOR_NAMES) == 5

    def test_vpin_range(self):
        v = np.ones(20)
        bv = np.full(20, 15.0)  # more buy volume → positive
        vpin = compute_vpin(v, bv)
        assert -1.0 <= vpin <= 1.0, f"VPIN out of range: {vpin}"
        assert vpin > 0.0, f"VPIN should be positive with excess buy volume: {vpin}"

    def test_vpin_negative(self):
        v = np.ones(20)
        bv = np.zeros(20)  # all sell volume → negative
        vpin = compute_vpin(v, bv)
        assert -1.0 <= vpin <= 1.0
        assert vpin < 0.0, f"VPIN should be negative with pure sell volume: {vpin}"

    def test_vpin_zero_volume(self):
        v = np.zeros(10)
        bv = np.zeros(10)
        assert compute_vpin(v, bv) == 0.0

    def test_orderbook_imbalance_long(self):
        bids = np.array([1000.0, 500.0])
        asks = np.array([100.0, 50.0])
        imb = compute_orderbook_imbalance(bids, asks)
        assert -1.0 <= imb <= 1.0
        assert imb > 0.5  # heavy buy-side pressure

    def test_orderbook_imbalance_short(self):
        bids = np.array([100.0, 50.0])
        asks = np.array([1000.0, 500.0])
        imb = compute_orderbook_imbalance(bids, asks)
        assert imb < -0.5  # heavy sell-side pressure

    def test_orderbook_imbalanced_equal(self):
        bids = np.array([100.0, 100.0])
        asks = np.array([100.0, 100.0])
        assert compute_orderbook_imbalance(bids, asks) == pytest.approx(0.0)

    def test_institutional_flow_normal(self):
        v = np.full(20, 1000.0)
        flow = compute_institutional_flow(v, 1000.0)
        assert 0.0 <= flow <= 3.0
        assert flow == pytest.approx(1.0)  # normal volume

    def test_institutional_flow_spike(self):
        v = np.full(20, 1000.0)
        v[-5:] = 5000.0  # 5x volume spike
        flow = compute_institutional_flow(v, 1000.0)
        assert flow > 1.5  # detecting the spike

    def test_momentum_positive(self):
        c = np.linspace(100, 110, 10)
        mom = compute_momentum(c)
        assert -1.0 <= mom <= 1.0
        assert mom > 0

    def test_momentum_negative(self):
        c = np.linspace(110, 100, 10)
        mom = compute_momentum(c)
        assert mom < 0

    def test_vwap_deviation_positive(self):
        # Price above VWAP
        c = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        v = np.array([100, 100, 100, 100, 2000])  # high vol at end (high price)
        dev = compute_vwap_deviation(c, v)
        assert -1.0 <= dev <= 1.0
        assert dev > 0  # price above weighted avg

    def test_vwap_deviation_zero_volume(self):
        c = np.array([100.0, 101.0])
        v = np.array([0.0, 0.0])
        assert compute_vwap_deviation(c, v) == 0.0

    def test_all_indicators_bounded(self):
        """No indicator may return NaN or inf."""
        c = np.random.randn(30) + 100
        v = np.random.rand(30) * 1000 + 100
        bv = v * 0.5
        bids = np.array([100.0])
        asks = np.array([100.0])

        alpha = compute_alpha(
            close=c, volume=v, buy_volume=bv,
            bid_sizes=bids, ask_sizes=asks,
        )
        for name in INDICATOR_NAMES:
            val = alpha[name]
            assert not np.isnan(val), f"{name} returned NaN"
            assert not np.isinf(val), f"{name} returned inf"

    def test_compute_alpha_returns_all_5(self):
        c = np.linspace(100, 110, 10)
        v = np.ones(10) * 1000
        bv = np.ones(10) * 500
        bids = np.array([100.0])
        asks = np.array([100.0])

        alpha = compute_alpha(c, v, bv, bids, asks)
        # All 5 indicators must be present; volatility is a risk metric, not an indicator
        indicator_keys = set(INDICATOR_NAMES)
        assert indicator_keys.issubset(set(alpha.keys()))
        for name in INDICATOR_NAMES:
            assert isinstance(alpha[name], float)
        assert "volatility" in alpha  # risk metric for the thinker
