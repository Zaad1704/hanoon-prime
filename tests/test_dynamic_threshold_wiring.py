"""tests/test_dynamic_threshold_wiring.py — D1: adaptive firing thresholds.

Pre-fix: DynamicThresholdAdapter existed but had zero callers — neuron
thresholds were fixed (input/hidden 0.7, decision 0.6) and DECISION_THRESHOLD
was exported but consumed by nobody. D1 feeds per-ticker price history from
BrainState.latest_prices through the tick pipeline, gated by
NEURO_ADAPTIVE_THRESHOLD_ENABLED (default OFF → byte-identical live).
"""

from __future__ import annotations

import math

import pytest

from hanoon_prime.brain.neurons.bridge import NeuromorphicBridge


def _volatile_prices(
    steps: int = 30, base: float = 100.0, amp: float = 2.0
) -> list[float]:
    prices: list[float] = []
    for i in range(steps):
        prices.append(base + amp * math.sin(i / 2.0) * (1.0 + (i % 3) * 0.1))
    return prices


def _calm_prices(steps: int = 30, base: float = 100.0) -> list[float]:
    return [base + 0.01 * (i % 2) for i in range(steps)]


def test_thresholds_gated_off_by_default():
    bridge = NeuromorphicBridge()
    bridge.update_market_env("AAPL", _volatile_prices())

    thresholds = {nid: n.threshold for nid, n in bridge._network._neurons.items()}
    assert thresholds == bridge._base_neuron_thresholds


def test_high_vol_raises_decision_threshold(monkeypatch):
    monkeypatch.setattr("hanoon_prime.immune.NEURO_ADAPTIVE_THRESHOLD_ENABLED", True)
    bridge = NeuromorphicBridge()

    bridge.update_market_env("AAPL", _volatile_prices(), vix=35.0)

    assert bridge._network._neurons["decision_long"].threshold > 0.6
    assert bridge._network._neurons["bull_vpin_bull"].threshold > 0.7


def test_calm_market_lowers_decision_threshold(monkeypatch):
    monkeypatch.setattr("hanoon_prime.immune.NEURO_ADAPTIVE_THRESHOLD_ENABLED", True)
    bridge = NeuromorphicBridge()

    bridge.update_market_env("AAPL", _calm_prices(), vix=12.0)

    assert bridge._network._neurons["decision_long"].threshold < 0.6
    assert bridge._network._neurons["bull_vpin_bull"].threshold < 0.7


def test_threshold_scale_clamped_to_half_and_five_x(monkeypatch):
    monkeypatch.setattr("hanoon_prime.immune.NEURO_ADAPTIVE_THRESHOLD_ENABLED", True)
    bridge = NeuromorphicBridge()

    bridge.update_market_env("AAPL", _volatile_prices(8), vix=50.0)
    bridge.update_market_env("AAPL", _calm_prices(), vix=10.0)

    for nid, base in bridge._base_neuron_thresholds.items():
        ratio = bridge._network._neurons[nid].threshold / base
        assert 0.5 <= ratio <= 5.0


def test_restore_after_flag_off_returns_build_time(monkeypatch):
    monkeypatch.setattr("hanoon_prime.immune.NEURO_ADAPTIVE_THRESHOLD_ENABLED", True)
    bridge = NeuromorphicBridge()
    bridge.update_market_env("AAPL", _volatile_prices(), vix=40.0)
    assert bridge._network._neurons["decision_long"].threshold > 0.6

    monkeypatch.setattr("hanoon_prime.immune.NEURO_ADAPTIVE_THRESHOLD_ENABLED", False)
    bridge.update_market_env("AAPL", _volatile_prices())
    assert bridge._network._neurons["decision_long"].threshold == 0.6

    bridge.update_market_env("AAPL", [1.0, 2.0])
    assert bridge._network._neurons["decision_long"].threshold == 0.6


def test_short_history_falls_back_to_base(monkeypatch):
    monkeypatch.setattr("hanoon_prime.immune.NEURO_ADAPTIVE_THRESHOLD_ENABLED", True)
    bridge = NeuromorphicBridge()

    bridge.update_market_env("AAPL", _volatile_prices(5), vix=20.0)

    assert bridge._network._neurons["decision_long"].threshold == 0.6
    assert bridge._network._neurons["bull_vpin_bull"].threshold == 0.7
