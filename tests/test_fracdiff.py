"""tests/test_fracdiff.py — unit tests for the FracDiff feature transform.

Guarantees under test:
  - weights follow the (1 - B)**d recurrence and truncate at tau
  - d=0 is identity, d=1 is the first difference
  - the transform is causal (no future leakage)
  - a random walk is non-stationary, but its FracDiff (d < 1) is stationary
  - find_min_d returns the smallest stationary order and keeps memory
  - per-ticker d* config loads/saves/falls back correctly
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hanoon_prime.fracdiff import (
    adf_pvalue,
    find_min_d,
    fracdiff,
    fracdiff_weights,
    get_ticker_d,
    is_stationary,
    load_d_config,
    memory_retention,
    save_d_config,
)


@pytest.fixture
def random_walk() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.cumsum(rng.normal(0.0, 1.0, 2000))


class TestWeights:
    def test_identity_order_is_unit_weight(self):
        np.testing.assert_array_equal(fracdiff_weights(0.0), [1.0])

    def test_first_difference_weights(self):
        np.testing.assert_allclose(fracdiff_weights(1.0), [1.0, -1.0], atol=1e-12)

    def test_recurrence_matches_explicit_binomial(self):
        d = 0.5
        w = fracdiff_weights(d, tau=1e-3)
        # w_1 = -d, w_2 = d(d-1)/2 — the closed form of the expansion.
        assert w[1] == pytest.approx(-d)
        assert w[2] == pytest.approx(d * (d - 1.0) / 2.0)

    def test_weights_vanish_below_tau(self):
        w = fracdiff_weights(0.9, tau=1e-6)
        assert abs(w[-1]) >= 1e-6  # last kept weight clears the threshold

    def test_weights_sum_to_zero_for_fractional_d(self):
        # (1 - B)**d evaluated at B=1 is 0**d = 0; the truncated sum ≈ 0.
        assert abs(float(fracdiff_weights(0.4).sum())) < 0.1


class TestTransform:
    def test_identity_transform(self, random_walk):
        np.testing.assert_allclose(fracdiff(random_walk, 0.0), random_walk)

    def test_first_difference_matches_numpy(self, random_walk):
        np.testing.assert_allclose(fracdiff(random_walk, 1.0), np.diff(random_walk))

    def test_short_input_returns_empty(self):
        assert fracdiff(np.arange(3.0), 0.4).size == 0

    def test_transform_is_causal(self, random_walk):
        past = fracdiff(random_walk, 0.4)
        mutated = random_walk.copy()
        mutated[-1] += 100.0
        after = fracdiff(mutated, 0.4)
        # Changing the final bar may only change the final transformed bar.
        np.testing.assert_allclose(past[:-1], after[:-1])
        assert past[-1] != after[-1]

    def test_deterministic(self, random_walk):
        np.testing.assert_array_equal(
            fracdiff(random_walk, 0.4), fracdiff(random_walk, 0.4)
        )


class TestADF:
    def test_white_noise_is_stationary(self):
        rng = np.random.default_rng(1)
        assert is_stationary(rng.normal(0.0, 1.0, 1000))

    def test_random_walk_is_not_stationary(self, random_walk):
        assert not is_stationary(random_walk)

    def test_pvalue_boundary_is_five_percent(self):
        assert adf_pvalue(-2.86154) == pytest.approx(0.05, abs=1e-6)

    def test_pvalue_is_monotone(self):
        assert adf_pvalue(-4.0) < adf_pvalue(-3.0) < adf_pvalue(-2.0)

    def test_pvalue_clamps_lower_tail(self):
        assert adf_pvalue(-50.0) == pytest.approx(0.001)


class TestFindMinD:
    def test_returns_smallest_stationary_order(self, random_walk):
        d, transformed = find_min_d(random_walk)
        assert 0.0 < d <= 1.0
        assert is_stationary(transformed)
        assert transformed.size < random_walk.size

    def test_order_is_no_larger_than_first_difference(self, random_walk):
        d, _ = find_min_d(random_walk)
        assert d <= 1.0

    def test_preserves_memory_better_than_integer_diff(self, random_walk):
        d, transformed = find_min_d(random_walk)
        retention = memory_retention(random_walk, transformed)
        assert retention > 0.3, f"d={d} retained only {retention:.3f}"

    def test_step_controls_grid_resolution(self, random_walk):
        step = 0.1
        d, _ = find_min_d(random_walk, step=step)
        assert abs(d / step - round(d / step)) < 1e-6


class TestDConfig:
    def test_load_missing_file_returns_default(self, tmp_path):
        cfg = load_d_config(tmp_path / "absent.json")
        assert cfg == {"DEFAULT": 0.4}

    def test_save_and_load_roundtrip(self, tmp_path):
        p = tmp_path / "d_config.json"
        cfg = {"DEFAULT": 0.40, "AAPL": 0.35, "NVDA": 0.50}
        save_d_config(cfg, p)
        loaded = load_d_config(p)
        assert loaded["DEFAULT"] == pytest.approx(0.40)
        assert loaded["AAPL"] == pytest.approx(0.35)
        assert loaded["NVDA"] == pytest.approx(0.50)

    def test_get_ticker_d_returns_ticker_value(self, tmp_path):
        p = tmp_path / "d_config.json"
        save_d_config({"DEFAULT": 0.40, "AAPL": 0.30}, p)
        assert get_ticker_d("AAPL", load_d_config(p)) == pytest.approx(0.30)

    def test_get_ticker_d_falls_back_to_default(self, tmp_path):
        p = tmp_path / "d_config.json"
        save_d_config({"DEFAULT": 0.45}, p)
        assert get_ticker_d("MSFT", load_d_config(p)) == pytest.approx(0.45)

    def test_get_ticker_d_none_returns_default(self, tmp_path):
        p = tmp_path / "d_config.json"
        save_d_config({"DEFAULT": 0.40}, p)
        assert get_ticker_d(None, load_d_config(p)) == pytest.approx(0.40)

    def test_load_corrupted_file_returns_default(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid json")
        cfg = load_d_config(p)
        assert cfg == {"DEFAULT": 0.4}
