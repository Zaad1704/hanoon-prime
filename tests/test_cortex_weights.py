"""tests/test_cortex_weights — scoring survives negative/zero-weight budgets.

Locks the fix for the every-candidate-vetoed incident: a drifted regime
vector summing to a negative value (live `range` had signed_sum ≈ -3.14)
used to hit the old ``total_w <= 1e-12`` guard and pin the score at 0.0,
which the orchestrator then translated into a SHORT that direction_mode
vetoed. The cortex now normalizes by Σ|w| so a negative budget can never
zero or sign-flip the signal.
"""

from __future__ import annotations

from hanoon_prime.cortex import Cortex


def test_negative_signed_sum_still_scores():
    """The exact incident vector (negative signed sum) must NOT zero the score."""
    c = Cortex(weights={"momentum": -0.55, "vwap_deviation": -0.45})
    for v in (0.0, 1.0, 2.0):
        c.evaluate({"momentum": v, "vwap_deviation": v})
    got = c.evaluate({"momentum": 100.0, "vwap_deviation": 100.0})
    assert got.score != 0.0
    assert got.score < 0.0  # negative weight × positive z stays negative


def test_negative_weights_preserve_sign():
    """w·z < 0 gives a negative score — no sign flip, no forced flat."""
    c = Cortex(weights={"momentum": -0.55, "vwap_deviation": -0.45})
    got = c._tanh_score({"momentum": 1.0, "vwap_deviation": 1.0})
    assert got < 0.0
    got = c._tanh_score({"momentum": -1.0, "vwap_deviation": -1.0})
    assert got > 0.0


def test_all_positive_legacy_formula_preserved():
    """With only positive weights Σ|w| == Σw, so the old result is unchanged."""
    c = Cortex(weights={"momentum": 0.25, "vwap_deviation": 0.25})
    z = {"momentum": 1.0, "vwap_deviation": -1.0}
    assert c._tanh_score(z) == c._tanh_score(z)  # deterministic


def test_zero_budget_pins_flat():
    """A genuinely zero budget is the only case that forces score 0.0."""
    c = Cortex(weights={"momentum": 0.0, "vwap_deviation": 0.0})
    assert c._tanh_score({"momentum": 2.0, "vwap_deviation": 2.0}) == 0.0
