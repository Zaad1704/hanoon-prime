"""tests/test_contrarian — MoE mean-reversion override port (off-by-default).

Locks the faithful Prime translation of rebuild moe_meanrev_extreme:
overbought extreme → contrarian SHORT, oversold → contrarian LONG.
Disabled by default (CONTRARIAN_MODE_ENABLED=False) so the live halt-safe
trend verdict is unchanged unless explicitly opted in.
"""

from __future__ import annotations

import pytest

from hanoon_prime.contrarian import contrarian_direction
from hanoon_prime.immune import CONTRARIAN_MODE_ENABLED


def test_contrarian_off_by_default():
    """No override unless a human opts in — halt-safe by construction."""
    assert CONTRARIAN_MODE_ENABLED is False


def test_contrarian_direction_noop_when_disabled(monkeypatch):
    """Flag off → 0 regardless of z, so cortex keeps its trend verdict."""
    monkeypatch.setattr("hanoon_prime.contrarian.CONTRARIAN_MODE_ENABLED", False)
    assert contrarian_direction({"vwap_deviation": 3.0}) == 0
    assert contrarian_direction({"vwap_deviation": -3.0}) == 0


@pytest.mark.parametrize("z,expected", [(2.5, -1), (3.0, -1)])
def test_contrarian_shorts_on_overbought(monkeypatch, z, expected):
    """Overbought (positive extreme) → contrarian SHORT (-1), rebuild-faithful."""
    monkeypatch.setattr("hanoon_prime.contrarian.CONTRARIAN_MODE_ENABLED", True)
    assert contrarian_direction({"vwap_deviation": z}) == expected


@pytest.mark.parametrize("z,expected", [(-2.5, 1), (-3.0, 1)])
def test_contrarian_longs_on_oversold(monkeypatch, z, expected):
    """Oversold (negative extreme) → contrarian LONG (+1)."""
    monkeypatch.setattr("hanoon_prime.contrarian.CONTRARIAN_MODE_ENABLED", True)
    assert contrarian_direction({"vwap_deviation": z}) == expected


def test_contrarian_ignores_non_meanrev_indicator(monkeypatch):
    """Only the mean-reversion oscillator (vwap_deviation) triggers override —

    not trend indicators (e.g. momentum). This mirrors rebuild: meanrev_extreme
    is an oscillator/expert, separate from momentum trend neurons.
    """
    monkeypatch.setattr("hanoon_prime.contrarian.CONTRARIAN_MODE_ENABLED", True)
    assert contrarian_direction({"momentum": 3.0}) == 0


def test_contrarian_below_extreme_is_noop(monkeypatch):
    """Mild z (within CONTRARIAN_EXTREME_Z) → no override."""
    monkeypatch.setattr("hanoon_prime.contrarian.CONTRARIAN_MODE_ENABLED", True)
    assert contrarian_direction({"vwap_deviation": 1.9}) == 0
    assert contrarian_direction({"vwap_deviation": -1.9}) == 0


def test_cortex_contrarian_override_flips_overbought_trend(monkeypatch):
    """End-to-end: extreme overbought vwap_deviation flips a trend BUY → SELL."""
    from hanoon_prime.cortex import Cortex

    monkeypatch.setattr("hanoon_prime.contrarian.CONTRARIAN_MODE_ENABLED", True)
    c = Cortex()
    # Warm the rolling z-history so the spike yields an extreme z.
    for v in (0.0, 1.0, 2.0):
        c.evaluate({"vwap_deviation": v})
    overbought = c.evaluate({"vwap_deviation": 100.0})
    # Trend side (w=0.25, z=3) would score ~+0.64 → BUY; contrarian → SELL.
    assert overbought.verdict == "SELL"
    assert overbought.direction == -1


def test_cortex_contrarian_off_leaves_trend_verdict():
    """With the flag off (default), the same signal keeps the trend verdict."""
    from hanoon_prime.cortex import Cortex

    c = Cortex()
    for v in (0.0, 1.0, 2.0):
        c.evaluate({"vwap_deviation": v})
    overbought = c.evaluate({"vwap_deviation": 100.0})
    assert overbought.verdict != "SELL"
