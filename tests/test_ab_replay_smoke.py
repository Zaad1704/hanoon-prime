"""tests/test_ab_replay_smoke.py — A/B replay harness smoke coverage.

The organ A/B harness must not rot: it is the instrument that will decide
promotion once the funnel actually trades. Smoke test runs SIGNAL mode on a
tiny synthetic fixture through OFF vs BLEND and asserts the measurement
contract (bar deck + per-variant deltas). P/L funnel mode is intentionally
not unit-tested — its trade-count is config-dependent (near-zero on the
committed fixtures by design).
"""

from __future__ import annotations

import numpy as np

import hanoon_prime.immune as imm
import scripts.ab_brain_replay as ab
from hanoon_prime.brain import orchestrator as orb


def _synthetic_fixture(bars: int = 400) -> dict[str, dict]:
    t = np.linspace(0.0, 8.0 * np.pi, bars)
    close = (
        100.0
        + 6.0 * np.sin(t)
        + 0.4 * np.cumsum(np.random.default_rng(7).normal(0, 1, bars))
    )
    close = np.maximum(close, 5.0)
    high = close * 1.004
    low = close * 0.996
    volume = np.full(bars, 1000.0)
    return {"SMK": {"close": close, "high": high, "low": low, "volume": volume}}


def test_signal_mode_emits_measurement_contract():
    fixtures = _synthetic_fixture()
    off = ab.replay_signals(fixtures, ab.Variant("OFF"))
    blend = ab.replay_signals(fixtures, ab.Variant("BLEND", blend=True))
    assert off.signals.get("SMK")
    assert blend.signals.get("SMK")
    assert len(off.signals["SMK"]) == len(blend.signals["SMK"]) > 0
    sig = ab.summarize_signal(off, blend)
    assert sig["variant"] == "BLEND"
    assert sig["n"] == len(off.signals["SMK"])
    assert 0.0 <= sig["side_flip_rate"] <= 1.0
    assert -1.0 <= sig["score_corr"] <= 1.0
    assert sig["mean_abs_score_delta"] >= 0.0


def test_signal_mode_loads_real_fixture_snapshot():
    """Replays the real loader path without depending on data/ in CI."""
    fixture = _synthetic_fixture()
    snap = ab.build_snap(fixture["SMK"], ab.WINDOW)
    assert snap is not None
    assert len(snap["prices"]) >= 20
    assert len(snap["close_arr"]) == ab.WINDOW
    assert float(snap["atr"]) > 0.0


def test_harness_restores_gates_after_blend_run():
    _synthetic_fixture()
    ab.replay_signals(_synthetic_fixture(), ab.Variant("BLEND", blend=True))
    assert orb.NEURO_BLEND_ENABLED is False
    assert imm.NEURO_LEARN_ENABLED is False
    assert imm.NEURO_ADAPTIVE_THRESHOLD_ENABLED is False
    assert imm.NEURO_MOE_GATE_ENABLED is False
