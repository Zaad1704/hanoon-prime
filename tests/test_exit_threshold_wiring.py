"""tests/test_exit_threshold_wiring.py — E1: nine-rule exit learning is live.

``AdaptiveThresholds.update_from_outcome`` had zero callers (dead path in
the trade-close loop). These tests prove every real IRONYCLADE close now
feeds the learned exit thresholds through
``orchestrator.on_trade_close → _learn_exit_thresholds``, that hold/peak
are extracted from the tracked position BEFORE deregister, and that no
source outside IRONYCLADE can reach the learner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hanoon_prime.brain.adaptive_thresholds import AdaptiveThresholds
from hanoon_prime.brain.exits import ExitPolicy
from hanoon_prime.brain.threshold_limits import LR_DOWN


@pytest.fixture
def thresholds(tmp_path: Path, monkeypatch):
    """A hermetic AdaptiveThresholds both entry-ladder and close learning use."""
    at = AdaptiveThresholds(filepath=tmp_path / "adaptive.json")
    monkeypatch.setattr(
        "hanoon_prime.brain.orchestrator.get_adaptive_thresholds", lambda: at
    )
    monkeypatch.setattr(
        "hanoon_prime.brain.exit_ladder.get_adaptive_thresholds", lambda: at
    )
    return at


def test_on_trade_close_feeds_giveback_learner(thresholds):
    """A real giveback loss tightens the learned trailing keep-ratio."""
    from hanoon_prime.brain.orchestrator import NeuromorphicBrain

    brain = NeuromorphicBrain()
    brain.on_trade_close(
        "AAA",
        won=False,
        pnl_pct=-0.04,
        direction=1,
        source="real_trade",
        exit_triggers=["giveback"],
    )
    assert thresholds.get_trail_keep_ratio() == pytest.approx(0.55 - LR_DOWN)
    assert thresholds.telemetry()["update_count"] == 1


def test_close_extracts_real_peak_before_deregister(thresholds):
    """Peak return is read live, so the profit-lock tier that fired is reinforced."""
    from hanoon_prime.brain.orchestrator import NeuromorphicBrain

    brain = NeuromorphicBrain()
    brain.exits.register("TSLA", 100.0)
    brain.check_exit("TSLA", 112.0, ib_pnl=12.0, direction=1)
    assert brain.exits.peak_return_fraction("TSLA", 1) == pytest.approx(0.12)
    brain.on_trade_close(
        "TSLA",
        won=True,
        pnl_pct=0.025,
        direction=1,
        source="real_trade",
        exit_triggers=["profit_lock"],
    )
    tier = thresholds.get_profit_lock_tiers()[0]
    assert tier[1] == pytest.approx(0.04 * 1.05)  # locked floor raised
    assert brain.exits.is_registered("TSLA") is False  # deregistered after


def test_non_ironclade_sources_cannot_learn(thresholds):
    """A backtest close must never touch the learned exit thresholds."""
    from hanoon_prime.brain.orchestrator import NeuromorphicBrain

    brain = NeuromorphicBrain()
    brain.on_trade_close(
        "BBB",
        won=False,
        pnl_pct=-0.06,
        direction=1,
        source="backtest",
        exit_triggers=["giveback"],
    )
    assert thresholds.telemetry()["update_count"] == 0
    assert thresholds.get_trail_keep_ratio() == pytest.approx(0.55)


def test_peak_return_fraction_is_direction_aware():
    """Longs track the high; shorts the low; no false peaks on either side."""
    long = ExitPolicy()
    long.register("L", 100.0)
    long._update_peaks("L", 112.0, 12.0, direction=1)
    assert long.peak_return_fraction("L", 1) == pytest.approx(0.12)

    short = ExitPolicy()
    short.register("S", 100.0)
    short._update_peaks("S", 90.0, 10.0, direction=-1)
    assert short.peak_return_fraction("S", -1) == pytest.approx(0.10)
    # Looking at a short's peak with a long lens must clamp to no-signal.
    assert short.peak_return_fraction("S", 1) == 0.0
