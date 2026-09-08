"""tests/test_calibration — prediction-error score nudge (MoE port).

Faithful port of rebuild prediction_error_adjustment: the realized win rate
of a score band minus that band's predicted win probability, clamped to
±CALIB_BOUND (0.10). Off-by-default (CALIBRATION_NUDGE_ENABLED=False) so the
live path is unchanged until a human opts in. When enabled, a band that
outperforms its prediction gets a positive "lean-in" nudge; an
underperforming band gets a negative "pull-back" nudge; thin data (n <
BAND_MIN_SAMPLES) and the disabled flag both yield 0.0.
"""

from __future__ import annotations

import pytest

from hanoon_prime.brain.config import BAND_MIN_SAMPLES
from hanoon_prime.brain.realized_ev import RealizedStats
from hanoon_prime.edge import score_to_win_prob
from hanoon_prime.immune import CALIB_BOUND, CALIBRATION_NUDGE_ENABLED


def _pred_wp(score: float) -> float:
    """Prime's predicted win prob for |score| (static PRIOR_TOP=0.60)."""
    return score_to_win_prob(abs(score))


def test_calibration_nudge_off_by_default():
    """Flag locked False — nudge must be a no-op even with loaded history."""
    assert CALIBRATION_NUDGE_ENABLED is False
    r = RealizedStats(persist=False)
    for _ in range(BAND_MIN_SAMPLES + 5):
        r.add_outcome(0.5, True, 1.0)
    assert r.calibration_adjustment(0.5) == 0.0


def test_calibration_nudge_thin_data_is_zero(monkeypatch):
    """Below BAND_MIN_SAMPLES the band is unreliable → no nudge (no deadlock)."""
    monkeypatch.setattr(
        "hanoon_prime.brain.realized_ev.CALIBRATION_NUDGE_ENABLED", True
    )
    r = RealizedStats(persist=False)
    r.add_outcome(0.5, True, 1.0)  # n=1 < 20
    assert r.calibration_adjustment(0.5) == 0.0


@pytest.mark.parametrize("score", [0.5, -0.5])
def test_calibration_nudge_leans_in_when_band_beats_prediction(monkeypatch, score):
    """Realized band WR > predicted → positive nudge (lean in)."""
    monkeypatch.setattr(
        "hanoon_prime.brain.realized_ev.CALIBRATION_NUDGE_ENABLED", True
    )
    r = RealizedStats(persist=False)
    # 20 wins / 20 losses => WR=0.50; predicted for |0.5| = 0.25+0.5*0.35=0.425.
    for _ in range(20):
        r.add_outcome(score, True, 1.0)
    for _ in range(20):
        r.add_outcome(score, False, -1.0)
    adj = r.calibration_adjustment(score)
    assert adj == pytest.approx(0.5 - _pred_wp(score))
    assert 0.0 < adj <= CALIB_BOUND


@pytest.mark.parametrize("score", [0.5, -0.5])
def test_calibration_nudge_pull_back_when_band_underperforms(monkeypatch, score):
    """Realized band WR < predicted → negative nudge (pull back toward HOLD)."""
    monkeypatch.setattr(
        "hanoon_prime.brain.realized_ev.CALIBRATION_NUDGE_ENABLED", True
    )
    r = RealizedStats(persist=False)
    # 7 wins / 13 losses => WR=0.35; predicted for |0.5| = 0.425 → underperforms.
    for _ in range(7):
        r.add_outcome(score, True, 1.0)
    for _ in range(13):
        r.add_outcome(score, False, -1.0)
    adj = r.calibration_adjustment(score)
    assert adj == pytest.approx(0.35 - _pred_wp(score))
    assert -CALIB_BOUND <= adj < 0.0


def test_calibration_nudge_clamped_to_bound(monkeypatch):
    """Divergence beyond ±CALIB_BOUND is hard-clamped (rebuild semantics)."""
    monkeypatch.setattr(
        "hanoon_prime.brain.realized_ev.CALIBRATION_NUDGE_ENABLED", True
    )
    r = RealizedStats(persist=False)
    # All wins at an extreme score: WR=1.0 >> predicted (0.565) → clamps to +0.10.
    for _ in range(BAND_MIN_SAMPLES + 5):
        r.add_outcome(0.9, True, 1.0)
    assert r.calibration_adjustment(0.9) == pytest.approx(CALIB_BOUND)
    # All losses: WR=0.0 << predicted → clamps to -0.10.
    r2 = RealizedStats(persist=False)
    for _ in range(BAND_MIN_SAMPLES + 5):
        r2.add_outcome(0.9, False, -1.0)
    assert r2.calibration_adjustment(0.9) == pytest.approx(-CALIB_BOUND)
