"""tests/test_somatic — Damasio-style gut-feel bias + precision dampening.

Phase C of AWAKENED_BRAIN.md: SomaticMarkerGenerator assembles pillar lean,
RPE mood, and the allostatic alarm into a bounded ±SOMATIC_MAX bias; a
negative marker precision-dampens the learned modifiers.
"""

from __future__ import annotations

import pytest

from hanoon_prime.brain.learning_config import (
    SOMATIC_DYS_PENALTY,
    SOMATIC_MAX,
    SOMATIC_PHASIC_GAIN,
    SOMATIC_PRECISION_DAMP,
    SOMATIC_PRECISION_LOOR,
    SOMATIC_TILT_GAIN,
    SOMATIC_TONIC_GAIN,
)
from hanoon_prime.brain.somatic import SomaticMarkerGenerator


def _fresh() -> SomaticMarkerGenerator:
    return SomaticMarkerGenerator()


def _pillar(tilt: float = 0.0) -> dict:
    return {"tilt": tilt, "edge": 0.0}


def _rpe(tonic: float = 0.0, phasic: float = 0.0) -> dict:
    return {"tonic": tonic, "phasic": phasic}


def _allostatic(dys: bool = False) -> dict:
    return {"dyshomeostatic": dys}


# ── Marker assembly ────────────────────────────────────────────────────
class TestMarkerBounds:
    def test_neutral_state_is_zero(self):
        m = _fresh().generate(_pillar(0.0), _rpe(0.0, 0.0), _allostatic(False))
        assert m == 0.0

    def test_marker_never_exceeds_bound(self):
        worst = _fresh().generate(_pillar(1.0), _rpe(-1.0, -1.0), _allostatic(True))
        assert -SOMATIC_MAX <= worst <= SOMATIC_MAX

    def test_all_none_inputs_is_zero(self):
        assert _fresh().generate(None, None, None) == 0.0


class TestPillarLean:
    def test_lean_pushes_negative(self):
        assert (
            _fresh().generate(_pillar(1.0), _rpe(), _allostatic(False))
            == -SOMATIC_TILT_GAIN
        )

    def test_no_lean_is_neutral(self):
        assert _fresh().generate(_pillar(0.0), _rpe(), _allostatic(False)) == 0.0

    def test_lean_never_half_contribution(self):
        assert (
            _fresh().generate(_pillar(0.5), _rpe(), _allostatic(False))
            == -SOMATIC_TILT_GAIN / 2
        )


class TestRpeMood:
    def test_positive_tonic_lifts(self):
        assert (
            _fresh().generate(_pillar(), _rpe(1.0), _allostatic(False))
            == SOMATIC_TONIC_GAIN
        )

    def test_negative_tonic_sinks(self):
        assert (
            _fresh().generate(_pillar(), _rpe(-1.0), _allostatic(False))
            == -SOMATIC_TONIC_GAIN
        )

    def test_phasic_contributes_along_sign(self):
        got = _fresh().generate(_pillar(), _rpe(0.0, 0.5), _allostatic(False))
        assert got == SOMATIC_PHASIC_GAIN * 0.5

    def test_non_numeric_channel_is_zero(self):
        assert (
            _fresh().generate(
                _pillar(), {"tonic": "bad", "phasic": None}, _allostatic()
            )
            == 0.0
        )


class TestAllostaticAlarm:
    def test_dyshomeostasis_penalty(self):
        assert (
            _fresh().generate(_pillar(), _rpe(), _allostatic(True))
            == SOMATIC_DYS_PENALTY
        )

    def test_alarm_plus_lean_stacks_within_bound(self):
        m = _fresh().generate(_pillar(1.0), _rpe(), _allostatic(True))
        assert m < 0.0
        assert m >= -SOMATIC_MAX


# ── Precision dampening ────────────────────────────────────────────────
class TestPrecision:
    def test_non_negative_marker_keeps_full_precision(self):
        for marker in (0.0, 0.05, SOMATIC_MAX):
            assert _fresh().precision_weight(marker) == 1.0

    def test_negative_marker_dampens_linearly(self):
        got = _fresh().precision_weight(-0.05)
        assert got == pytest.approx(1.0 - SOMATIC_PRECISION_DAMP * 0.05)

    def test_precision_floored_at_floor(self):
        assert _fresh().precision_weight(-SOMATIC_MAX) >= SOMATIC_PRECISION_LOOR

    def test_precision_never_exceeds_one(self):
        g = _fresh()
        assert g.precision_weight(0.0) <= 1.0
        assert g.precision_weight(-0.01) <= 1.0

    def test_marker_then_precision_roundtrip(self):
        g = _fresh()
        marker = g.generate(_pillar(1.0), _rpe(-0.4), _allostatic(True))
        assert g.precision_weight(marker) >= SOMATIC_PRECISION_LOOR


class TestSnapshot:
    def test_shape(self):
        g = _fresh()
        marker = g.generate(_pillar(1.0), _rpe(-0.5), _allostatic(False))
        g.precision_weight(marker)
        snap = g.snapshot()
        assert set(snap) == {"marker", "precision"}
        assert snap["marker"] == pytest.approx(marker)
