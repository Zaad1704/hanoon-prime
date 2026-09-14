"""tests/test_allostasis — homeostatic setpoint, dyshomeostasis, dynamic fallen.

Phase B of AWAKENED_BRAIN.md: AllostaticController keeps per-regime
expected-edge setpoints; dynamic fallen threshold tightens the pillar
line when the regime has learned a negative norm; sustained deviations
past ALLOS_MARGIN trip dyshomeostasis.
"""

from __future__ import annotations

from hanoon_prime.brain.allostasis import AllostaticController
from hanoon_prime.brain.learning_config import ALLOS_MARGIN, ALLOS_VIOLATION_MIN
from hanoon_prime.brain.pillar_awareness import (
    STATE_FALLEN,
    STATE_TIPPING,
    STATE_WARMING,
    _apply_state,
    compute_pillar_awareness,
)


def _fresh() -> AllostaticController:
    return AllostaticController(persist=False)


def _record(edge: float, trades: int = 20) -> dict:
    """Build a win/loss record with the given edge and enough trades."""
    return {"edge": edge, "trades": trades, "wins": 0, "losses": 0}


# ── AllostaticController ─────────────────────────────────────────────
class TestSetpointTracking:
    def test_fresh_setpoint_is_zero(self):
        out = _fresh().update(_record(0.05, trades=0), "trend")
        assert out["setpoint"] == 0.0
        assert out["trades"] == 0

    def test_setpoint_shifts_toward_edge(self):
        ctrl = _fresh()
        for _ in range(40):
            out = ctrl.update(_record(0.05), "trend")
        assert out["setpoint"] > 0.0
        assert out["setpoint"] < 0.05  # slow alpha, never fully there

    def test_setpoint_unchanged_before_MIN_TRADES(self):
        ctrl = _fresh()
        out = ctrl.update(_record(0.08, trades=5), "trend")
        assert out["setpoint"] == 0.0
        assert out["trades"] == 5

    def test_negative_setpoint_affects_deviation(self):
        ctrl = _fresh()
        for _ in range(30):
            ctrl.update(_record(-0.03), "choppy")
        out = ctrl.setpoint("choppy")
        assert out["setpoint"] < 0.0
        assert out["deviation"] != 0.0


class TestDyshomeostasis:
    def test_violations_accumulate(self):
        ctrl = _fresh()
        # Seed enough trades so MIN_TRADES gate passes
        for _ in range(15):
            ctrl.update(_record(0.0), "trend")
        # Now swing to a sustained negative edge
        for _ in range(ALLOS_VIOLATION_MIN + 2):
            ctrl.update(_record(-0.20), "trend")
        assert ctrl.is_dyshomeostatic("trend")

    def test_violations_reset_on_good_update(self):
        ctrl = _fresh()
        for _ in range(ALLOS_VIOLATION_MIN + 1):
            ctrl.update(_record(-0.20), "trend")
        assert ctrl.is_dyshomeostatic("trend")
        for _ in range(5):
            ctrl.update(_record(0.0), "trend")
        assert not ctrl.is_dyshomeostatic("trend")

    def test_no_dyshomeostasis_with_zero_deviation(self):
        ctrl = _fresh()
        for _ in range(ALLOS_VIOLATION_MIN + 1):
            ctrl.update(_record(0.0), "trend")
        assert not ctrl.is_dyshomeostatic("trend")

    def test_margin_bounds_tolerance(self):
        ctrl = _fresh()
        edge_just_inside = ALLOS_MARGIN * 0.9
        for _ in range(ALLOS_VIOLATION_MIN + 1):
            ctrl.update(_record(edge_just_inside), "trend")
        assert not ctrl.is_dyshomeostatic("trend")

    def test_unknown_regime_defaults(self):
        ctrl = _fresh()
        out = ctrl.setpoint("unknown")
        assert out["setpoint"] == 0.0
        assert not out["dyshomeostatic"]


class TestPerRegime:
    def test_regimes_are_independent(self):
        ctrl = _fresh()
        for i in range(30):
            ctrl.update(_record(0.05), "trend")
            ctrl.update(_record(-0.15), "choppy")
            if i == 4:
                # Early: deviation-from-norm is large -> choppy is alarmed,
                # trend is not. Later the setpoint adapts to the -0.15 norm
                # and the transient alarm clears (deviation shrinks).
                assert ctrl.is_dyshomeostatic("choppy")
                assert not ctrl.is_dyshomeostatic("trend")
        trend = ctrl.setpoint("trend")
        choppy = ctrl.setpoint("choppy")
        assert trend["setpoint"] > 0.0
        assert choppy["setpoint"] < 0.0
        assert trend["setpoint"] > choppy["setpoint"]
        assert not ctrl.is_dyshomeostatic("trend")
        assert not ctrl.is_dyshomeostatic("choppy")


# ── Persistence ───────────────────────────────────────────────────────
class TestPersistence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "allo.json"
        first = AllostaticController(filepath=path)
        first.update(_record(-0.03), "trend")
        second = AllostaticController(filepath=path)
        assert second.setpoint("trend") == first.setpoint("trend")

    def test_missing_file_starts_fresh(self, tmp_path):
        ctrl = AllostaticController(filepath=tmp_path / "absent.json")
        assert ctrl.setpoint("trend")["setpoint"] == 0.0

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "allo.json"
        path.write_text("NOT JSON")
        ctrl = AllostaticController(filepath=path)
        assert ctrl.setpoint("trend")["setpoint"] == 0.0


# ── Snapshot ──────────────────────────────────────────────────────────
class TestSnapshot:
    def test_shape(self):
        ctrl = _fresh()
        ctrl.update(_record(0.0), "trend")
        snap = ctrl.snapshot()
        assert "trend" in snap
        assert set(snap["trend"]) == {
            "regime",
            "setpoint",
            "deviation",
            "dyshomeostatic",
            "violations",
            "trades",
        }


# ── Dynamic fallen threshold via pillar_awareness ─────────────────────
class TestDynamicFallen:
    def test_negative_setpoint_tightens_fallen(self):
        """With negative setpoint, a moderate loss is now FALLEN."""
        p = _apply_state_inverted(setpoint_edge=-0.05)
        # edge -0.07 → below setpoint → FALLEN (would have been TIPPING at PILLAR_EDGE_FALL)
        assert p["state"] == STATE_FALLEN

    def test_positive_setpoint_keeps_structural_floor(self):
        p = _apply_state_inverted(setpoint_edge=+0.05)
        # edge -0.07 > -0.10 → still TIPPING with structural floor
        assert p["state"] == STATE_TIPPING

    def test_no_setpoint_keeps_structural_floor(self):
        p = _apply_state_inverted(setpoint_edge=None)
        assert p["state"] == STATE_TIPPING


class TestComputeAwarenessSetpoint:
    def test_delegation_to_decorate(self):
        record = {"edge": 0.03, "trades": 20, "wins": 0, "losses": 0}
        p = compute_pillar_awareness(record, setpoint_edge=0.01)
        assert p["setpoint"] == 0.01
        assert p["setpoint_deviation"] == 0.02
        assert p["below_setpoint"] is False

    def test_below_setpoint_flag(self):
        record = {"edge": -0.02, "trades": 20, "wins": 0, "losses": 0}
        p = compute_pillar_awareness(record, setpoint_edge=0.01)
        assert p["below_setpoint"] is True
        assert p["setpoint_deviation"] < 0.0

    def test_warming_gets_setpoint_keys(self):
        p = compute_pillar_awareness(None, setpoint_edge=0.02)
        assert p["state"] == STATE_WARMING
        assert p["setpoint"] == 0.02


# ── Helper ────────────────────────────────────────────────────────────
def _apply_state_inverted(setpoint_edge=None):
    """Run _apply_state directly and return the mutated base dict."""
    base = {"edge": -0.07, "trades": 15, "state": STATE_WARMING, "upright": True}
    _apply_state(base, setpoint_edge=setpoint_edge)
    return base
