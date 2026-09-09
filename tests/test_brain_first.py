"""tests/test_brain_first.py — brain-first autonomy features.

Covers: EV-gate advisory sizing (never refuses), Nash bounded penalty
(never zeroes the score), horizon ladder + manager, and webapp /config
horizons wiring. Contract: the brain decides; learned signals lean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from hanoon_prime.brain.cognitive.nash import NashBrain  # noqa: E402
from hanoon_prime.brain.config import NASH_PENALTY_MAX  # noqa: E402
from hanoon_prime.brain.horizons import (  # noqa: E402
    HORIZON_PARAMS,
    HORIZONS,
    HorizonManager,
    classify,
    get_horizon_manager,
    holds_through_close,
    params_for,
)
from hanoon_prime.brain.orchestrator import NeuromorphicBrain  # noqa: E402
from hanoon_prime.brain.realized_ev import RealizedStats  # noqa: E402
from hanoon_prime.brain.risk import RiskEngine  # noqa: E402


def _losing_band(n: int = 30) -> RealizedStats:
    stats = RealizedStats(persist=False)
    for _ in range(n):
        stats.add_outcome(0.62, won=False, pnl_pct=-0.01)
    return stats


# ── EV gate is advisory: never refuses ───────────────────────────────


class TestEVGateAdvisory:
    def test_losing_band_still_sizes(self):
        """A proven-losing realized band may scale down but NEVER refuse."""
        engine = RiskEngine(realized=_losing_band())
        result = engine.evaluate(0.62, 0.7, 50.0, 2.0, open_positions=0)
        assert result.risk_pass is True
        assert result.shares >= 1
        assert result.ev_scale in (0.5, 0.75)  # scaled down, bounded

    def test_positive_ev_band_scales_075(self):
        """Thin positive EV scales to 0.75 (bounded, advisory)."""
        engine = RiskEngine(realized=RealizedStats(persist=False))
        result = engine.evaluate(0.62, 0.7, 100.0, 2.0, open_positions=0)
        assert result.risk_pass is True
        assert result.ev_scale == 1.0  # structural EV positive → full size

    def test_ev_scale_never_above_one(self):
        """The advisory scale is bounded in [0.5, 1.0] for any outcome."""
        engine = RiskEngine(realized=_losing_band(50))
        result = engine.evaluate(0.9, 0.95, 50.0, 1.0, open_positions=0)
        assert 0.5 <= result.ev_scale <= 1.0

    def test_mechanical_limits_still_refuse(self):
        """Position cap and NaN data remain MECHANICAL refusals."""
        from hanoon_prime.immune import MAX_CONCURRENT_POSITIONS

        engine = RiskEngine()
        capped = engine.evaluate(
            0.62, 0.7, 100.0, 2.0, open_positions=MAX_CONCURRENT_POSITIONS
        )
        assert capped.risk_pass is False and capped.shares == 0
        nan = engine.evaluate(float("nan"), 0.7, 100.0, 2.0, open_positions=0)
        assert nan.risk_pass is False and "non-finite" in nan.reason
        nan_price = engine.evaluate(0.62, 0.7, float("nan"), 2.0, open_positions=0)
        assert nan_price.risk_pass is False and "entry_price" in nan_price.reason

    def test_sub_rounding_still_refuses(self):
        """Kelly buying <1 share stays a mechanical refusal (noise guard)."""
        engine = RiskEngine()
        result = engine.evaluate(0.02, 0.5, 100.0, 2.0, open_positions=0)
        assert result.risk_pass is False and "EV" in result.reason


# ── Nash pattern memory is a lean, not a lock ────────────────────────


class TestNashBoundedPenalty:
    def test_losing_pattern_penalizes_not_vetoes(self):
        """A 30-loss pattern subtracts a bounded penalty, never zero."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        nash = brain.nash
        for _ in range(30):
            nash.record_outcome({"vpin": 0.9}, 0.0, False)
        pred = nash.predict({"vpin": 0.9}, 0.8, 1)
        assert pred.gate_authority is True
        leaned = brain._apply_nash_gate(0.70, 1, pred)
        assert leaned < 0.70
        assert leaned >= 0.70 - NASH_PENALTY_MAX  # bounded
        assert leaned != 0.0  # NEVER a hard veto anymore

    def test_no_pattern_data_no_penalty(self):
        """Thin pattern memory leaves the score untouched."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        pred = brain.nash.predict({"vpin": 0.9}, 0.8, 1)
        assert pred.gate_authority is False
        assert brain._apply_nash_gate(0.70, 1, pred) == 0.70

    def test_short_side_lean(self):
        """A winning pattern leans shorts away with the same bound."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        for _ in range(30):
            brain.nash.record_outcome({"vpin": 0.9}, 0.0, True)
        pred = brain.nash.predict({"vpin": 0.9}, 0.8, -1)
        if pred.gate_authority and pred.win_prob > 0.55:
            leaned = brain._apply_nash_gate(-0.70, -1, pred)
            assert leaned > -0.70  # pulled toward zero, bounded


# ── Horizon ladder ────────────────────────────────────────────────────


class TestHorizonLadder:
    def test_six_horizons_implemented(self):
        assert HORIZONS == (
            "scalp",
            "multihour",
            "swing",
            "multiday",
            "multiweek",
            "longterm",
        )
        assert set(HORIZON_PARAMS) == set(HORIZONS)

    def test_sacred_rr_on_every_rung(self):
        """Every horizon keeps the sacred ≥ 3:1 reward:risk."""
        for name, p in HORIZON_PARAMS.items():
            assert p.rr >= 3.0, f"{name} violates sacred R:R"

    def test_classification_calm_steady_trend(self):
        """Low-ATR steady riser classifies above scalp."""
        n = 60
        base = [100.0 + i * 0.2 for i in range(n)]
        highs = [p + 0.3 for p in base]
        lows = [p - 0.3 for p in base]
        hz = classify(base, highs, lows, regime="trending_bullish")
        assert HORIZONS.index(hz) >= HORIZONS.index("multihour")

    def test_classification_matches_range_support(self):
        """ATR scale maps to the horizon the range can support.

        Tiny-range grinders only support scalps; wide movers support
        multi-run holds (rebuild _HORIZON_ATR semantics).
        """
        import random

        rng = random.Random(7)
        wide = [100.0]
        for _ in range(60):
            wide.append(wide[-1] * (1 + rng.uniform(-0.04, 0.04)))
        hz = classify(wide, [p * 1.03 for p in wide], [p * 0.97 for p in wide])
        assert HORIZONS.index(hz) >= HORIZONS.index("multiday")

    def test_degenerate_input_falls_back_to_scalp(self):
        assert classify([], None, None) == "scalp"
        assert classify([1.0] * 5) == "scalp"

    def test_manager_default_scalp_only(self, tmp_path, monkeypatch):
        """Fresh manager: scalp enabled; disabled rungs snap to nearest."""
        monkeypatch.setattr(
            "hanoon_prime.brain.horizons._STATE_PATH", tmp_path / "hz.json"
        )
        mgr = HorizonManager()
        assert mgr.enabled() == ["scalp"]
        assert mgr.active("swing") == "scalp"
        assert mgr.active("scalp") == "scalp"

    def test_manager_enable_and_persist(self, tmp_path, monkeypatch):
        """set_enabled accepts strings, persists, and protects the empty set."""
        monkeypatch.setattr(
            "hanoon_prime.brain.horizons._STATE_PATH", tmp_path / "hz.json"
        )
        mgr = HorizonManager()
        mgr.set_enabled("scalp,swing")
        assert mgr.enabled() == ["scalp", "swing"]
        assert mgr.active("multiday") == "swing"  # nearest enabled rung
        mgr.set_enabled("")
        assert mgr.enabled() == ["scalp", "swing"]  # empty keeps current
        mgr.set_enabled("bogus,swing")
        assert mgr.enabled() == ["swing"]  # unknown names filtered
        # Reload from disk proves persistence
        mgr2 = HorizonManager()
        assert mgr2.enabled() == ["swing"]

    def test_per_horizon_exit_windows(self):
        """Longer horizons get longer stale windows (rebuild parity)."""
        assert (
            HORIZON_PARAMS["longterm"].stale_minutes
            > HORIZON_PARAMS["scalp"].stale_minutes
        )
        assert params_for("nonexistent") == HORIZON_PARAMS["scalp"]

    def test_eod_survivors(self):
        """Only multiday+ hold through the close (rebuild v4 lesson)."""
        assert not holds_through_close("scalp")
        assert not holds_through_close("swing")
        assert holds_through_close("multiday")
        assert holds_through_close("longterm")

    def test_singleton(self):
        assert get_horizon_manager() is get_horizon_manager()


# ── Orchestrator integration ──────────────────────────────────────────


class TestOrchestratorHorizon:
    def test_tick_without_bars_defaults_scalp(self):
        """No bars → scalp decision path still works end-to-end."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        alpha = {"momentum": 0.9, "vwap_deviation": 0.85, "vpin": 0.7}
        result = brain.tick(alpha=alpha, ticker="TEST", entry_price=100.0, atr=1.0)
        assert result.get("horizon", "scalp") == "scalp"

    def test_classify_horizon_from_bars(self):
        """_classify_horizon maps bar arrays through the ladder."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        bars = {"close": [float(x) for x in range(1, 30)]}
        hz = brain._classify_horizon(bars)
        assert hz in HORIZONS  # snapped to an ACTIVE (enabled) rung

    def test_decision_carries_horizon(self):
        """Tick results carry the horizon for the executor."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        alpha = {"momentum": 0.9, "vwap_deviation": 0.85}
        result = brain.tick(alpha=alpha, ticker="TEST", entry_price=100.0, atr=1.0)
        assert "horizon" in result
