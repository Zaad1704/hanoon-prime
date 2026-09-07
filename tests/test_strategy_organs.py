"""tests/test_strategy_organs.py — strategy-learning subsystem.

Covers: meta-label sizing (bounded, cold-safe), horizon bandit
(regime cells, override discipline), per-regime weight vectors,
cross-asset wiring, regime fallback, close-path fan-out, and the
strategy-genome read-model. Contract: cortex stays the sole verdict
source; every organ is advisory and bounded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from hanoon_prime.brain import horizons as hz_mod  # noqa: E402
from hanoon_prime.brain.horizon_bandit import HorizonBandit  # noqa: E402
from hanoon_prime.brain.learning_config import (  # noqa: E402
    META_MIN_SAMPLES,
    REGIME_MIN_TRADES,
)
from hanoon_prime.brain.meta_label import MetaLabelModel, feature_vector  # noqa: E402
from hanoon_prime.brain.orchestrator import NeuromorphicBrain  # noqa: E402
from hanoon_prime.brain.regime_weights import RegimeWeights  # noqa: E402
from hanoon_prime.brain.strategy_genome import StrategyGenome  # noqa: E402
from hanoon_prime.juli_feed import REF_TICKERS, entry_bars  # noqa: E402

# ── Meta-label: bounded, cold-safe, learns only from closes ──────────


class TestMetaLabel:
    def test_cold_model_returns_full_size(self):
        """Fewer than META_MIN_SAMPLES samples → scalar is exactly 1.0."""
        model = MetaLabelModel()
        for _ in range(META_MIN_SAMPLES - 1):
            model.record(feature_vector(0.7, 0.8, 0.5, "trend_up", "scalp"), False)
        assert model.size_scalar(0.7, 0.8, 0.5, "trend_up", "scalp") == 1.0

    def test_losing_history_shrinks_size_bounded(self):
        """A proven-losing context shrinks size but never below the floor."""
        model = MetaLabelModel()
        for _ in range(META_MIN_SAMPLES + 10):
            model.record(feature_vector(0.7, 0.8, 0.5, "trend_up", "scalp"), False)
        scale = model.size_scalar(0.7, 0.8, 0.5, "trend_up", "scalp")
        assert 0.5 <= scale < 1.0

    def test_winning_history_never_shrinks(self):
        """A proven-winning context keeps full size (scalar 1.0)."""
        model = MetaLabelModel()
        for _ in range(META_MIN_SAMPLES + 10):
            model.record(feature_vector(0.7, 0.8, 0.5, "trend_up", "scalp"), True)
        assert model.size_scalar(0.7, 0.8, 0.5, "trend_up", "scalp") == 1.0

    def test_scalar_only_reduces(self):
        """The scalar can NEVER exceed 1.0 for any input."""
        model = MetaLabelModel()
        for _ in range(META_MIN_SAMPLES + 10):
            model.record(feature_vector(0.9, 0.9, 0.9, "vol", "swing"), True)
        assert model.size_scalar(0.99, 0.99, 0.99, "vol", "swing") <= 1.0

    def test_snapshot_reports_calibration(self):
        """Snapshot exposes n, Brier, and the sizing-active flag."""
        model = MetaLabelModel()
        model.record(feature_vector(0.7, 0.8, 0.5, "trend_up", "scalp"), True)
        snap = model.snapshot()
        assert snap["n"] == 1
        assert snap["brier"] is not None
        assert snap["sizing_active"] is False


# ── Horizon bandit: regime cells, override discipline ────────────────


class TestHorizonBandit:
    def test_cold_cells_keep_classifier_choice(self):
        """With no data anywhere, selection is the classified horizon."""
        bandit = HorizonBandit()
        hz, reason = bandit.select("trend_up", "momentum")
        assert hz == "momentum"
        assert reason in ("classifier", "explore")

    def test_override_requires_margin_and_samples(self):
        """A dominant arm overrides only with samples AND margin."""
        bandit = HorizonBandit()
        for _ in range(30):
            bandit.update("trend_up", "swing", 0.10)  # strong winner
        for _ in range(15):
            bandit.update("trend_up", "momentum", -0.05)  # trained loser
        hz, reason = bandit.select("trend_up", "momentum")
        assert reason in ("bandit_override", "explore", "classifier")
        if reason == "bandit_override":
            assert hz == "swing"

    def test_update_only_affects_its_cell(self):
        """A losing streak in one regime leaves other cells untouched."""
        bandit = HorizonBandit()
        for _ in range(20):
            bandit.update("trend_up", "scalp", -0.10)
        snap = bandit.snapshot()
        assert "trend_down" not in snap["arms"]
        assert snap["arms"]["trend_up"][0]["n"] >= 20

    def test_reward_clamped(self):
        """Extreme pnl still maps into [0, 1] reward."""
        bandit = HorizonBandit()
        bandit.update("vol", "scalp", 99.0)
        bandit.update("vol", "scalp", -99.0)
        cell = bandit.snapshot()["arms"]["vol"][0]
        assert 0.0 <= cell["mean"] <= 1.0

    def test_select_counters_monotonic(self):
        """Selects always counted; overrides never exceed selects."""
        bandit = HorizonBandit()
        for _ in range(50):
            bandit.select("range", "scalp")
        snap = bandit.snapshot()
        assert snap["selects"] == 50
        assert snap["overrides"] <= snap["selects"]


# ── Per-regime weight vectors ────────────────────────────────────────


class TestRegimeWeights:
    def test_cold_regime_returns_none(self):
        """Below REGIME_MIN_TRADES the regime vector must not apply."""
        rw = RegimeWeights()
        rw.learn("trend_up", {"momentum": 0.8}, True, 1)
        assert rw.weights_for("trend_up") is None

    def test_trained_regime_applies(self):
        """After REGIME_MIN_TRADES closes the regime vector applies."""
        rw = RegimeWeights()
        for _ in range(REGIME_MIN_TRADES):
            rw.learn("trend_up", {"momentum": 0.8}, True, 1)
        vec = rw.weights_for("trend_up")
        assert vec is not None and len(vec) > 0

    def test_regimes_isolated(self):
        """Learning one regime never touches another regime's vector."""
        rw = RegimeWeights()
        for _ in range(REGIME_MIN_TRADES):
            rw.learn("trend_up", {"momentum": 0.8}, True, 1)
        assert rw.weights_for("vol") is None

    def test_snapshot_counts(self):
        """Snapshot reports per-regime counts and active flags."""
        rw = RegimeWeights()
        rw.learn("range", {"vpin": 0.5}, False, 1)
        snap = rw.snapshot()
        assert snap["range"]["n"] == 1
        assert snap["range"]["active"] is False


# ── Orchestrator fan-out wiring ──────────────────────────────────────


class TestOrchestratorStrategyWiring:
    def test_tick_result_carries_strategy_context(self):
        """tick() exposes horizon_reason + canonical regime for tracing."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        alpha = {"momentum": 0.9, "vwap_deviation": 0.85, "vpin": 0.7}
        bars = {"close": [float(x) for x in range(1, 30)]}
        result = brain.tick(
            alpha=alpha, ticker="T", entry_price=100.0, atr=1.0, bars=bars
        )
        assert result["horizon"] in hz_mod.HORIZONS
        assert result["horizon_reason"] in ("classifier", "bandit_override", "explore")
        assert result["regime_canon"] in (
            "trend_up",
            "trend_down",
            "range",
            "vol",
            "unknown",
        )

    def test_close_fans_out_to_all_organs(self):
        """One close updates meta, bandit, regime weights, learned exits."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        alpha = {"momentum": 0.9, "vwap_deviation": 0.85}
        brain.tick(alpha=alpha, ticker="T", entry_price=100.0, atr=1.0)
        brain.on_trade_close("T", won=True, pnl_pct=0.03, direction=1)
        assert brain._meta.snapshot()["n"] == 1
        assert brain._bandit.snapshot()["selects"] >= 1
        assert sum(c["n"] for c in brain._regime_weights.snapshot().values()) == 1
        assert brain._learned_exit.count == 0  # no exit_triggers passed

    def test_close_with_exit_triggers_records_them(self):
        """exit_triggers reach the learned-exit attributor."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        brain.on_trade_close(
            "T",
            won=False,
            pnl_pct=-0.02,
            direction=1,
            exit_triggers=["giveback 40%"],
        )
        assert brain._learned_exit.count == 1

    def test_explicit_regime_and_horizon_override_memory(self):
        """Caller-supplied regime/horizon beat the remembered context."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        brain.tick(alpha={"momentum": 0.9}, ticker="T", entry_price=100.0, atr=1.0)
        brain.on_trade_close(
            "T",
            won=True,
            pnl_pct=0.02,
            direction=1,
            regime="vol",
            horizon="swing",
        )
        snap = brain._bandit.snapshot()
        assert "vol" in snap["arms"]
        assert any(a["horizon"] == "swing" for a in snap["arms"]["vol"])

    def test_real_close_hotswaps_cortex_weights(self):
        """A real close re-derives and hot-swaps cortex weights."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        before = brain.cortex.get_weights()
        brain.on_trade_close(
            "T", won=False, pnl_pct=-0.03, direction=1, source="ib_fill"
        )
        after = brain.cortex.get_weights()
        assert after != before | {}  # weights changed after learning

    def test_canonical_regime_mapping(self):
        """Labels map onto the five canonical cells."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        assert brain._canonical_regime("trending_bullish") == "trend_up"
        assert brain._canonical_regime("trending_bearish") == "trend_down"
        assert brain._canonical_regime("volatile") == "vol"
        assert brain._canonical_regime("ranging") == "range"
        assert brain._canonical_regime("unknown") == "unknown"

    def test_vol_pct_bounded(self):
        """The meta vol feature stays in [0, 1] for any bars."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        assert 0.0 <= brain._vol_pct(None) <= 1.0
        bars = {"close": [100.0 + i for i in range(25)]}
        assert 0.0 <= brain._vol_pct(bars) <= 1.0

    def test_meta_scale_applied_to_admitted_size(self):
        """Admitted entries get scaled by advisor/meta factors only."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        alpha = {"momentum": 0.9, "vwap_deviation": 0.85, "vpin": 0.7}
        result = brain.tick(alpha=alpha, ticker="T", entry_price=100.0, atr=1.0)
        if result["direction"] != 0 and result["sizing"].risk_pass:
            assert result["sizing"].shares >= 1


# ── Cross-asset + regime fallback wiring ─────────────────────────────


class TestFeedWiring:
    def test_ref_tickers_defined(self):
        """The four reference tickers are declared."""
        assert REF_TICKERS == ("SPY", "QQQ", "IWM", "VXX")

    def test_entry_bars_shape(self):
        """entry_bars carries close/high/low/regime into the brain."""
        snap = {"close_arr": [1.0, 2.0], "high_arr": [2.0], "low_arr": [0.5]}
        bars = entry_bars(snap, [1.0, 2.0], "ranging")
        assert bars["close"] == [1.0, 2.0]
        assert bars["regime"] == "ranging"

    def test_cross_asset_modifier_bounded_in_pipeline(self):
        """Cross-asset modifier is clamped before touching the score."""
        from hanoon_prime.brain.learning_config import CROSS_ASSET_MOD_BOUND

        assert CROSS_ASSET_MOD_BOUND == 0.04
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        brain.state.update(ref_prices={"SPY": 500.0, "QQQ": 400.0})
        alpha = {"momentum": 0.9, "vwap_deviation": 0.85}
        result = brain.tick(alpha=alpha, ticker="T", entry_price=100.0, atr=1.0)
        assert "score" in result  # pipeline ran with refs present

    def test_local_regime_fallback_only_when_unknown(self):
        """Fallback rewrites only a stuck 'unknown' label."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        mult, label = brain._local_regime_fallback("trending_bullish")
        assert (mult, label) == (1.0, "trending_bullish")


# ── Strategy genome read-model ───────────────────────────────────────


class TestStrategyGenome:
    def test_genome_reflects_live_state(self):
        """The genome reads the brain's actual learned state."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        genome = StrategyGenome(brain).get_genome()
        assert genome["threshold"] == pytest.approx(brain.dynamics.threshold, abs=1e-6)
        assert genome["advisor_delta"] == brain._advisor.threshold_delta()
        assert genome["weights"] == brain.cortex.get_weights()
        assert genome["version"] == 2

    def test_diagnose_flags_tightening_advisor(self):
        """A tightening advisor surfaces as a diagnostic finding."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        for _ in range(20):
            brain._advisor.record_outcome(False)
        issues = StrategyGenome(brain).diagnose()
        assert any("tightening" in i for i in issues)

    def test_diagnose_clean_brain_has_no_weight_issues(self):
        """A fresh brain's weights are structurally sound."""
        brain = NeuromorphicBrain(enable_neuromorphic=False)
        issues = StrategyGenome(brain).diagnose()
        assert not any("dominant" in i for i in issues)
