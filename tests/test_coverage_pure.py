"""test_coverage_pure — pure-logic tests for previously-untested core modules.

Targets the deterministic, side-effect-free brains that were sitting at 0%:
exits, weight_enforcer, affective, dynamics, episodic, memory, debate,
reflection, the reflection buffer, and the learning supervisor.
"""

from __future__ import annotations

import time

import numpy as np

from hanoon_prime.brain.affective import AFFECTIVE_MOD_BOUND, Affective
from hanoon_prime.brain.config import (
    CONSOLIDATION_PULSES,
    DEFAULT_WEIGHTS,
    EPISODIC_CAPACITY,
    STALE_EXIT_MINUTES,
)
from hanoon_prime.brain.debate import DebateLayer, VerdictContext
from hanoon_prime.brain.dynamics import (
    HYSTERESIS_DELTA,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    Dynamics,
)
from hanoon_prime.brain.episodic import EpisodicMemory
from hanoon_prime.brain.exits import ExitPolicy
from hanoon_prime.brain.memory import JuliMemory
from hanoon_prime.brain.reflection import Reflector, TradeClose
from hanoon_prime.brain.weight_enforcer import MAX_WEIGHT, WeightEnforcer, get_enforcer
from hanoon_prime.reflection.buffer import BUY, SELL, Fill, Trade, TradeBuffer
from hanoon_prime.reflection.supervisor import LearningSupervisor

# A representative alpha using real indicator names so episodic/reflectors see keys.
_SAMPLE_ALPHA = {
    "vpin": 0.3,
    "orderbook_imbalance": 0.2,
    "institutional_flow": 0.6,
    "momentum": 0.55,
    "vwap_deviation": 0.1,
    "rsi": 0.6,
    "macd_hist": 0.2,
    "bollinger_position": 0.5,
    "adx": 0.4,
    "stoch_k": 0.3,
    "mfi": 0.45,
}


# ── ExitPolicy ──────────────────────────────────────────────────────────
class TestExitPolicy:
    def test_eval_unregistered_is_hold(self):
        sig = ExitPolicy().evaluate("TSLA", current_price=200.0)
        assert sig.should_exit is False
        assert sig.reason == ""

    def test_profit_lock_triggers(self):
        pol = ExitPolicy()
        pol.register("TSLA", entry_price=100.0)
        # peak gain hits the first tier (0.10) then current giveback falls below lock
        pol.evaluate("TSLA", current_price=100.0, ib_unrealized_pnl=12.0)
        sig = pol.evaluate("TSLA", current_price=100.0, ib_unrealized_pnl=3.0)
        assert sig.should_exit is True
        assert sig.exit_type == "profit_lock"

    def test_giveback_triggers(self):
        pol = ExitPolicy()
        pol.register("TSLA", entry_price=100.0)
        pol.evaluate("TSLA", current_price=100.0, ib_unrealized_pnl=10.0)
        sig = pol.evaluate("TSLA", current_price=100.0, ib_unrealized_pnl=4.0)
        assert sig.should_exit is True
        assert sig.exit_type == "giveback"

    def test_stale_exit(self):
        pol = ExitPolicy()
        pol.register("TSLA", entry_price=100.0)
        pol._entry_ts["TSLA"] = time.time() - (STALE_EXIT_MINUTES * 60.0 + 5.0)
        sig = pol.evaluate("TSLA", current_price=100.0)
        assert sig.should_exit is True
        assert sig.exit_type == "stale"

    def test_consolidation_exit(self):
        pol = ExitPolicy()
        pol.register("TSLA", entry_price=100.0)
        sig = None
        for _ in range(CONSOLIDATION_PULSES):
            sig = pol.evaluate("TSLA", current_price=100.0)
        assert sig is not None
        assert sig.should_exit is True
        assert sig.exit_type == "consolidation"

    def test_hold_when_healthy(self):
        pol = ExitPolicy()
        pol.register("TSLA", entry_price=100.0)
        sig = pol.evaluate("TSLA", current_price=102.0, ib_unrealized_pnl=2.0)
        assert sig.should_exit is False

    def test_deregister_clears_state(self):
        pol = ExitPolicy()
        pol.register("TSLA", entry_price=100.0)
        pol.deregister("TSLA")
        assert pol.evaluate("TSLA", current_price=100.0).should_exit is False


# ── WeightEnforcer ──────────────────────────────────────────────────────
class TestWeightEnforcer:
    def test_on_adapt_caps_oversized_weight(self):
        eng = WeightEnforcer()
        w = {"a": 0.5, "b": 0.3}
        eng.on_adapt(w, "a")
        assert w["a"] == MAX_WEIGHT

    def test_repair_on_load_empty(self):
        eng = WeightEnforcer()
        assert eng.repair_on_load({}) is False

    def test_repair_on_load_bounds(self):
        eng = WeightEnforcer()
        w = dict(DEFAULT_WEIGHTS)
        w["momentum"] = 0.5  # exceeds MAX_WEIGHT
        repaired = eng.repair_on_load(w)
        assert repaired is True
        assert w["momentum"] <= MAX_WEIGHT

    def test_repair_on_load_membership(self):
        eng = WeightEnforcer()
        w = dict(DEFAULT_WEIGHTS)
        w["momentum"] = 0.5
        w["bogus_key"] = 0.1
        del w["vpin"]
        eng.repair_on_load(w)
        assert "bogus_key" not in w
        assert "vpin" in w

    def test_check_integrity_ok(self):
        eng = WeightEnforcer()
        report = eng.check_integrity(dict(DEFAULT_WEIGHTS))
        assert report["ok"] is True
        assert report["issues"] == []

    def test_check_integrity_detects_issues(self):
        eng = WeightEnforcer()
        report = eng.check_integrity({"a": 0.001, "b": 0.9})
        assert report["ok"] is False
        assert len(report["issues"]) > 0

    def test_get_enforcer_singleton(self):
        assert get_enforcer() is get_enforcer()


# ── Affective ───────────────────────────────────────────────────────────
class TestAffective:
    def test_below_window_returns_default(self):
        aff = Affective(window=20)
        aff.update(True, 0.01)
        state = aff.evaluate()
        assert state.fear == 0.0
        assert state.greed == 0.0
        assert state.modifier == 0.0

    def test_wins_produce_greed(self):
        aff = Affective()
        for _ in range(5):
            aff.update(True, 0.02)
        state = aff.evaluate()
        assert state.greed > 0.0
        assert state.streak == 5

    def test_losses_produce_fear(self):
        aff = Affective()
        for _ in range(5):
            aff.update(False, -0.02)
        state = aff.evaluate()
        assert state.fear > 0.0
        assert state.streak == -5

    def test_modifier_bounded(self):
        aff = Affective()
        for _ in range(10):
            aff.update(True, 0.05)
        state = aff.evaluate()
        assert -AFFECTIVE_MOD_BOUND <= state.modifier <= AFFECTIVE_MOD_BOUND


# ── Dynamics ────────────────────────────────────────────────────────────
class TestDynamics:
    def test_process_returns_reason_and_bounds(self):
        dyn = Dynamics()
        score, reason = dyn.process(0.8, direction=1)
        assert -1.0 <= score <= 1.0
        assert "vel=" in reason

    def test_hysteresis_on_direction_change(self):
        dyn = Dynamics()
        dyn.process(0.8, direction=1)
        # opposite direction triggers the hysteresis penalty
        assert dyn._apply_hysteresis(0.8, direction=-1) == -HYSTERESIS_DELTA

    def test_adapt_threshold_raises_on_error(self):
        dyn = Dynamics(base_threshold=0.58)
        dyn.adapt_threshold(0.7)  # > 0.6
        assert dyn.threshold > 0.58
        assert dyn.threshold <= THRESHOLD_MAX

    def test_adapt_threshold_lowers_on_low_error(self):
        dyn = Dynamics(base_threshold=0.58)
        dyn.adapt_threshold(0.1)  # < 0.3
        assert dyn.threshold < 0.58
        assert dyn.threshold >= THRESHOLD_MIN

    def test_adapt_threshold_raises_at_boundary(self):
        dyn = Dynamics(base_threshold=0.58)
        dyn.adapt_threshold(0.50)  # >= 0.50 (loss error)
        assert abs(dyn.threshold - 0.59) < 1e-9

    def test_adapt_threshold_lowers_at_boundary(self):
        dyn = Dynamics(base_threshold=0.58)
        dyn.adapt_threshold(0.44)  # < 0.45 (confident win)
        assert abs(dyn.threshold - 0.575) < 1e-9

    def test_no_adapt_in_dead_band(self):
        dyn = Dynamics(base_threshold=0.58)
        dyn.adapt_threshold(0.47)  # between 0.45 and 0.50 — no change
        assert dyn.threshold == 0.58

    def test_refractory_then_expires(self):
        dyn = Dynamics()
        dyn.set_refractory(2.0)
        assert dyn._apply_refractory() < 0.0
        dyn._refractory_until = time.time() - 1.0
        assert dyn._apply_refractory() == 0.0

    def test_threshold_stays_in_band(self):
        dyn = Dynamics()
        rng = np.random.default_rng(0)
        for _ in range(130):
            dyn.process(float(rng.uniform(-1, 1)), direction=1)
        assert THRESHOLD_MIN <= dyn.threshold <= THRESHOLD_MAX


# ── Episodic ────────────────────────────────────────────────────────────
class TestEpisodic:
    def test_predict_below_min_samples(self):
        mem = EpisodicMemory()
        exp_ret, conf = mem.predict(_SAMPLE_ALPHA)
        assert exp_ret == 0.0
        assert conf == 0.0
        assert mem.size == 0

    def test_add_then_predict(self):
        mem = EpisodicMemory()
        for _ in range(15):
            mem.add(_SAMPLE_ALPHA, 0.03)
        exp_ret, conf = mem.predict(_SAMPLE_ALPHA)
        assert isinstance(exp_ret, float)
        assert 0.0 <= conf <= 1.0
        assert abs(mem.modifier(_SAMPLE_ALPHA)) <= 0.1

    def test_recall_similar(self):
        mem = EpisodicMemory()
        for _ in range(12):
            mem.add(_SAMPLE_ALPHA, 0.02)
        results = mem.recall_similar(_SAMPLE_ALPHA, k=3)
        assert len(results) <= 3
        assert all("won" in r and "distance" in r for r in results)


# ── JuliMemory ──────────────────────────────────────────────────────────
class TestJuliMemory:
    def test_weights_default_and_roundtrip(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "state.json")
        assert mem.get_weights() == dict(DEFAULT_WEIGHTS)
        mem.set_weights(dict(DEFAULT_WEIGHTS))
        # reload from disk
        mem2 = JuliMemory(path=tmp_path / "state.json")
        assert mem2.get_weights() == dict(DEFAULT_WEIGHTS)

    def test_set_weights_repairs_corruption(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "state.json")
        w = dict(DEFAULT_WEIGHTS)
        w["momentum"] = 0.9
        w["stray"] = 0.1
        mem.set_weights(w)
        loaded = mem.get_weights()
        assert "stray" not in loaded
        assert loaded["momentum"] <= MAX_WEIGHT

    def test_add_episode_and_cap(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "state.json")
        for i in range(EPISODIC_CAPACITY + 5):
            mem.add_episode([float(i)], 0.01, ticker="TSLA")
        assert len(mem.get_episodes()) == EPISODIC_CAPACITY

    def test_record_score_history_capped(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "state.json")
        for i in range(520):
            mem.record_score("TSLA", float(i % 10) / 10.0)
        assert len(mem.get_score_history("TSLA")) == 500

    def test_pred_error_ema(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "state.json")
        mem.update_pred_error(0.9, 1.0)
        assert 0.0 <= mem.pred_error <= 1.0

    def test_record_outcome_and_win_rate(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "state.json")
        mem.record_outcome(True)
        mem.record_outcome(True)
        mem.record_outcome(False)
        assert mem.total_trades == 3
        assert 0.5 < mem.win_rate < 1.0

    def test_add_lesson_and_cap(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "state.json")
        for i in range(510):
            mem.add_lesson({"ticker": "TSLA", "i": i})
        # cap invariant: lessons never exceed the trim trigger (500);
        # and once trimmed they stay bounded well under it.
        assert 250 <= len(mem.get_lessons()) <= 500

    def test_threshold_clamped(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "state.json")
        mem.threshold = 0.99
        assert mem.threshold == 0.70
        mem.threshold = -1.0
        assert mem.threshold == 0.45  # THRESHOLD_MIN

    def test_snapshot_keys(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "state.json")
        snap = mem.snapshot()
        for key in (
            "weights",
            "episodes",
            "win_rate",
            "total_trades",
            "pred_error_ema",
            "threshold",
            "lessons",
        ):
            assert key in snap

    def test_load_corrupt_does_not_crash(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        mem = JuliMemory(path=p)
        assert mem.get_weights() == dict(DEFAULT_WEIGHTS)


# ── DebateLayer ─────────────────────────────────────────────────────────
class TestDebateLayer:
    def test_approve_with_strong_alpha(self):
        layer = DebateLayer()
        alpha = {
            "momentum": 0.8,
            "adx": 0.7,
            "rsi": 0.9,
            "vpin": 0.5,
            "orderbook_imbalance": 0.5,
            "institutional_flow": 0.5,
            "vwap_deviation": 0.5,
        }
        ctx = VerdictContext(regime="normal", ev=0.06, rr=3.0)
        verdict = layer.debate("TSLA", alpha, score=0.6, ctx=ctx)
        assert verdict.approved is True
        assert 0.1 <= verdict.confidence <= 0.9

    def test_reject_with_weak_alpha(self):
        layer = DebateLayer()
        ctx = VerdictContext(regime="volatile", ev=0.0, rr=1.0)
        verdict = layer.debate("TSLA", {}, score=0.2, ctx=ctx)
        assert verdict.approved is False
        assert layer.modifier == 0.0

    def test_cache_returns_same(self):
        layer = DebateLayer()
        ctx = VerdictContext(regime="normal", ev=0.06, rr=3.0)
        alpha = {"momentum": 0.8, "adx": 0.7, "rsi": 0.9, "vpin": 0.5}
        first = layer.debate("TSLA", alpha, score=0.6, ctx=ctx)
        second = layer.debate("TSLA", alpha, score=0.6, ctx=ctx)
        assert first is second


# ── Reflector ───────────────────────────────────────────────────────────
class TestReflector:
    def test_on_trade_close_winner(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "m.json")
        episodic = EpisodicMemory()
        refl = Reflector(mem, episodic)
        trade = TradeClose(
            ticker="TSLA",
            won=True,
            pnl_pct=0.08,
            direction=1,
            alpha=_SAMPLE_ALPHA,
            predicted_score=0.6,
        )
        refl.on_trade_close(trade)
        assert mem.total_trades == 1
        assert len(mem.get_episodes()) == 1
        assert len(mem.get_lessons()) == 1

    def test_on_trade_close_loser_no_lesson(self, tmp_path):
        mem = JuliMemory(path=tmp_path / "m.json")
        refl = Reflector(mem, EpisodicMemory())
        trade = TradeClose(
            ticker="TSLA",
            won=False,
            pnl_pct=0.01,
            direction=1,
            alpha=_SAMPLE_ALPHA,
            predicted_score=0.3,
        )
        refl.on_trade_close(trade)
        assert mem.total_trades == 1
        assert len(mem.get_lessons()) == 0


# ── TradeBuffer ─────────────────────────────────────────────────────────
class TestTradeBuffer:
    def test_open_does_not_close(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "buf.json")
        result = buf.on_fill(
            Fill(ticker="TSLA", side=BUY, qty=10, price=100.0, time=1.0)
        )
        assert result is None
        assert buf.snapshot()["open"] == 1

    def test_round_trip_winner(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "buf.json")
        buf.on_fill(Fill(ticker="TSLA", side=BUY, qty=10, price=100.0, time=1.0))
        closed = buf.on_fill(
            Fill(ticker="TSLA", side=SELL, qty=10, price=110.0, time=2.0)
        )
        assert closed is not None
        assert closed.win is True
        assert closed.pnl > 0
        assert buf.get_win_rate() == 1.0
        assert buf.get_total_pnl() > 0
        assert buf.snapshot()["open"] == 0

    def test_callback_fired(self, tmp_path):
        seen = []
        buf = TradeBuffer(
            filepath=tmp_path / "buf.json",
            on_trade_closed=seen.append,
        )
        buf.on_fill(Fill(ticker="TSLA", side=BUY, qty=10, price=100.0, time=1.0))
        buf.on_fill(Fill(ticker="TSLA", side=SELL, qty=10, price=90.0, time=2.0))
        assert len(seen) == 1
        assert seen[0].win is False

    def test_persisted_trades_reload(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "buf.json")
        buf.on_fill(Fill(ticker="TSLA", side=BUY, qty=10, price=100.0, time=1.0))
        buf.on_fill(Fill(ticker="TSLA", side=SELL, qty=10, price=110.0, time=2.0))
        buf2 = TradeBuffer(filepath=tmp_path / "buf.json")
        assert buf2.get_total_pnl() > 0


# ── LearningSupervisor ──────────────────────────────────────────────────
class TestLearningSupervisor:
    def test_memory_none_attribute_exists(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        sup = LearningSupervisor(buf, memory=None)
        assert sup._memory is None

    def test_on_trade_close_records_outcome(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "b.json")

        class FakeMem:
            def __init__(self):
                self.calls = 0

            def record_outcome(self, won):
                self.calls += 1
                assert isinstance(won, bool)

            def update_pred_error(self, predicted, actual):
                self.calls += 1
                assert predicted is not None and actual is not None

            def set_weights(self, weights):
                pass

        mem = FakeMem()
        sup = LearningSupervisor(buf, memory=mem)
        sup.on_trade_close(Trade("t1", "TSLA", 0, 1, 100, 110, 10, 100, 0, True))
        # Single-writer contract (v2.1): the supervisor NEVER writes memory
        # (the brain owns learning). It previously double-wrote outcomes and
        # polluted the pred-error EMA with entry/exit prices.
        assert mem.calls == 0

    def test_force_review_empty(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        sup = LearningSupervisor(buf, memory=None)
        report = sup.force_review()
        assert report["summary"] == {}
        assert report["action_items"] == []

    def test_force_review_with_losing_trade(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        sup = LearningSupervisor(buf, memory=None)
        buf.on_fill(Fill(ticker="TSLA", side=BUY, qty=1, price=100.0, time=10.0))
        buf.on_fill(Fill(ticker="TSLA", side=SELL, qty=1, price=90.0, time=11.0))
        report = sup.force_review()
        assert report["summary"]["total"] == 1
        assert len(report["action_items"]) >= 1

    def test_run_daily_and_weekly_does_not_crash(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        sup = LearningSupervisor(buf, memory=None)
        sup._run_daily()
        sup._run_weekly()

    def test_start_stop_thread(self, tmp_path, monkeypatch):
        # Neutralise the 60s sleep so the daemon loop exits promptly.
        monkeypatch.setattr(
            "hanoon_prime.reflection.supervisor.time.sleep", lambda *_: None
        )
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        sup = LearningSupervisor(buf, memory=None)
        sup.start()
        assert sup._running is True
        sup.stop()
        assert sup._running is False

    def test_apply_one_decay_path(self, tmp_path):
        # Five losing trades for one ticker -> "Decay weight" action item.
        # JuliMemory has no micro_adjust_weight, so the attempt is caught.
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        mem = JuliMemory(path=tmp_path / "mem.json")
        sup = LearningSupervisor(buf, memory=mem)
        for i in range(5):
            t0 = 1000.0 + i
            buf.on_fill(Fill(ticker="TSLA", side=BUY, qty=1, price=100.0, time=t0))
            buf.on_fill(Fill(ticker="TSLA", side=SELL, qty=1, price=90.0, time=t0 + 1))
        report = sup.force_review()
        assert any(
            item.startswith("Decay weight for TSLA") for item in report["action_items"]
        )
        assert sup._memory is mem

    def test_snapshot_fields(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        sup = LearningSupervisor(buf, memory=None)
        snap = sup.snapshot()
        assert set(snap) == {"running", "last_daily", "last_weekly", "buffer"}
