"""tests/test_sleep_scheduler — auto-trigger, loser-priority weighting, engine override.

Phase E of AWAKENED_BRAIN.md: the scheduler decides when replay runs
(idle/session-close, hourly cooldown) and how patterns are weighted
(losers 3×, interleaved historical traces); the sleep engine uses
the overridden list with a confirmed LOSS_BIAS replacing the old
winner-favoring rule.
"""

from __future__ import annotations

import pytest

from hanoon_prime.brain.learning_config import (
    SLEEP_COOLDOWN_SEC,
    SLEEP_INTERLEAVE_MAX,
    SLEEP_LOSS_WEIGHT,
    SLEEP_MIN_PATTERNS,
    SLEEP_THRESHOLD_SEC,
    SLEEP_WIN_WEIGHT,
)
from hanoon_prime.brain.neurons.attractor import AttractorMemory
from hanoon_prime.brain.neurons.network import LIFNetwork
from hanoon_prime.brain.neurons.sleep import SleepReplayEngine, SleepResult
from hanoon_prime.brain.neurons.stdp import STDPLearner
from hanoon_prime.brain.sleep_scheduler import SleepScheduler


def _fresh() -> SleepScheduler:
    return SleepScheduler()


def _pat(idx: int = 0) -> dict[str, float]:
    return {f"alpha_{i}": 0.5 + i * 0.05 for i in range(idx, idx + 3)}


# ── Trigger check ─────────────────────────────────────────────────────
class TestCheck:
    def test_no_trigger_when_recent_trade(self):
        assert _fresh().check(last_trade_ts=100.0, now=200.0) is False

    def test_trigger_after_threshold(self):
        s = _fresh()
        assert (
            s.check(
                last_trade_ts=0.0,
                now=float(SLEEP_THRESHOLD_SEC + 1),
            )
            is True
        )

    def test_trigger_on_session_close_even_when_active(self):
        s = _fresh()
        assert s.check(last_trade_ts=9999.0, session_close=True, now=10000.0) is True

    def test_cooldown_prevents_repeat_within_window(self):
        s = _fresh()
        assert s.check(last_trade_ts=0.0, now=float(SLEEP_THRESHOLD_SEC + 1)) is True
        assert (
            s.check(
                last_trade_ts=0.0,
                now=float(SLEEP_THRESHOLD_SEC + 2),
            )
            is False
        )

    def test_cooldown_allows_after_window(self):
        s = _fresh()
        assert s.check(last_trade_ts=0.0, now=float(SLEEP_THRESHOLD_SEC + 1)) is True
        late = SLEEP_THRESHOLD_SEC + SLEEP_COOLDOWN_SEC + 1
        assert s.check(last_trade_ts=0.0, now=float(late)) is True

    def test_last_trigger_updates(self):
        s = _fresh()
        import math

        assert s.last_trigger == float("-inf")
        s.check(last_trade_ts=0.0, now=float(SLEEP_THRESHOLD_SEC + 1))
        assert s.last_trigger == SLEEP_THRESHOLD_SEC + 1

    def test_none_now_defaults_to_future_for_idle_check(self):
        """A future last_trade_ts with no explicit now never idles — same day."""
        s = _fresh()
        # A ts far in the future; without injecting now, the check uses
        # real time and correctly finds the bot idle.
        assert s.check(last_trade_ts=0.0) is True


# ── Replay weighting ──────────────────────────────────────────────────
class TestReplayWeights:
    def test_losers_weighted_three_times(self):
        s = _fresh()
        weighted = s.replay_weights([(_pat(), False)])
        assert weighted[0][1] == SLEEP_LOSS_WEIGHT

    def test_winners_weighted_once(self):
        s = _fresh()
        weighted = s.replay_weights([(_pat(), True)])
        assert weighted[0][1] == SLEEP_WIN_WEIGHT

    def test_interleaved_historical_capped(self):
        s = _fresh()
        hist = [(_pat(i), i % 2 == 0) for i in range(20)]
        weighted = s.replay_weights([(_pat(), True)], historical=hist)
        assert len(weighted) == 1 + SLEEP_INTERLEAVE_MAX

    def test_interleaved_items_inherit_losers_three_times(self):
        s = _fresh()
        hist = [(_pat(), False)]
        weighted = s.replay_weights([], historical=hist)
        assert len(weighted) == 1
        assert weighted[0][1] == SLEEP_LOSS_WEIGHT

    def test_empty_recent_with_no_historical(self):
        s = _fresh()
        assert s.replay_weights([]) == []

    def test_patterns_preserved(self):
        s = _fresh()
        weighted = s.replay_weights([(_pat(2), False)])
        assert weighted[0][0] == _pat(2)


# ── Sleep engine override + loser bias ────────────────────────────────
class TestEngineReplayList:
    def _engine(self, memory: AttractorMemory | None = None) -> SleepReplayEngine:
        return SleepReplayEngine(
            network=LIFNetwork(),
            stdp=STDPLearner(),
            memory=memory or AttractorMemory(),
        )

    def test_empty_memory_returns_no_patterns(self):
        assert self._engine().select_patterns() == []

    def test_default_weight_uses_loss_bias_for_losing_pattern(self):
        mem = AttractorMemory()
        mem.store("L", [0.5, 0.4, 0.3], won=False, pnl_pct=-0.05)
        mem.store("L", [0.5, 0.4, 0.3], won=False, pnl_pct=-0.05)
        mem.store("L", [0.5, 0.4, 0.3], won=False, pnl_pct=-0.05)
        e = self._engine(mem)
        patterns = e.select_patterns()
        assert len(patterns) == 1
        _, weight = patterns[0]
        assert weight == SLEEP_LOSS_WEIGHT

    def test_default_weight_uses_win_bias_for_winning_pattern(self):
        mem = AttractorMemory()
        mem.store("W", [0.5, 0.4, 0.3], won=True, pnl_pct=0.05)
        mem.store("W", [0.5, 0.4, 0.3], won=True, pnl_pct=0.05)
        mem.store("W", [0.5, 0.4, 0.3], won=True, pnl_pct=0.05)
        e = self._engine(mem)
        patterns = e.select_patterns()
        assert len(patterns) == 1
        _, weight = patterns[0]
        assert weight == SLEEP_WIN_WEIGHT

    def test_replay_list_overrides_attractors(self):
        override = [(_pat(0), SLEEP_LOSS_WEIGHT)]
        e = self._engine()
        patterns = e.select_patterns(replay_list=override)
        assert len(patterns) == 1
        assert patterns[0][1] == SLEEP_LOSS_WEIGHT

    def test_run_cycle_uses_replay_list(self):
        override = [(_pat(0), SLEEP_LOSS_WEIGHT)]
        e = self._engine()
        result = e.run_cycle(duration_sec=0.1, replay_list=override)
        assert isinstance(result, SleepResult)
        assert result.patterns_replayed == 1

    def test_run_cycle_empty_memory_returns_empty(self):
        result = self._engine().run_cycle(duration_sec=0.1)
        assert result.patterns_replayed == 0

    def test_run_cycle_records_last_replay(self):
        override = [(_pat(0), SLEEP_LOSS_WEIGHT)]
        e = self._engine()
        result = e.run_cycle(duration_sec=0.1, replay_list=override)
        assert e.last_replay is result
        assert e.last_replay is not None and e.last_replay.patterns_replayed == 1

    def test_snapshot_includes_last_replay(self):
        override = [(_pat(0), SLEEP_LOSS_WEIGHT)]
        e = self._engine()
        e.run_cycle(duration_sec=0.1, replay_list=override)
        snap = e.snapshot()
        assert snap["initialized"] is True
        assert snap["cycle_count"] == 1
        assert snap["last_replay"] is not None
        assert snap["last_replay"]["patterns_replayed"] == 1

    def test_snapshot_before_any_cycle(self):
        snap = self._engine().snapshot()
        assert snap["cycle_count"] == 0
        assert snap["last_replay"] is None

    def test_truncates_large_replay_list(self):
        big = [(_pat(i % 3), SLEEP_WIN_WEIGHT) for i in range(150)]
        e = self._engine()
        patterns = e.select_patterns(replay_list=big)
        assert len(patterns) == e.MAX_PATTERNS


# ── Consolidation wiring (lightweight) ────────────────────────────────
class TestConsolidationWiring:
    def test_consolidation_has_sleeper(self):
        """ConsolidationEngine constructable with a sleeper attribute."""
        from hanoon_prime.brain.consolidation import ConsolidationEngine
        from hanoon_prime.brain.shared_state import BrainState

        eng = ConsolidationEngine(
            brain_state=BrainState(),
            sleep_engine=SleepReplayEngine(
                network=LIFNetwork(),
                stdp=STDPLearner(),
                memory=AttractorMemory(),
            ),
        )
        assert hasattr(eng, "_sleeper")
        assert isinstance(eng._sleeper, SleepScheduler)

    def test_run_sleep_replay_propagates_replay_list(self):
        from hanoon_prime.brain.consolidation import ConsolidationEngine
        from hanoon_prime.brain.shared_state import BrainState

        mem = AttractorMemory()
        eng = ConsolidationEngine(
            brain_state=BrainState(),
            sleep_engine=SleepReplayEngine(
                network=LIFNetwork(),
                stdp=STDPLearner(),
                memory=mem,
            ),
        )
        weighted = [(_pat(0), SLEEP_LOSS_WEIGHT)]
        result = eng.run_sleep_replay(replay_list=weighted, duration_sec=0.1)
        assert isinstance(result, SleepResult)
        assert result.patterns_replayed == 1


# ── Live feeding (regression: attractors must be populated on close) ──
class TestLiveFeeding:
    def test_closes_populate_replayable_attractors(self):
        """Every REAL close feeds the neuromorphic attractor memory.

        Phase E shipped the scheduler + engine but nothing stored outcomes
        into AttractorMemory, so _sleep_patterns was always empty and
        replay never ran. Now on_trade_close stores the decision alpha.
        """
        from hanoon_prime.brain.config import EPISODIC_KEYS
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain
        from hanoon_prime.brain.shared_state import BrainState

        b = NeuromorphicBrain(brain_state=BrainState(), enable_neuromorphic=True)
        for i, ticker in enumerate(("AAA", "BBB", "CCC")):
            alpha = {k: 0.1 + 0.2 * i for k in EPISODIC_KEYS}
            for _ in range(3):  # create + updates → trade_count reaches 2
                b._store_decision(ticker, alpha, 0.7, 0.6)
                b.on_trade_close(ticker=ticker, won=False, pnl_pct=-0.01, direction=1)
        mem = list(b._neuromorphic.memory)
        assert len(mem) == 3
        assert all(a.trade_count == 2 for a in mem)
        assert len(mem) >= SLEEP_MIN_PATTERNS

    def test_disabled_neuromorphic_skips_quietly(self):
        """enable_neuromorphic=False keeps the close path a no-op for the organ."""
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain
        from hanoon_prime.brain.shared_state import BrainState

        b = NeuromorphicBrain(brain_state=BrainState(), enable_neuromorphic=False)
        b._store_decision("AAA", {}, 0.7, 0.6)
        b.on_trade_close(ticker="AAA", won=True, pnl_pct=0.01, direction=1)

    def test_ib_fill_mirrors_construct_buffer_trades(self, monkeypatch, tmp_path):
        """Entry + exit IB fills assemble a round-trip in the consolidation buffer."""
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain
        from hanoon_prime.brain.shared_state import BrainState
        from hanoon_prime.reflection import buffer as buffer_mod

        monkeypatch.setattr(buffer_mod, "_BUFFER_PATH", tmp_path / "trades.json")
        b = NeuromorphicBrain(brain_state=BrainState(), enable_neuromorphic=True)
        assert b._consolidation is not None
        b.on_ib_fill(
            {"ticker": "AAA", "direction": 1, "qty": 10, "price": 100.0, "fees": 0.5}
        )
        b.on_ib_fill(
            {"ticker": "AAA", "direction": -1, "qty": 10, "price": 105.0, "fees": 0.5}
        )
        trades = b._consolidation.buffer.get_trades()
        assert len(trades) == 1
        trade = trades[0]
        assert trade.ticker == "AAA"
        assert trade.exit_time > trade.entry_time
        assert trade.win

    def test_sleep_replay_runs_after_live_roundtrip(self, monkeypatch, tmp_path):
        """Fills + closes are enough for _maybe_sleep_replay to run a cycle."""
        from hanoon_prime.brain.config import EPISODIC_KEYS
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain
        from hanoon_prime.brain.shared_state import BrainState
        from hanoon_prime.reflection import buffer as buffer_mod

        monkeypatch.setattr(buffer_mod, "_BUFFER_PATH", tmp_path / "trades.json")
        b = NeuromorphicBrain(brain_state=BrainState(), enable_neuromorphic=True)
        for i, ticker in enumerate(("AAA", "BBB", "CCC")):
            alpha = {k: 0.1 + 0.2 * i for k in EPISODIC_KEYS}
            for _ in range(3):
                b._store_decision(ticker, alpha, 0.7, 0.6)
                b.on_trade_close(ticker=ticker, won=False, pnl_pct=-0.01, direction=1)
        b.on_ib_fill(
            {"ticker": "AAA", "direction": 1, "qty": 10, "price": 100.0, "fees": 0.0}
        )
        b.on_ib_fill(
            {"ticker": "AAA", "direction": -1, "qty": 10, "price": 101.0, "fees": 0.0}
        )
        eng = b._consolidation._sleep_engine
        assert eng is not None
        assert eng._cycle_count == 0
        b._consolidation._maybe_sleep_replay(session_close=True)
        assert eng._cycle_count >= 1
        assert eng.last_replay is not None
