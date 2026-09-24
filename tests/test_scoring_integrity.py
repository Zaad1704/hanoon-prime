"""tests/test_scoring_integrity.py — scoring math + replay polarity regressions.

Regression: FIX-2026-09-23-06 (blend), FIX-2026-09-23-07 (penny bar),
FIX-2026-09-23-08 (sleep polarity).

Pre-fix defects (all now reversed):
- Score blending: with NEURO_BLEND_ENABLED=False, ``_score_pipeline`` still
  computed ``0.7*(base.score + cal_adj) + 0.3*0.0`` — a fixed 0.7x damping
  of every cortex score for zero neuro contribution. The penny gate, the
  dynamics threshold band, and ENTRY_THRESHOLD are all calibrated against
  un-damped cortex units, so the damping silently distorted every gate.
- Sleep replay: ``_replay_pattern`` inferred the reward sign from the drive
  magnitude (``1.0 if weight >= 1.0 else -1.0``). Losers replay at 3x
  drive, so EVERY loser was rewarded (+1.0) — sleep consolidated the
  losing behavior. Polarity is now explicit: ``(pattern, drive, won)``.
- PENNY_SCORE_BAR=0.85 is kept and re-derived against the un-damped
  scale: sub-dollar names face a bar strictly above the normal entry bar.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hanoon_prime.immune import ENTRY_THRESHOLD, PENNY_SCORE_BAR


def _bull_alpha() -> dict[str, float]:
    return {
        k: 0.9
        for k in (
            "vpin",
            "orderbook_imbalance",
            "institutional_flow",
            "momentum",
            "vwap_deviation",
        )
    }


# ── Blend formula (Regression: FIX-2026-09-23-06) ─────────────────────
class TestBlendFormula:
    def _blended_raw(self, monkeypatch, *, gate_on: bool, neuro_score: float) -> float:
        """Run _score_pipeline with every non-formula term neutralized.

        regime_mul=1, somatic=(0,0), mods=0, absorption=0, _stabilize echoes
        its input, so the captured ``raw`` IS the blended score exactly.
        """
        from hanoon_prime.brain import orchestrator as orch_mod
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain

        brain = NeuromorphicBrain()
        monkeypatch.setattr(orch_mod, "NEURO_BLEND_ENABLED", gate_on)
        monkeypatch.setattr(
            brain.cortex,
            "evaluate",
            lambda alpha, prior_top=None: SimpleNamespace(
                score=0.8, direction=1, confidence=0.7
            ),
        )
        monkeypatch.setattr(brain, "_calibration_nudge", lambda score: 0.05)
        monkeypatch.setattr(
            brain, "_compute_neuro_score", lambda alpha, ticker: neuro_score
        )
        monkeypatch.setattr(
            brain.nash,
            "predict",
            lambda alpha, score, direction: SimpleNamespace(
                confidence=0.5, win_prob=0.5
            ),
        )
        monkeypatch.setattr(brain, "_somatic_context", lambda: (0.0, 0.0))
        monkeypatch.setattr(brain, "_compute_mods", lambda *a: 0.0)
        monkeypatch.setattr(
            brain, "_absorption_score_mod", staticmethod(lambda *a: 0.0)
        )
        captured: dict[str, float] = {}

        def _spy_stabilize(raw: float, nash_pred):
            captured["raw"] = raw
            return raw, "test", 1

        monkeypatch.setattr(brain, "_stabilize", _spy_stabilize)
        brain._score_pipeline("T", _bull_alpha(), 1.0, 0.0, 0.0)
        assert "raw" in captured
        return captured["raw"]

    def test_gate_off_passes_cortex_through_undamped(self, monkeypatch):
        """Regression: FIX-2026-09-23-06. Gate off -> base.score + cal_adj.

        The old code returned 0.7*(0.8+0.05) = 0.595 here — a silent 30%
        haircut on every score for zero neuro contribution.
        """
        raw = self._blended_raw(monkeypatch, gate_on=False, neuro_score=0.0)
        assert raw == pytest.approx(0.85)
        assert raw != pytest.approx(0.7 * 0.85)

    def test_gate_on_restores_70_30_blend(self, monkeypatch):
        """Re-enabling the gate restores the 0.7/0.3 blend exactly."""
        raw = self._blended_raw(monkeypatch, gate_on=True, neuro_score=0.5)
        assert raw == pytest.approx(0.7 * 0.85 + 0.3 * 0.5)


# ── Penny bar (Regression: FIX-2026-09-23-07) ─────────────────────────
class TestPennyBar:
    def test_penny_bar_value_preserved(self):
        """0.85 kept — change detector: any retune needs a new derivation.

        Re-derived 2026-09-23 against the un-damped cortex scale
        (tanh units, |score| < 1): atanh(0.85) ~= 1.26 sigma of conviction
        surviving the Nash penalty. Lowering this to admit more
        sub-dollar trades must be a deliberate, documented decision.
        """
        assert PENNY_SCORE_BAR == 0.85

    def test_penny_bar_above_normal_entry_threshold(self):
        """Sub-dollar names always face a bar regular names never see."""
        assert PENNY_SCORE_BAR > ENTRY_THRESHOLD

    def test_penny_gate_receives_thought_score(self, monkeypatch):
        """The gate is fed float(thought.score) — the blended score."""
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain

        brain = NeuromorphicBrain()
        seen: dict[str, tuple] = {}

        class _Policy:
            def is_session_active(self, session):
                return True

            def is_direction_allowed(self, side):
                return True

            def is_penny_bar_cleared(self, ticker, last, score):
                seen["args"] = (ticker, last, score)
                return True, ""

        monkeypatch.setattr(brain, "trading_policy", _Policy())
        monkeypatch.setattr(brain, "_absorption_flow_gate", lambda *a: None)
        thought = SimpleNamespace(direction=1, score=0.42, confidence=0.7)
        brain._apply_fast_gates(
            "T", {"last": 0.5}, thought, {"authorized": True}, "rth"
        )
        assert seen["args"][2] == 0.42


# ── Sleep replay polarity (Regression: FIX-2026-09-23-08) ────────────
class TestSleepPolarity:
    def _engine(self):
        from hanoon_prime.brain.neurons.attractor import AttractorMemory
        from hanoon_prime.brain.neurons.network import LIFNetwork
        from hanoon_prime.brain.neurons.sleep import SleepReplayEngine
        from hanoon_prime.brain.neurons.stdp import STDPLearner

        return SleepReplayEngine(
            network=LIFNetwork(),
            stdp=STDPLearner(),
            memory=AttractorMemory(),
        )

    def test_loser_replay_applies_negative_reward(self, monkeypatch):
        """Regression: FIX-2026-09-23-08. The crux: a loser replays at
        3x drive (weight 3.0 >= 1.0), which the old sign inference read
        as a WIN (+1.0). Polarity is now explicit — losers get -1.0.
        """
        engine = self._engine()
        # Bare LIFNetwork: encode_pattern maps onto existing input neurons
        # only, so the pattern may be empty here. That is fine — the reward
        # sign under test does not depend on spikes, only on the explicit
        # ``won`` polarity threaded into _replay_pattern.
        pattern = engine.encode_pattern([0.9] * 11)
        rewards: list[float] = []
        monkeypatch.setattr(engine._stdp, "apply_reward", lambda r: rewards.append(r))
        engine._replay_pattern(pattern, 3.0, False)
        assert rewards == [-1.0]

    def test_winner_replay_applies_positive_reward(self, monkeypatch):
        engine = self._engine()
        pattern = engine.encode_pattern([0.9] * 11)
        rewards: list[float] = []
        monkeypatch.setattr(engine._stdp, "apply_reward", lambda r: rewards.append(r))
        engine._replay_pattern(pattern, 1.0, True)
        assert rewards == [1.0]

    def test_run_cycle_threads_polarity_end_to_end(self, monkeypatch):
        """A mixed (loser, winner) replay list yields (-1.0, +1.0)."""
        engine = self._engine()
        pattern = engine.encode_pattern([0.9] * 11)
        rewards: list[float] = []
        monkeypatch.setattr(engine._stdp, "apply_reward", lambda r: rewards.append(r))
        result = engine.run_cycle(
            duration_sec=1.0,
            replay_list=[(pattern, 3.0, False), (pattern, 1.0, True)],
        )
        assert result.patterns_replayed == 2
        assert rewards == [-1.0, 1.0]
