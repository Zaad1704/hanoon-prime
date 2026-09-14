"""tests/test_metacog — confidence-of-confidence reliability, surprise, sizing.

Phase F of AWAKENED_BRAIN.md: MetaMonitor computes a rolling calibration
correlation, shrinks sizing when it degrades, detects novel situations,
and biases exploration/retreat from surprise + pillar health. Also
verifies orchestrator wiring (shared-state keys, size scaling).
"""

from __future__ import annotations

import pytest

from hanoon_prime.brain.config import EPISODIC_KEYS
from hanoon_prime.brain.episodic import EpisodicMemory
from hanoon_prime.brain.learning_config import (
    METACOG_BINS,
    METACOG_CURIOUS_SCALE,
    METACOG_MIN_SAMPLES,
    METACOG_RETREAT_SCALE,
    METACOG_SHRINK_BAD,
    METACOG_SHRINK_WEAK,
    METACOG_SURPRISE_THRESHOLD,
)
from hanoon_prime.brain.metacog import MetaMonitor, conf_bin


def _m() -> MetaMonitor:
    return MetaMonitor(persist=False)


def _alpha(vals: list[float] | None = None) -> dict[str, float]:
    base = [0.5] * len(EPISODIC_KEYS)
    for i, v in enumerate(vals or []):
        if i < len(base):
            base[i] = v
    return dict(zip(EPISODIC_KEYS, base))


# ── Binning ───────────────────────────────────────────────────────────
class TestConfBin:
    def test_low_conf_to_low_bin(self):
        assert conf_bin(0.05) == 0

    def test_mid_conf(self):
        assert conf_bin(0.5) == METACOG_BINS // 2

    def test_high_conf_to_top_bin(self):
        assert conf_bin(0.95) == METACOG_BINS - 1

    def test_clamped(self):
        assert conf_bin(-1.0) == 0
        assert conf_bin(2.0) == METACOG_BINS - 1


# ── Reliability / calibration ─────────────────────────────────────────
class TestReliability:
    def test_no_samples_defaults_reliable(self):
        assert _m().reliability() == 1.0

    def test_insufficient_samples_not_penalized(self):
        m = _m()
        for i in range(METACOG_MIN_SAMPLES - 1):
            m.update(conf=0.5 + 0.05 * i, won=True)
        assert m.reliability() == 1.0

    def test_well_calibrated_high_reliability(self):
        m = _m()
        # Higher confidence consistently correlates with wins.
        for bin_idx in range(METACOG_BINS):
            for _ in range(3):
                m.update(conf=(bin_idx + 0.5) / METACOG_BINS, won=bin_idx >= 3)
        assert m.reliability() > 0.85

    def test_inverted_calibration_low_reliability(self):
        m = _m()
        # High confidence correlates with losses.
        for bin_idx in range(METACOG_BINS):
            for _ in range(3):
                m.update(conf=(bin_idx + 0.5) / METACOG_BINS, won=bin_idx <= 1)
        assert m.reliability() < 0.35

    def test_reliability_normalized_to_unit_range(self):
        m = _m()
        for bin_idx in range(METACOG_BINS):
            for _ in range(8):
                m.update(conf=(bin_idx + 0.5) / METACOG_BINS, won=bin_idx % 2 == 0)
        assert 0.0 <= m.reliability() <= 1.0


# ── Sizing scaling ────────────────────────────────────────────────────
class TestSizingScalar:
    def test_reliable_full_size(self):
        m = _m()
        for i in range(20):
            m.update(conf=0.95, won=i % 2 == 0)
        # Half wins at same bin → near-zero correlation signal → mild shrink.
        assert 0.4 <= m.reliability() <= 0.6
        assert m.sizing_scalar() == METACOG_SHRINK_WEAK

    def test_unreliable_strong_shrink(self):
        m = _m()
        # Strongly anti-correlated: highs lose, lows win.
        for bin_idx in range(METACOG_BINS):
            for _ in range(4):
                m.update(conf=(bin_idx + 0.5) / METACOG_BINS, won=bin_idx == 0)
        assert m.sizing_scalar() == METACOG_SHRINK_BAD

    def test_excellent_calibration_full_size(self):
        m = _m()
        xs = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        for bin_idx in xs:
            m.update(conf=(bin_idx + 0.5) / METACOG_BINS, won=bin_idx >= 2)
        assert m.sizing_scalar() == 1.0


# ── Surprise / novelty ────────────────────────────────────────────────
class TestSurprise:
    def test_empty_memory_not_surprising(self):
        assert _m().surprise(_alpha(), EpisodicMemory()) == 0.0

    def test_novel_situation_flagged(self):
        mem = EpisodicMemory()
        for _ in range(10):
            mem.add(_alpha([1.0] * 4), outcome=1.0)
        assert _m().surprise(_alpha([0.0] * 4), mem) > 0.3

    def test_familiar_situation_not_surprising(self):
        mem = EpisodicMemory()
        for _ in range(10):
            mem.add(_alpha([0.51] * 4), outcome=1.0)
        assert _m().surprise(_alpha([0.5] * 4), mem) < 0.2


# ── Curiosity drive ───────────────────────────────────────────────────
class TestCuriosity:
    def test_no_drive_when_unsurprising(self):
        m = _m()
        assert m.curiosity_scale(0.2, "upright") == 1.0
        assert m.curiosity_scale(0.2, "fallen") == 1.0

    def test_explore_when_stable(self):
        m = _m()
        assert m.curiosity_scale(METACOG_SURPRISE_THRESHOLD + 0.1, "warming") == (
            METACOG_CURIOUS_SCALE
        )

    def test_retreat_when_falling(self):
        m = _m()
        for state in ("tipping", "fallen"):
            assert m.curiosity_scale(METACOG_SURPRISE_THRESHOLD + 0.1, state) == (
                METACOG_RETREAT_SCALE
            )


# ── Persistence ───────────────────────────────────────────────────────
class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        m = MetaMonitor(filepath=tmp_path / "m.json")
        m.update(0.9, True)
        m.update(0.2, False)
        data = m.save()
        n = MetaMonitor(persist=False)
        n.load(data)
        assert list(n._samples) == list(m._samples)

    def test_file_roundtrip(self, tmp_path):
        m = MetaMonitor(filepath=tmp_path / "m.json")
        m.update(0.9, True)
        m.update(0.2, False)
        n = MetaMonitor(filepath=tmp_path / "m.json")
        assert list(n._samples) == list(m._samples)

    def test_corrupt_file_starts_empty(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text("{not json")
        m = MetaMonitor(filepath=p)
        assert m.size == 0

    def test_clear_wipes(self, tmp_path):
        m = MetaMonitor(filepath=tmp_path / "m.json")
        m.update(0.9, True)
        m.clear()
        assert m.size == 0
        n = MetaMonitor(filepath=tmp_path / "m.json")
        assert n.size == 0


# ── Orchestrator wiring ───────────────────────────────────────────────
class TestOrchestratorWiring:
    def test_brain_initializes_meta_monitor(self):
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain
        from hanoon_prime.brain.shared_state import BrainState

        b = NeuromorphicBrain(brain_state=BrainState(), enable_neuromorphic=False)
        assert b._meta_cog is not None
        b._meta_cog.clear()
        assert b._meta_cog.size == 0

    def test_score_pipeline_exposes_surprise(self):
        from tests.test_brain_organs import _alpha, _brain

        b = _brain()
        ctx = b._score_pipeline("TEST", _alpha(), 1.0, 0.0, 0.0)
        assert "surprise" in ctx
        assert 0.0 <= ctx["surprise"] <= 1.0

    def test_learn_from_real_updates_calibration(self):
        from tests.test_brain_organs import _brain

        b = _brain()
        b._meta_cog.clear()
        before = b._meta_cog.size
        b._last_conf["TEST"] = 0.9
        b._learn_from_real(
            ticker="TEST",
            won=True,
            pnl_pct=0.05,
            direction=1,
            regime="bull",
            rpe_surprise=0.0,
        )
        assert b._meta_cog.size == before + 1
        b._meta_cog.clear()
