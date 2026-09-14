"""tests/test_rpe — multi-timescale dopamine reward-prediction error.

Phase A of AWAKENED_BRAIN.md: phasic/tonic/meta value channels with signed
prediction error, a bounded surprise-driven learning-rate modulator, and
survival across restarts.
"""

from __future__ import annotations

from hanoon_prime.brain.learning_config import RPE_LR_MAX, RPE_LR_MIN
from hanoon_prime.brain.rpe import MultiTimescaleRPE, lr_modulator


def _fresh() -> MultiTimescaleRPE:
    return MultiTimescaleRPE(persist=False)


# ── lr_modulator (surprise → learning-rate multiplier) ────────────────
class TestLrModulator:
    def test_zero_surprise_is_neutral(self):
        assert lr_modulator(0.0) == 1.0

    def test_full_surprise_hits_ceiling(self):
        assert lr_modulator(1.0) == RPE_LR_MAX

    def test_bounded_never_escapes(self):
        for s in (-3.0, -1.0, 0.0, 0.3, 2.0, 7.0):
            assert RPE_LR_MIN <= lr_modulator(s) <= RPE_LR_MAX

    def test_negative_surprise_uses_magnitude(self):
        assert lr_modulator(-0.5) == lr_modulator(0.5)


# ── Channel semantics ─────────────────────────────────────────────────
class TestFirstUpdate:
    def test_prior_seeds_channels(self):
        res = _fresh().update(0.6, True, "trend")
        assert res["phasic"] == 0.4
        assert res["tonic"] == 0.4
        assert res["v_fast"] == 0.732  # 0.6 + 0.33 * 0.4
        assert res["v_slow"] == 0.608  # 0.6 + 0.02 * 0.4

    def test_miss_surprise_is_negative(self):
        res = _fresh().update(0.7, False, "trend")
        assert res["phasic"] == -0.7
        assert res["surprise"] == 0.7


class TestSignFlip:
    def test_loss_after_win_negates_phasic(self):
        rpe = _fresh()
        rpe.update(0.6, True, "trend")
        assert rpe.update(0.6, False, "trend")["phasic"] < 0.0
        assert rpe.tonic_rpe < 0.0

    def test_win_after_loss_positives_phasic(self):
        rpe = _fresh()
        rpe.update(0.5, False, "trend")
        assert rpe.update(0.5, True, "trend")["phasic"] > 0.0
        assert rpe.phasic_rpe > 0.0


class TestSurprise:
    def test_decays_on_consistent_wins(self):
        rpe = _fresh()
        first = rpe.update(0.5, True, "trend")["surprise"]
        last = first
        for _ in range(9):
            last = rpe.update(0.5, True, "trend")["surprise"]
        assert last < first
        assert rpe.surprise < first

    def test_grows_after_flip(self):
        rpe = _fresh()
        for _ in range(10):
            rpe.update(0.5, True, "trend")
        calm = rpe.surprise
        shocked = rpe.update(0.5, False, "trend")
        assert shocked["surprise"] > calm
        assert shocked["surprise"] > 0.4


class TestTimescales:
    def test_fast_tracks_slower_memory(self):
        rpe = _fresh()
        for _ in range(20):
            rpe.update(0.5, True, "trend")
        snap = rpe.snapshot()
        assert snap["v_fast"] > snap["v_slow"]

    def test_meta_is_per_regime(self):
        rpe = _fresh()
        for _ in range(4):
            rpe.update(0.5, True, "trend")
        assert "choppy" not in rpe.snapshot()["v_meta"]
        res = rpe.update(0.5, False, "choppy")
        assert res["meta"] == -0.5  # untouched prior: 0.5
        assert rpe.snapshot()["v_meta"]["choppy"] == 0.475

    def test_count_accumulates(self):
        rpe = _fresh()
        for i in range(5):
            rpe.update(0.5, i % 2 == 0, "trend")
        assert rpe.count == 5


# ── Persistence ───────────────────────────────────────────────────────
class TestPersistence:
    def test_roundtrip_restores_channels(self, tmp_path):
        path = tmp_path / "rpe.json"
        first = MultiTimescaleRPE(filepath=path)
        first.update(0.6, True, "trend")
        first.update(0.6, False, "trend")
        second = MultiTimescaleRPE(filepath=path)
        # Value estimates + count persist; the last-error fields are
        # intentionally in-memory only (they describe the last trigger).
        for key in ("count", "v_fast", "v_slow", "v_meta"):
            assert second.snapshot()[key] == first.snapshot()[key]
        assert second.count == 2

    def test_missing_file_starts_fresh(self, tmp_path):
        rpe = MultiTimescaleRPE(filepath=tmp_path / "absent.json")
        assert rpe.count == 0
        assert rpe.snapshot()["v_fast"] == 0.5
        assert rpe.snapshot()["v_slow"] == 0.5

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "rpe.json"
        path.write_text("not json")
        rpe = MultiTimescaleRPE(filepath=path)
        assert rpe.count == 0


# ── Snapshot shape ────────────────────────────────────────────────────
class TestSnapshot:
    def test_shape(self):
        rpe = _fresh()
        rpe.update(0.5, True, "trend")
        assert set(rpe.snapshot()) == {
            "count",
            "v_fast",
            "v_slow",
            "v_meta",
            "phasic_rpe",
            "tonic_rpe",
        }
