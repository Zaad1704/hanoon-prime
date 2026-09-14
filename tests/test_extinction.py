"""tests/test_extinction — context-gated inhibition + episodic context tags.

Phase D of AWAKENED_BRAIN.md: patterns degrade inside a context → an
inhibition weight grows; retrieval is excitation − inhibition (both
context-gated); a regime re-entry renews the suppressed traces.
"""

from __future__ import annotations

import json

import pytest

from hanoon_prime.brain.config import EPISODIC_KEYS
from hanoon_prime.brain.episodic import EpisodicMemory
from hanoon_prime.brain.extinction import ExtinctionTracker, conf_bin_label, context_key
from hanoon_prime.brain.learning_config import (
    EXTINCT_DECAY,
    EXTINCT_MAX,
    EXTINCT_MIN_PATTERNS,
    EXTINCT_STEP,
)
from hanoon_prime.brain.telemetry_summaries import extinction_summary

REGIME = "trend"
CONF = 0.8
HORIZON = "swing"


def _alpha(over=()) -> dict:
    """11-key alpha vector; optional (index, value) overrides for neighbors."""
    vec = {k: 0.5 for k in EPISODIC_KEYS}
    for idx, value in over:
        vec[EPISODIC_KEYS[idx]] = value
    return vec


def _fresh() -> ExtinctionTracker:
    return ExtinctionTracker(persist=False)


def _lose(tracker, n: int = EXTINCT_MIN_PATTERNS) -> float:
    """Record ``n`` consecutive losses in the default context; return inhibition."""
    for _ in range(n):
        tracker.record(_alpha(), -0.05, REGIME, CONF, HORIZON)
    return tracker.inhibition(_alpha(), REGIME, CONF, HORIZON)


# ── Context tagging helpers ───────────────────────────────────────────
class TestContextTags:
    def test_conf_bin_label_buckets_low_mid_high(self):
        assert conf_bin_label(0.50) == "low"
        assert conf_bin_label(0.60) == "mid"
        assert conf_bin_label(0.90) == "high"

    def test_context_key_folds_conf_into_bucket(self):
        assert context_key("trend", 0.9, "scalp") == "trend|high|scalp"

    def test_context_key_accepts_string_conf(self):
        assert context_key("range", "mid", "swing") == "range|mid|swing"


# ── Core extinction dynamics ──────────────────────────────────────────
class TestInhibitionGrowth:
    def test_no_inhibition_before_min_patterns(self):
        t = _fresh()
        for _ in range(EXTINCT_MIN_PATTERNS - 1):
            t.record(_alpha(), -0.05, REGIME, CONF, HORIZON)
        assert t.inhibition(_alpha(), REGIME, CONF, HORIZON) == 0.0

    def test_repeated_losses_grow_inhibition(self):
        t = _fresh()
        _lose(t)
        assert t.inhibition(_alpha(), REGIME, CONF, HORIZON) == pytest.approx(
            EXTINCT_STEP
        )

    def test_record_returns_updated_inhibition(self):
        t = _fresh()
        last = 0.0
        for _ in range(EXTINCT_MIN_PATTERNS):
            last = t.record(_alpha(), -0.05, REGIME, CONF, HORIZON)
        assert last == pytest.approx(EXTINCT_STEP)

    def test_good_outcomes_decay_inhibition(self):
        t = _fresh()
        _lose(t)
        for _ in range(5):
            t.record(_alpha(), 0.03, REGIME, CONF, HORIZON)
        assert t.inhibition(_alpha(), REGIME, CONF, HORIZON) < EXTINCT_STEP

    def test_inhibition_floor_at_zero(self):
        t = _fresh()
        _lose(t)
        for _ in range(200):
            t.record(_alpha(), 0.03, REGIME, CONF, HORIZON)
        assert t.inhibition(_alpha(), REGIME, CONF, HORIZON) == 0.0

    def test_inhibition_capped_at_extinct_max(self):
        t = _fresh()
        for _ in range(400):
            t.record(_alpha(), -0.05, REGIME, CONF, HORIZON)
        assert t.inhibition(_alpha(), REGIME, CONF, HORIZON) == EXTINCT_MAX


# ── Context gating ────────────────────────────────────────────────────
class TestContextGating:
    def test_inhibition_is_context_local(self):
        t = _fresh()
        _lose(t)
        assert t.inhibition(_alpha(), "range", CONF, HORIZON) == 0.0
        assert t.inhibition(_alpha(), REGIME, 0.4, HORIZON) == 0.0
        assert t.inhibition(_alpha(), REGIME, CONF, "scalp") == 0.0

    def test_neighbor_overlap_scores_partial_inhibition(self):
        t = _fresh()
        _lose(t)
        near = _alpha(over=((0, 0.0),))
        value = t.inhibition(near, REGIME, CONF, HORIZON)
        assert 0.0 < value < EXTINCT_STEP + 1e-9

    def test_full_flip_gets_no_inhibition(self):
        t = _fresh()
        _lose(t)
        far = _alpha(over=tuple((i, 0.0 if i % 2 == 0 else 1.0) for i in range(11)))
        assert t.inhibition(far, REGIME, CONF, HORIZON) == 0.0

    def test_two_shared_dims_scale_partial_inhibition(self):
        t = _fresh()
        _lose(t)
        overlap2 = _alpha(over=tuple((i, 1.0) for i in range(9)))
        full = t.inhibition(_alpha(), REGIME, CONF, HORIZON)
        assert t.inhibition(overlap2, REGIME, CONF, HORIZON) == pytest.approx(
            full * 2 / 11
        )


# ── Renewal: regime re-entry ──────────────────────────────────────────
class TestReactivation:
    def test_reactivate_clears_matching_regime_only(self):
        t = _fresh()
        _lose(t)
        for _ in range(EXTINCT_MIN_PATTERNS):
            t.record(_alpha(), -0.05, "range", CONF, HORIZON)
        cleared = t.reactivate("range")
        assert cleared == 1
        assert t.inhibition(_alpha(), "range", CONF, HORIZON) == 0.0
        assert t.inhibition(_alpha(), REGIME, CONF, HORIZON) > 0.0

    def test_reactivate_returns_zero_when_nothing_suppressed(self):
        t = _fresh()
        t.record(_alpha(), 0.03, REGIME, CONF, HORIZON)
        assert t.reactivate(REGIME) == 0

    def test_reactivate_miss_does_not_touch_other_regimes(self):
        t = _fresh()
        _lose(t)
        t.reactivate("other_regime")
        assert t.inhibition(_alpha(), REGIME, CONF, HORIZON) > 0.0


# ── Persistence + bounds ──────────────────────────────────────────────
class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        t = _fresh()
        _lose(t)
        data = t.save()
        back = ExtinctionTracker(filepath=tmp_path / "noop.json", persist=False)
        back.load(data)
        assert back.inhibition(_alpha(), REGIME, CONF, HORIZON) == pytest.approx(
            EXTINCT_STEP
        )
        assert back.size == t.size == 1

    def test_restore_via_file_roundtrip(self, tmp_path):
        path = tmp_path / "extinct.json"
        t = ExtinctionTracker(filepath=path, persist=True)
        _lose(t)
        restored = ExtinctionTracker(filepath=path, persist=False)
        assert restored.inhibition(_alpha(), REGIME, CONF, HORIZON) == pytest.approx(
            EXTINCT_STEP
        )

    def test_corrupt_persist_starts_empty(self, tmp_path):
        path = tmp_path / "extinct.json"
        path.write_text("{not json")
        t = ExtinctionTracker(filepath=path, persist=False)
        assert t.size == 0
        assert t.inhibition(_alpha(), REGIME, CONF, HORIZON) == 0.0

    def test_clear_wipes_cells(self):
        t = _fresh()
        _lose(t)
        t.clear()
        assert t.size == 0
        assert t.inhibition(_alpha(), REGIME, CONF, HORIZON) == 0.0


class TestBounds:
    def test_inhibition_never_negative_or_over_max(self, tmp_path):
        t = ExtinctionTracker(filepath=tmp_path / "b.json", persist=False)
        for i in range(200):
            outcome = -0.02 if i % 3 == 0 else 0.01
            t.record(_alpha(), outcome, REGIME, CONF, HORIZON)
        value = t.inhibition(_alpha(), REGIME, CONF, HORIZON)
        assert 0.0 - 1e-12 <= value <= EXTINCT_MAX + 1e-12

    def test_decay_is_bounded_step(self):
        t = _fresh()
        _lose(t)
        t.record(_alpha(), 0.03, REGIME, CONF, HORIZON)
        t.record(_alpha(), 0.03, REGIME, CONF, HORIZON)
        before = t.inhibition(_alpha(), REGIME, CONF, HORIZON)
        t.record(_alpha(), 0.03, REGIME, CONF, HORIZON)
        assert before - t.inhibition(_alpha(), REGIME, CONF, HORIZON) == pytest.approx(
            EXTINCT_DECAY
        )

    def test_save_is_json_serializable(self):
        t = _fresh()
        _lose(t)
        json.dumps(t.save())  # must not raise


# ── Episodic context tags ─────────────────────────────────────────────
class TestEpisodicContextTags:
    def test_add_stores_context_and_clear_resets(self):
        mem = EpisodicMemory()
        mem.add(_alpha(), 0.02, "trend|high|scalp")
        assert mem._contexts[0] == "trend|high|scalp"
        mem.clear()
        assert all(c == "" for c in mem._contexts)

    def test_context_filter_selects_matching_neighbors(self):
        mem = EpisodicMemory()
        for _ in range(12):
            mem.add(_alpha(), 0.02, "trend|high|scalp")
        for _ in range(12):
            mem.add(_alpha(), -0.03, "range|low|swing")
        pos, _ = mem.predict(_alpha(), context="trend|high|scalp")
        assert pos == pytest.approx(0.02)
        neg, _ = mem.predict(_alpha(), context="range|low|swing")
        assert neg == pytest.approx(-0.03)

    def test_sparse_context_falls_back_to_full_buffer(self):
        mem = EpisodicMemory()
        for _ in range(12):
            mem.add(_alpha(), 0.02, "trend|high|scalp")
        mem.add(_alpha(), 0.02, "lonely|mid|scalp")
        pos, _ = mem.predict(_alpha(), context="lonely|mid|scalp")
        assert pos == pytest.approx(0.02)

    def test_modifier_routes_context(self):
        mem = EpisodicMemory()
        for _ in range(12):
            mem.add(_alpha(), 0.02, "trend|high|scalp")
        for _ in range(12):
            mem.add(_alpha(), -0.03, "range|low|swing")
        assert mem.modifier(_alpha(), "trend|high|scalp") > 0.0
        assert mem.modifier(_alpha(), "range|low|swing") < 0.0


# ── Telemetry snapshot ────────────────────────────────────────────────
class TestSnapshot:
    def _snap(self) -> dict:
        return extinction_summary(_fresh().save().get("cells", []))

    def test_empty_tracker_snapshot(self):
        snap = self._snap()
        assert snap["size"] == 0
        assert snap["contexts"] == 0
        assert snap["inhibited"] == 0
        assert snap["total_patterns"] == 0
        assert snap["cells"] == []

    def test_snapshot_counts_and_sorts_cells(self):
        t = _fresh()
        _lose(t)
        t.record(_alpha([(1, 0.2)]), 0.02, REGIME, CONF, HORIZON)  # untouched cell
        snap = extinction_summary(t.save().get("cells", []))
        assert snap["size"] == 2
        assert snap["contexts"] == 1
        assert snap["inhibited"] == 1
        assert snap["total_patterns"] == EXTINCT_MIN_PATTERNS + 1
        # Strongest-active cell (highest inhibition) lands first.
        assert len(snap["cells"]) == 2
        assert snap["cells"][0]["inhibition"] >= snap["cells"][1]["inhibition"]

    def test_snapshot_is_json_serializable(self):
        t = _fresh()
        _lose(t)
        json.dumps(extinction_summary(t.save().get("cells", [])))  # must not raise
