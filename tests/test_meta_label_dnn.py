"""Tests for brain.meta_label_dnn — deep meta-labeler gatekeeper."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hanoon_prime.brain.learning_config import (
    META_CUT,
    META_DNN_HIDDEN,
    META_WIN_THRESHOLD,
)
from hanoon_prime.brain.meta_label_dnn import (
    MetaDNN,
    calculate_meta_size_scale,
    expand_features,
)

# ── Feature expansion ────────────────────────────────────────────────


class TestExpandFeatures:
    def test_base_length(self):
        """Expanded vector has 9 elements (one-hots pruned, MTF appended)."""
        feats = expand_features(0.8, 0.5, 0.3)
        assert len(feats) == 9

    def test_direction_default(self):
        """Direction defaults to 1."""
        feats = expand_features(0.5, 0.5, 0.5)
        assert feats[3] == 1.0

    def test_direction_short(self):
        """Direction=-1 propagates."""
        feats = expand_features(0.5, 0.5, 0.5, direction=-1)
        assert feats[3] == -1.0

    def test_score_clamped(self):
        """|score| is clamped to [0, 1]."""
        feats = expand_features(0.5, 5.0, 0.5)
        assert feats[1] == pytest.approx(1.0)
        feats = expand_features(0.5, -5.0, 0.5)
        assert feats[1] == pytest.approx(1.0)

    def test_extra_features(self):
        """ATR ratio, OBI, VPIN propagate."""
        feats = expand_features(0.5, 0.5, 0.5, atr_ratio=0.15, obi=0.3, vpin=0.7)
        assert feats[4] == pytest.approx(0.15)
        assert feats[5] == pytest.approx(0.3)
        assert feats[6] == pytest.approx(0.7)

    def test_mtf_features(self):
        """MTF features propagate: tf5_align and tf15_vol."""
        feats = expand_features(0.5, 0.5, 0.5, tf5_align=0.42, tf15_vol=1.7)
        assert feats[7] == pytest.approx(0.42)
        assert feats[8] == pytest.approx(1.7)
        neutral = expand_features(0.5, 0.5, 0.5)
        assert neutral[7] == pytest.approx(0.0)
        assert neutral[8] == pytest.approx(1.0)


# ── MLP core ─────────────────────────────────────────────────────────


class TestMetaDNN:
    def test_build_creates_layers(self):
        """MetaDNN builds weight matrices on first forward pass."""
        model = MetaDNN(path=Path("/tmp/_test_dnn_build.json"))
        model._build()
        assert len(model._layers) == len(META_DNN_HIDDEN) + 1
        assert model._layers[0][0].shape == (9, META_DNN_HIDDEN[0])

    def test_forward_shape(self):
        """Forward pass returns scalar output."""
        model = MetaDNN(path=Path("/tmp/_test_dnn_fwd.json"))
        x = np.zeros((1, 9), dtype=np.float64)
        out, cache = model.forward(x)
        assert out.shape == (1, 1)
        assert 0.0 <= float(out[0, 0]) <= 1.0

    def test_predict_returns_float(self):
        """predict() returns a float in [0, 1]."""
        model = MetaDNN(path=Path("/tmp/_test_dnn_pred.json"))
        feats = [0.5] * 9
        p = model.predict(feats)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_infer_admit_above_threshold(self):
        """High-confidence model admits when P(Win) >= threshold."""
        model = MetaDNN(path=Path("/tmp/_test_dnn_infer.json"))
        # Force high weights on first layer to get strong output
        rng = np.random.default_rng(42)
        w0 = rng.normal(5.0, 0.1, (9, META_DNN_HIDDEN[0]))
        b0 = np.zeros(META_DNN_HIDDEN[0])
        w1 = rng.normal(5.0, 0.1, (META_DNN_HIDDEN[0], META_DNN_HIDDEN[1]))
        b1 = np.zeros(META_DNN_HIDDEN[1])
        w2 = rng.normal(5.0, 0.1, (META_DNN_HIDDEN[1], 1))
        b2 = np.zeros(1)
        model._layers = [
            (w0.astype(np.float64), b0),
            (w1.astype(np.float64), b1),
            (w2.astype(np.float64), b2),
        ]
        model._built = True
        admit, p, scale = model.infer([0.9] * 9)
        assert admit is True
        assert p >= META_WIN_THRESHOLD

    def test_infer_veto_below_threshold(self):
        """Low-confidence model vetoes when P(Win) < threshold."""
        model = MetaDNN(path=Path("/tmp/_test_dnn_veto.json"))
        rng = np.random.default_rng(99)
        w0 = rng.normal(-5.0, 0.1, (9, META_DNN_HIDDEN[0]))
        b0 = np.zeros(META_DNN_HIDDEN[0])
        w1 = rng.normal(-5.0, 0.1, (META_DNN_HIDDEN[0], META_DNN_HIDDEN[1]))
        b1 = np.zeros(META_DNN_HIDDEN[1])
        w2 = rng.normal(-5.0, 0.1, (META_DNN_HIDDEN[1], 1))
        b2 = np.zeros(1)
        model._layers = [
            (w0.astype(np.float64), b0),
            (w1.astype(np.float64), b1),
            (w2.astype(np.float64), b2),
        ]
        model._built = True
        admit, p, scale = model.infer([0.1] * 9)
        assert admit is False
        assert p < META_WIN_THRESHOLD
        assert scale == 0.0  # de Prado: below threshold -> zero allocation

    def test_size_scale_monotonic(self):
        """size_scale increases as P(Win) approaches threshold."""
        model = MetaDNN(path=Path("/tmp/_test_dnn_mono.json"))
        # Cold models bypass to admittance; build so infer() consults predict().
        model._built = True
        model._layers = [(np.ones((9, 1)), np.zeros(1))]
        # Use fake predict to test scale calculation directly
        original_predict = model.predict
        model.predict = lambda feats: 0.3  # below threshold
        _, _, s1 = model.infer([0.0] * 9)
        model.predict = lambda feats: 0.45  # still below threshold
        _, _, s2 = model.infer([0.0] * 9)
        model.predict = lambda feats: 0.60  # above threshold
        _, _, s3 = model.infer([0.0] * 9)
        model.predict = original_predict
        # Below threshold: both zero; above threshold: positive scale
        assert s1 == 0.0
        assert s2 == 0.0
        assert s3 > 0.0

    def test_train_reduces_loss(self):
        """Training loop reduces BCE loss over epochs."""
        model = MetaDNN(path=Path("/tmp/_test_dnn_train.json"))
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (200, 9)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.float64)
        losses = model.train(X, y, epochs=50, lr=0.002, batch_size=32)
        assert len(losses) == 50
        # Second half average should be lower than first half
        first_half = sum(losses[:25]) / 25
        second_half = sum(losses[25:]) / 25
        assert second_half < first_half

    def test_save_load_roundtrip(self, tmp_path: Path):
        """Persisted model loads identical weights."""
        p = tmp_path / "dnn_rt.json"
        model = MetaDNN(path=p)
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 9)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.float64)
        model.train(X, y, epochs=5, lr=0.01, batch_size=16)
        feats = rng.normal(0, 1, 9).tolist()
        p1 = model.predict(feats)
        model2 = MetaDNN(path=p)
        p2 = model2.predict(feats)
        assert p1 == pytest.approx(p2, abs=1e-6)

    def test_load_corrupt_file(self, tmp_path: Path):
        """Corrupt file → fresh model, no crash."""
        p = tmp_path / "bad.json"
        p.write_text("NOT VALID JSON {{{")
        model = MetaDNN(path=p)
        assert not model._built  # weights not loaded

    def test_load_missing_file(self, tmp_path: Path):
        """Missing file → fresh model, no crash."""
        model = MetaDNN(path=tmp_path / "nonexistent.json")
        assert not model._built


# ── Gatekeeper integration with MetaLabelModel ────────────────────────


class TestMetaLabelGate:
    def test_gate_returns_valid_when_dnn_enabled(self, tmp_path: Path):
        """With DNN enabled, gate() returns valid (admit, p, scale)."""
        from hanoon_prime.brain.meta_label import MetaLabelModel

        model = MetaLabelModel(path=tmp_path / "meta.json")
        admit, p, scale = model.gate(0.5, 0.5, 0.3, "range", "scalp")
        assert isinstance(admit, bool)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
        assert 0.5 <= scale <= 1.0

    def test_gate_fallback_when_dnn_disabled(self, tmp_path: Path):
        """When META_DNN_ENABLED is False, gate() falls back to shallow logistic."""
        import hanoon_prime.brain.learning_config as lc
        import hanoon_prime.brain.meta_label as ml

        old = lc.META_DNN_ENABLED
        try:
            lc.META_DNN_ENABLED = False
            ml.META_DNN_ENABLED = False
            model = ml.MetaLabelModel(path=tmp_path / "meta.json")
            admit, p, scale = model.gate(0.5, 0.5, 0.3, "range", "scalp")
            assert admit is True
            assert isinstance(p, float)
            assert 0.0 <= p <= 1.0
        finally:
            lc.META_DNN_ENABLED = old
            ml.META_DNN_ENABLED = old

    def test_gate_fallback_size_matches(self, tmp_path: Path):
        """Fallback gate size matches size_scalar when META_MIN_SAMPLES met."""
        import hanoon_prime.brain.learning_config as lc
        import hanoon_prime.brain.meta_label as ml

        old = lc.META_DNN_ENABLED
        try:
            lc.META_DNN_ENABLED = False
            ml.META_DNN_ENABLED = False
            model = ml.MetaLabelModel(path=tmp_path / "meta.json")
            fv = ml.feature_vector(0.5, 0.5, 0.3, "range", "scalp")
            for _ in range(25):
                model.record(fv, won=True)
            s1 = model.size_scalar(0.5, 0.5, 0.3, "range", "scalp")
            _, _, s2 = model.gate(0.5, 0.5, 0.3, "range", "scalp")
            assert s2 == pytest.approx(s1, abs=0.01)
        finally:
            lc.META_DNN_ENABLED = old
            ml.META_DNN_ENABLED = old


# ── Dynamic Kelly bet sizing ────────────────────────────────────────


class TestCalculateMetaSizeScale:
    def test_below_threshold(self):
        """P(Win) below threshold returns 0.0 (vetoed)."""
        assert calculate_meta_size_scale(0.519) == 0.0
        assert calculate_meta_size_scale(0.0) == 0.0
        assert calculate_meta_size_scale(-0.1) == 0.0

    def test_at_threshold(self):
        """P(Win) at threshold returns 0.0 (marginal admission)."""
        assert calculate_meta_size_scale(0.52) == 0.0

    def test_half_allocation(self):
        """P(Win) = 0.76 returns 0.5 (half allocation)."""
        assert calculate_meta_size_scale(0.76) == pytest.approx(0.5, abs=0.001)

    def test_full_allocation(self):
        """P(Win) = 1.0 returns 1.0 (full allocation)."""
        assert calculate_meta_size_scale(1.0) == 1.0

    def test_monotonic(self):
        """Size scale is strictly monotonically increasing above threshold."""
        scales = [
            calculate_meta_size_scale(p) for p in [0.52, 0.60, 0.70, 0.80, 0.90, 1.0]
        ]
        assert scales == sorted(scales)
        assert all(s > 0 for s in scales[1:])

    def test_custom_threshold(self):
        """Custom threshold shifts the scale function."""
        assert calculate_meta_size_scale(0.60, threshold=0.55) == pytest.approx(
            (0.60 - 0.55) / (1.0 - 0.55), abs=0.001
        )
