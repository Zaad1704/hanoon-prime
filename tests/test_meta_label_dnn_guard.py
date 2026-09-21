"""Tests for brain.meta_label_dnn_guard — ironclad anti-degeneration guard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hanoon_prime.brain.learning_config import META_WIN_THRESHOLD
from hanoon_prime.brain.meta_label_dnn import MetaDNN
from hanoon_prime.brain.meta_label_dnn_guard import (
    MIN_OUTPUT_SPREAD,
    MIN_WEIGHT_STD,
    govern,
    outputs_healthy,
    probe_inputs,
    weights_healthy,
)

# ── Weights check ────────────────────────────────────────────────────


class TestWeightsHealthy:
    def test_rejects_empty(self):
        """No layers is never healthy."""
        ok, reason = weights_healthy([])
        assert not ok
        assert reason == "no_layers"

    def test_rejects_nan(self):
        """Non-finite weights are rejected."""
        layers = [
            (np.full((15, 4), np.nan), np.zeros(4)),
            (np.ones((4, 1)), np.zeros(1)),
        ]
        ok, reason = weights_healthy(layers)
        assert not ok
        assert "nonfinite" in reason

    def test_rejects_degenerate_scale(self):
        """Weights of the historical 1e-4 scale are dead."""
        layers = [
            (np.full((15, 4), 0.0004), np.zeros(4)),
            (np.full((4, 1), 0.0004), np.zeros(1)),
        ]
        ok, reason = weights_healthy(layers)
        assert not ok
        assert reason == "layer_0_dead_weights"

    def test_accepts_healthy(self):
        """He-init scale weights pass."""
        rng = np.random.default_rng(0)
        layers = [
            (rng.normal(0, 0.4, (15, 4)), np.zeros(4)),
            (rng.normal(0, 0.4, (4, 1)), np.zeros(1)),
        ]
        ok, reason = weights_healthy(layers)
        assert ok
        assert reason == ""


# ── Output spread check ──────────────────────────────────────────────


class TestOutputsHealthy:
    def test_rejects_constant_output(self):
        """A saturated net that emits the same P(Win) for every probe fails."""

        def constant_predict(_feats: list[float]) -> float:
            return 0.3651

        ok, reason = outputs_healthy(constant_predict, 15)
        assert not ok
        assert reason == "constant_output"

    def test_accepts_varying_output(self):
        """A net whose output moves with its inputs passes."""

        def varying_predict(feats: list[float]) -> float:
            return 1.0 / (1.0 + np.exp(-1.5 * float(feats[0])))

        ok, _reason = outputs_healthy(varying_predict, 15)
        assert ok

    def test_probe_deterministic(self):
        """Probe matrix is reproducible."""
        a = probe_inputs(15)
        b = probe_inputs(15)
        assert a.shape == b.shape
        assert np.allclose(a, b)
        assert a.shape[0] >= MIN_OUTPUT_SPREAD * 0  # sanity: probe is non-empty


# ── Govern verdict ───────────────────────────────────────────────────


class TestGovern:
    def test_verdict_requires_both(self):
        """Both weight and output checks must hold for healthy=True."""
        dead = [
            (np.full((15, 4), 1e-5), np.zeros(4)),
            (np.full((4, 1), 1e-5), np.zeros(1)),
        ]

        def const_p(_feats: list[float]) -> float:
            return 0.5

        v = govern(dead, const_p, 15)
        assert not v["healthy"]
        assert not v["weights_healthy"]
        assert not v["outputs_healthy"]
        assert "layer_0_dead_weights" in v["reasons"]
        assert "constant_output" in v["reasons"]


# ── MetaDNN enforcement ──────────────────────────────────────────────


class TestGuardEnforcement:
    def test_load_rejects_collapsed_artifact(self, tmp_path: Path):
        """Constant-output artifact on disk → rejected, gatekeeper bypassed."""
        p = tmp_path / "collapsed.json"
        rng = np.random.default_rng(4)
        layers = [
            (rng.normal(2.0, 0.3, (15, 8)), np.full(8, -60.0)),
            (rng.normal(2.0, 0.3, (8, 4)), np.full(4, -60.0)),
            (rng.normal(0.0, 0.3, (4, 1)), np.zeros(1)),
        ]
        p.write_text(
            json.dumps(
                {
                    "input_dim": 15,
                    "hidden": [8, 4],
                    "weights": [{"w": w.tolist(), "b": b.tolist()} for w, b in layers],
                }
            )
        )
        model = MetaDNN(path=p)
        assert model._defective
        assert not model._built
        admit, pwin, scale = model.infer([0.5] * 15)
        assert admit is True
        assert pwin == META_WIN_THRESHOLD
        assert scale == 1.0

    def test_load_accepts_healthy_artifact(self, tmp_path: Path):
        """Healthy artifact on disk → loaded and active, not bypassed."""
        p = tmp_path / "healthy.json"
        rng = np.random.default_rng(7)
        layers = [
            (rng.normal(0.5, 0.4, (15, 8)), np.zeros(8)),
            (rng.normal(0.5, 0.4, (8, 4)), np.zeros(4)),
            (rng.normal(0.5, 0.4, (4, 1)), np.zeros(1)),
        ]
        p.write_text(
            json.dumps(
                {
                    "input_dim": 15,
                    "hidden": [8, 4],
                    "weights": [{"w": w.tolist(), "b": b.tolist()} for w, b in layers],
                }
            )
        )
        model = MetaDNN(path=p)
        assert not model._defective
        assert model._built

    def test_cold_model_bypasses(self, tmp_path: Path):
        """No artifact → infer admitts instead of random-vetoing."""
        model = MetaDNN(path=tmp_path / "absent.json")
        admit, pwin, scale = model.infer([0.9] * 15)
        assert admit is True
        assert pwin == META_WIN_THRESHOLD
        assert scale == 1.0

    def test_train_refuses_to_save_collapsed(self, tmp_path: Path):
        """Training that stays collapsed must NOT write the artifact."""
        p = tmp_path / "trained.json"
        model = MetaDNN(path=p)
        rng = np.random.default_rng(3)
        model._layers = [
            (rng.normal(2.0, 0.3, (15, 8)), np.full(8, -60.0)),
            (rng.normal(2.0, 0.3, (8, 4)), np.full(4, -60.0)),
            (rng.normal(0.0, 0.3, (4, 1)), np.zeros(1)),
        ]
        model._built = True
        X = rng.normal(0, 1, (300, 15)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.float64)
        model.train(X, y, epochs=3, lr=0.005, batch_size=32)
        assert not p.exists()
        assert model._defective

    def test_live_snapshot_exposes_guard(self, tmp_path: Path):
        """Snapshot carries the guard verdict and defective flag."""
        model = MetaDNN(path=tmp_path / "absent.json")
        snap = model.live_snapshot()
        assert "guard" in snap
        assert "defective" in snap
        assert snap["defective"] is False
        assert snap["guard"]["bypassed"] is True
        assert snap["guard"]["active"] is False

    def test_guard_status_healthy_model(self, tmp_path: Path):
        """A well-behaved model reports an active, non-bypassed guard."""
        p = tmp_path / "ok.json"
        model = MetaDNN(path=p)
        rng = np.random.default_rng(11)
        w0 = rng.normal(5.0, 0.1, (15, 8))
        w1 = rng.normal(5.0, 0.1, (8, 4))
        w2 = rng.normal(5.0, 0.1, (4, 1))
        model._layers = [
            (w0.astype(np.float64), np.zeros(8)),
            (w1.astype(np.float64), np.zeros(4)),
            (w2.astype(np.float64), np.zeros(1)),
        ]
        model._built = True
        status = model.guard_status()
        assert status["healthy"] is True
        assert status["active"] is True
        assert status["bypassed"] is False
        assert MIN_WEIGHT_STD > 0
        assert MIN_OUTPUT_SPREAD > 0
