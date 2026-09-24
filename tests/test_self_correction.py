"""Tests for brain.self_correction — DNN self-review loop.

Covers the PredictionLedger (including ticker-specific resolution),
CalibrationMonitor (classification accuracy / Brier / slope / gap),
DeratingPolicy (anti-snap-back re-rating, drift thresholds), the monotonic
derating invariant (weight 0.0 forces a block), fail-closed blocking for
missing/corrupt/never-trained/defective models, the CorrectionJournal with
its persistent reviewed audit, the review-only consolidation hook, and
proof that no automatic retraining exists anywhere in the module.

Regression: FIX-2026-09-23-13
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

import hanoon_prime.brain.self_correction as sc
from hanoon_prime.brain.meta_label_dnn import DNN_ABSTAIN_P_WIN, MetaDNN
from hanoon_prime.brain.self_correction import CalibrationMonitor, PredictionLedger
from hanoon_prime.brain.self_correction_policy import (
    CorrectionJournal,
    DeratingPolicy,
    apply_derating,
    evaluate_weight,
    request_retrain,
)


def _bad_snapshot() -> dict[str, float]:
    """Critically miscalibrated snapshot."""
    return {"n": 100.0, "accuracy": 0.30, "brier": 0.40, "slope": 0.10, "gap": 0.30}


def _good_snapshot() -> dict[str, float]:
    """Healthy snapshot."""
    return {"n": 150.0, "accuracy": 0.70, "brier": 0.15, "slope": 1.00, "gap": 0.02}


# ── PredictionLedger ─────────────────────────────────────────────────


class TestPredictionLedger:
    def test_log_and_resolve(self, tmp_path: Path):
        """Decisions log; resolution pairs p_win with the outcome."""
        ledger = PredictionLedger(tmp_path)
        pid = ledger.log_decision(
            ticker="AAPL",
            p_win=0.7,
            features=[0.1] * 9,
            size_scale=0.3,
            admitted=True,
            abstained=False,
            weight=1.0,
            drift_z=0.5,
        )
        assert pid
        rid = ledger.resolve_latest(
            ticker="AAPL", realized_pnl=-5.0, won=False, hold_minutes=12.5
        )
        assert rid == pid
        pairs, total = ledger.calibration_data()
        assert total == 1
        assert pairs == [(pytest.approx(0.7), False)]

    def test_resolve_nothing_open_returns_none(self, tmp_path: Path):
        """Resolving with no open prediction returns None."""
        ledger = PredictionLedger(tmp_path)
        assert ledger.resolve_latest(ticker="AAPL", realized_pnl=1.0, won=True) is None

    def test_resolve_matches_most_recent_open(self, tmp_path: Path):
        """The most recent still-open prediction for the ticker is resolved."""
        ledger = PredictionLedger(tmp_path)
        first = ledger.log_decision(
            ticker="AAPL",
            p_win=0.6,
            features=[0.1] * 9,
            size_scale=0.2,
            admitted=True,
            abstained=False,
            weight=1.0,
            drift_z=0.0,
        )
        second = ledger.log_decision(
            ticker="MSFT",
            p_win=0.8,
            features=[0.2] * 9,
            size_scale=0.4,
            admitted=True,
            abstained=False,
            weight=1.0,
            drift_z=0.0,
        )
        assert (
            ledger.resolve_latest(ticker="MSFT", realized_pnl=3.0, won=True) == second
        )
        assert (
            ledger.resolve_latest(ticker="AAPL", realized_pnl=-1.0, won=False) == first
        )
        pairs, total = ledger.calibration_data()
        assert total == 2
        assert pairs[0] == (pytest.approx(0.6), False)  # newest first
        assert pairs[1] == (pytest.approx(0.8), True)

    def test_resolve_is_ticker_specific(self, tmp_path: Path):
        """A close for one ticker never resolves another ticker's prediction."""
        ledger = PredictionLedger(tmp_path)
        ledger.log_decision(
            ticker="MSFT",
            p_win=0.2,
            features=[0.0] * 9,
            size_scale=0.2,
            admitted=True,
            abstained=False,
            weight=1.0,
            drift_z=0.0,
        )
        assert ledger.resolve_latest(ticker="AAPL", realized_pnl=5.0, won=True) is None
        # The MSFT prediction is still open and resolves on its own close.
        pid = ledger.resolve_latest(ticker="MSFT", realized_pnl=-2.0, won=False)
        assert pid is not None

    def test_untagged_open_is_resolvable(self, tmp_path: Path):
        """The gate API logs no ticker yet: untagged opens stay resolvable.

        Documented limitation — the gate() signature carries no ticker, so
        every logged decision has ticker=\"\". An untagged open is eligible
        for any ticker's resolution; once the gate tags tickers, tagged
        opens resolve strictly (see test_resolve_is_ticker_specific).
        """
        ledger = PredictionLedger(tmp_path)
        pid = ledger.log_decision(
            p_win=0.8,
            features=[0.1] * 9,
            size_scale=0.4,
            admitted=True,
            abstained=False,
            weight=1.0,
            drift_z=0.0,
        )
        assert ledger.resolve_latest(ticker="AAPL", realized_pnl=-4.0, won=False) == pid

    def test_record_fields(self, tmp_path: Path):
        """The logged record carries every required field."""
        ledger = PredictionLedger(tmp_path)
        ledger.log_decision(
            ticker="NVDA",
            p_win=0.65,
            features=[0.3] * 9,
            size_scale=0.25,
            admitted=True,
            abstained=False,
            weight=0.5,
            drift_z=1.2,
        )
        rec = json.loads((tmp_path / "predictions.jsonl").read_text().splitlines()[0])
        for field in (
            "ts",
            "ticker",
            "p_win",
            "predicted_edge",
            "features_hash",
            "features",
            "size_scale",
            "admitted",
            "abstained",
            "weight",
            "drift_z",
        ):
            assert field in rec, f"missing ledger field: {field}"
        assert rec["predicted_edge"] == pytest.approx(0.15)
        assert len(rec["features"]) == 9


# ── CalibrationMonitor ───────────────────────────────────────────────


class TestCalibrationMonitor:
    def test_empty_sample(self):
        """Empty input yields a neutral snapshot."""
        s = CalibrationMonitor.summarize([])
        assert s["n"] == 0.0

    def test_accuracy_brier_gap(self):
        """Accuracy, Brier score, and overconfidence gap are exact."""
        pairs = [(0.9, True)] * 70 + [(0.9, False)] * 30
        s = CalibrationMonitor.summarize(pairs)
        assert s["n"] == 100.0
        assert s["accuracy"] == pytest.approx(0.7)
        assert s["brier"] == pytest.approx((0.01 * 70 + 0.81 * 30) / 100)
        assert s["gap"] == pytest.approx(0.9 - 0.7)

    def test_accuracy_is_classification_not_win_rate(self):
        """Accuracy compares (p >= 0.5) against the realized class."""
        # 50% win rate, but every 0.9-call won and every 0.1-call lost.
        pairs = [(0.9, True)] * 50 + [(0.1, False)] * 50
        s = CalibrationMonitor.summarize(pairs)
        assert s["accuracy"] == pytest.approx(1.0)
        # And the mirror: 50% win rate with every call wrong.
        pairs = [(0.9, False)] * 50 + [(0.1, True)] * 50
        s = CalibrationMonitor.summarize(pairs)
        assert s["accuracy"] == pytest.approx(0.0)

    def test_slope_of_perfectly_calibrated(self):
        """A perfectly calibrated forecaster has slope ~1."""
        pairs = (
            [(0.8, True)] * 80
            + [(0.8, False)] * 20
            + [(0.2, True)] * 20
            + [(0.2, False)] * 80
        )
        s = CalibrationMonitor.summarize(pairs)
        assert s["slope"] == pytest.approx(1.0)
        assert s["accuracy"] == pytest.approx(0.8)

    def test_slope_zero_when_no_spread(self):
        """Constant forecasts give slope 0, not NaN."""
        s = CalibrationMonitor.summarize([(0.6, True)] * 10)
        assert s["slope"] == 0.0


# ── DeratingPolicy ───────────────────────────────────────────────────


class TestDeratingPolicy:
    def test_healthy_stays_full(self):
        """A healthy, undrifted model keeps weight 1.0."""
        w, _ = DeratingPolicy.target_weight(_good_snapshot(), 0.0)
        assert w == 1.0

    def test_cold_start_is_conservative(self):
        """Unknown track record -> 0.5, never full weight."""
        w, reason = DeratingPolicy.target_weight({"n": 0.0}, 0.0)
        assert w == 0.5
        assert "cold" in reason

    def test_degraded_calibration(self):
        """Weak calibration halves the weight."""
        snap = {"n": 100.0, "accuracy": 0.52, "brier": 0.22, "slope": 0.8, "gap": 0.05}
        w, _ = DeratingPolicy.target_weight(snap, 0.0)
        assert w == 0.5

    def test_critical_calibration_abstains(self):
        """Critically miscalibrated -> weight 0.0."""
        w, _ = DeratingPolicy.target_weight(_bad_snapshot(), 0.0)
        assert w == 0.0

    def test_drift_thresholds(self):
        """z>3 caps at 0.5; z>5 forces abstain; sign does not matter."""
        snap = _good_snapshot()
        assert DeratingPolicy.target_weight(snap, 2.9)[0] == 1.0
        assert DeratingPolicy.target_weight(snap, 3.1)[0] == 0.5
        assert DeratingPolicy.target_weight(snap, -4.0)[0] == 0.5
        assert DeratingPolicy.target_weight(snap, 5.1)[0] == 0.0
        assert DeratingPolicy.target_weight(snap, -5.1)[0] == 0.0

    def test_drift_cannot_rescue_bad_calibration(self):
        """Drift caps apply via min(): a bad model stays at 0.0."""
        w, _ = DeratingPolicy.target_weight(_bad_snapshot(), 3.5)
        assert w == 0.0


# ── Monotonic derating invariant ─────────────────────────────────────


class TestApplyDerating:
    def test_never_grows_size(self):
        """Derated size never exceeds the requested size."""
        for w in (1.0, 0.5, 0.0, 2.0):
            _, _, s = apply_derating(True, 0.9, 0.4, w, threshold=0.52)
            assert s <= 0.4

    def test_never_lowers_bar(self):
        """The entry bar only rises under derating, never falls."""
        assert apply_derating(True, 0.52, 0.3, 1.0, threshold=0.52)[0] is True
        assert apply_derating(True, 0.53, 0.3, 1.0, threshold=0.52)[0] is True
        assert apply_derating(True, 0.53, 0.3, 0.5, threshold=0.52)[0] is False
        assert apply_derating(True, 0.60, 0.3, 0.5, threshold=0.52)[0] is True

    def test_zero_weight_forces_block(self):
        """Weight 0.0 blocks even a high-confidence admission."""
        assert apply_derating(True, 0.95, 0.4, 0.0, threshold=0.52) == (
            False,
            0.95,
            0.0,
        )
        assert apply_derating(True, 0.99, 0.4, 0.0, threshold=0.52)[0] is False

    def test_size_monotonic_in_weight(self):
        """Size shrinks monotonically as the weight falls."""
        scales = [
            apply_derating(True, 0.8, 0.5, w, threshold=0.52)[2]
            for w in (1.0, 0.5, 0.0)
        ]
        assert scales == sorted(scales, reverse=True)

    def test_full_weight_is_neutral(self):
        """Weight 1.0 reproduces the raw gate decision exactly."""
        assert apply_derating(True, 0.7, 0.35, 1.0, threshold=0.52) == (True, 0.7, 0.35)
        assert apply_derating(False, 0.4, 0.0, 1.0, threshold=0.52) == (False, 0.4, 0.0)


# ── Anti-snap-back re-rating ─────────────────────────────────────────


class TestEvaluateWeight:
    def test_derates_down_immediately(self, tmp_path: Path):
        """A bad evaluation cuts the weight at once."""
        w, note = evaluate_weight(_bad_snapshot(), 0.0, 100, directory=tmp_path)
        assert w == 0.0
        assert "derated" in note

    def test_rerate_needs_100_new_samples(self, tmp_path: Path):
        """Upward moves wait for 100 new resolved samples."""
        w, _ = evaluate_weight(_bad_snapshot(), 0.0, 100, directory=tmp_path)
        assert w == 0.0
        w, note = evaluate_weight(_good_snapshot(), 0.0, 150, directory=tmp_path)
        assert w == 0.0
        assert "withheld" in note

    def test_rerate_moves_one_step_per_evaluation(self, tmp_path: Path):
        """0.0 -> 0.5 -> 1.0, never straight back to full."""
        assert evaluate_weight(_bad_snapshot(), 0.0, 100, directory=tmp_path)[0] == 0.0
        assert evaluate_weight(_good_snapshot(), 0.0, 200, directory=tmp_path)[0] == 0.5
        assert evaluate_weight(_good_snapshot(), 0.0, 300, directory=tmp_path)[0] == 1.0

    def test_state_survives_reload(self, tmp_path: Path):
        """The derated weight persists across evaluations."""
        evaluate_weight(_bad_snapshot(), 0.0, 100, directory=tmp_path)
        w, _ = evaluate_weight(_good_snapshot(), 0.0, 120, directory=tmp_path)
        assert w == 0.0  # baseline remembered; only 20 new samples

    def test_critical_derate_files_retrain_request(self, tmp_path: Path):
        """A transition to 0.0 files a human-approval retrain request."""
        w, _ = evaluate_weight(_bad_snapshot(), 0.0, 100, directory=tmp_path)
        assert w == 0.0
        reqs = list((tmp_path / "retrain_requests").glob("*.json"))
        assert len(reqs) == 1
        req = json.loads(reqs[0].read_text())
        assert req["status"] == "PENDING_HUMAN_APPROVAL"
        assert "drift" in req["reason"] or "miscalibrated" in req["reason"]


# ── Fail-closed inference ────────────────────────────────────────────


class TestFailClosed:
    def test_missing_model_blocks(self, tmp_path: Path, caplog):
        """Missing artifact -> block (False, 0.5, 0.0) + CRITICAL log."""
        dnn = MetaDNN(path=tmp_path / "nope.json")
        assert dnn.abstain_reason == "missing_artifact"
        with caplog.at_level(logging.CRITICAL):
            out = dnn.infer([0.5] * 9)
        assert out == (False, DNN_ABSTAIN_P_WIN, 0.0)
        assert DNN_ABSTAIN_P_WIN == 0.5
        assert any(
            "BLOCK" in r.message and "missing_artifact" in r.message
            for r in caplog.records
            if r.levelno >= logging.CRITICAL
        )

    def test_corrupt_model_blocks(self, tmp_path: Path):
        """Corrupt artifact -> block, never a fabricated score."""
        p = tmp_path / "bad.json"
        p.write_text("NOT VALID JSON {{{")
        dnn = MetaDNN(path=p)
        assert dnn.abstain_reason == "corrupt_artifact"
        assert dnn.infer([0.5] * 9) == (False, DNN_ABSTAIN_P_WIN, 0.0)

    def test_never_trained_model_blocks(self, tmp_path: Path):
        """Valid weights without a training report -> block."""
        p = tmp_path / "dnn.json"
        dnn = MetaDNN(path=p)
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 9))
        y = (X[:, 0] > 0).astype(float)
        dnn.train(X, y, epochs=5, lr=0.01, batch_size=16)
        # A trained model IS usable: train() wrote artifact + report.
        assert MetaDNN(path=p).abstain_reason is None
        # Lose the report -> the same weights are untrusted: block.
        p.with_suffix(".report.json").unlink()
        dnn3 = MetaDNN(path=p)
        assert dnn3.abstain_reason == "never_trained"
        assert dnn3.infer([0.5] * 9) == (False, DNN_ABSTAIN_P_WIN, 0.0)

    def test_guard_rejected_model_blocks(self, tmp_path: Path):
        """A guard-rejected (defective) artifact blocks DNN-gated entries."""
        p = tmp_path / "dnn.json"
        dnn = MetaDNN(path=p)
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 9))
        y = (X[:, 0] > 0).astype(float)
        dnn.train(X, y, epochs=5, lr=0.01, batch_size=16)
        assert dnn.abstain_reason is None
        # Simulate the ironclad guard rejecting the artifact on reload.
        dnn._defective = True
        dnn._defect_reason = "guard_rejected"
        dnn._refresh_model_status()
        assert dnn.abstain_reason == "guard_rejected"
        assert dnn.infer([0.5] * 9) == (False, DNN_ABSTAIN_P_WIN, 0.0)

    def test_train_makes_model_usable(self, tmp_path: Path):
        """train() writes artifact + report: the model stops blocking."""
        p = tmp_path / "dnn.json"
        dnn = MetaDNN(path=p)
        assert dnn.abstain_reason == "missing_artifact"
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 9))
        y = (X[:, 0] > 0).astype(float)
        dnn.train(X, y, epochs=5, lr=0.01, batch_size=16)
        assert dnn.abstain_reason is None
        assert p.with_suffix(".report.json").exists()

    def test_inference_exception_blocks(self, tmp_path: Path, caplog, monkeypatch):
        """A throwing predict() blocks fail-closed at CRITICAL."""
        p = tmp_path / "dnn.json"
        dnn = MetaDNN(path=p)
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 9))
        y = (X[:, 0] > 0).astype(float)
        dnn.train(X, y, epochs=5, lr=0.01, batch_size=16)
        assert dnn.abstain_reason is None

        def _boom(features):
            raise RuntimeError("boom")

        monkeypatch.setattr(dnn, "predict", _boom)
        with caplog.at_level(logging.CRITICAL):
            out = dnn.infer([0.5] * 9)
        assert out == (False, DNN_ABSTAIN_P_WIN, 0.0)
        assert dnn.abstain_reason == "inference_error"
        assert any(
            "inference failed" in r.message
            for r in caplog.records
            if r.levelno >= logging.CRITICAL
        )

    def test_block_keeps_drift_history_clean(self, tmp_path: Path):
        """Blocks are not predictions: drift history stays empty."""
        dnn = MetaDNN(path=tmp_path / "nope.json")
        for _ in range(10):
            dnn.infer([0.5] * 9)
        assert dnn._p_win_history == []
        assert dnn.drift_zscore() == 0.0

    def test_block_log_is_throttled(self, tmp_path: Path, caplog):
        """Repeated blocks do not spam CRITICAL on every call."""
        dnn = MetaDNN(path=tmp_path / "nope.json")
        with caplog.at_level(logging.CRITICAL):
            for _ in range(5):
                dnn.infer([0.5] * 9)
        blocks = [
            r
            for r in caplog.records
            if r.levelno >= logging.CRITICAL and "BLOCK" in r.message
        ]
        assert len(blocks) == 1


# ── CorrectionJournal ────────────────────────────────────────────────


class TestCorrectionJournal:
    def test_queues_high_confidence_mistakes(self, tmp_path: Path):
        """Confident-win-that-lost and confident-loss-that-won queue."""
        j = CorrectionJournal(tmp_path)
        assert (
            j.maybe_enqueue(
                pred_id="a",
                p_win=0.8,
                features=[0.1] * 9,
                realized_pnl=-2.0,
                won=False,
                ticker="AAPL",
            )
            is True
        )
        assert (
            j.maybe_enqueue(
                pred_id="b",
                p_win=0.2,
                features=[0.1] * 9,
                realized_pnl=3.0,
                won=True,
                ticker="MSFT",
            )
            is True
        )
        assert (
            j.maybe_enqueue(
                pred_id="c",
                p_win=0.8,
                features=[0.1] * 9,
                realized_pnl=2.0,
                won=True,
                ticker="AAPL",
            )
            is False
        )
        items = j.drain()
        assert len(items) == 2
        assert items[0]["features"] == [0.1] * 9
        assert items[0]["ticker"] == "AAPL"
        assert j.drain() == []

    def test_drain_preserves_reviewed_audit(self, tmp_path: Path):
        """Drained records persist in the append-only reviewed journal."""
        j = CorrectionJournal(tmp_path)
        j.maybe_enqueue(
            pred_id="a",
            p_win=0.8,
            features=[0.1] * 9,
            realized_pnl=-2.0,
            won=False,
            ticker="AAPL",
        )
        items = j.drain()
        assert len(items) == 1
        assert j.drain() == []  # queue is empty after drain
        reviewed = [
            json.loads(line)
            for line in (tmp_path / "reviewed.jsonl").read_text().splitlines()
        ]
        assert len(reviewed) == 1
        assert reviewed[0]["pred_id"] == "a"
        assert reviewed[0]["ticker"] == "AAPL"
        assert "reviewed_ts" in reviewed[0]
        # A second drain appends; the audit is append-only, never rewritten.
        j.maybe_enqueue(
            pred_id="b",
            p_win=0.2,
            features=[0.2] * 9,
            realized_pnl=3.0,
            won=True,
            ticker="MSFT",
        )
        j.drain()
        reviewed = [
            json.loads(line)
            for line in (tmp_path / "reviewed.jsonl").read_text().splitlines()
        ]
        assert [r["pred_id"] for r in reviewed] == ["a", "b"]

    def test_resolve_auto_enqueues_mistake(self, tmp_path: Path):
        """Resolving a high-confidence loss queues it for review."""
        ledger = PredictionLedger(tmp_path)
        ledger.log_decision(
            ticker="NVDA",
            p_win=0.85,
            features=[0.2] * 9,
            size_scale=0.4,
            admitted=True,
            abstained=False,
            weight=1.0,
            drift_z=0.0,
        )
        ledger.resolve_latest(ticker="NVDA", realized_pnl=-10.0, won=False)
        items = CorrectionJournal(tmp_path).drain()
        assert len(items) == 1
        assert items[0]["ticker"] == "NVDA"
        assert items[0]["p_win"] == pytest.approx(0.85)


# ── Live gate wiring ─────────────────────────────────────────────────


class TestGateWiring:
    def test_unusable_dnn_blocks_and_logs(self, tmp_path: Path, monkeypatch):
        """An unusable DNN blocks the entry and the decision is ledgered."""
        import hanoon_prime.brain.meta_label as ml

        monkeypatch.setenv("SELF_CORRECTION_DIR", str(tmp_path))
        monkeypatch.setattr(ml, "_WEIGHT_CACHE", None)
        dnn = MetaDNN(path=tmp_path / "nope.json")
        out = ml.gate_with_derating(dnn, [0.5] * 9, True, 0.5, 0.0)
        assert out == (False, 0.5, 0.0)
        lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["abstained"] is True
        assert rec["admitted"] is False

    def test_zero_weight_blocks(self, tmp_path: Path, monkeypatch):
        """A derated-to-zero weight blocks even a usable model."""
        import time

        import hanoon_prime.brain.meta_label as ml

        monkeypatch.setenv("SELF_CORRECTION_DIR", str(tmp_path))
        monkeypatch.setattr(ml, "_WEIGHT_CACHE", (time.time(), 0.0, "test"))

        class _FakeDNN:
            abstain_reason = None

            def drift_zscore(self):
                return 0.0

        out = ml.gate_with_derating(_FakeDNN(), [0.5] * 9, True, 0.9, 0.4)
        # Stepping aside neutralizes the score: the model's p_win is not
        # trusted at all once the derated weight hits 0.0.
        assert out == (False, 0.5, 0.0)

    def test_dnn_gate_exception_blocks(self, tmp_path: Path, monkeypatch):
        """A throwing gatekeeper blocks instead of falling through."""
        import hanoon_prime.brain.meta_label as ml

        monkeypatch.setenv("SELF_CORRECTION_DIR", str(tmp_path))

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(ml, "_get_dnn", _boom)
        model = ml.MetaLabelModel(path=tmp_path / "meta.json")
        assert model.gate(0.5, 0.5, 0.3, "range", "scalp") == (False, 0.5, 0.0)

    def test_healthy_model_passes_through(self, tmp_path: Path, monkeypatch):
        """A usable model with full weight is returned unchanged."""
        import time

        import hanoon_prime.brain.meta_label as ml

        monkeypatch.setenv("SELF_CORRECTION_DIR", str(tmp_path))
        monkeypatch.setattr(ml, "_WEIGHT_CACHE", (time.time(), 1.0, "test"))

        class _FakeDNN:
            abstain_reason = None

            def drift_zscore(self):
                return 0.0

        out = ml.gate_with_derating(_FakeDNN(), [0.5] * 9, True, 0.7, 0.35)
        assert out == (True, 0.7, 0.35)


# ── Consolidation hook: review-only ──────────────────────────────────


class TestConsolidationHook:
    def _method_source(self, name: str) -> str:
        import ast

        repo = Path(__file__).resolve().parents[1]
        src = (repo / "src" / "hanoon_prime" / "brain" / "consolidation.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                seg = ast.get_source_segment(src, node)
                assert seg is not None
                return seg
        raise AssertionError(f"{name} not found in consolidation.py")

    def test_self_correction_cycle_is_review_only(self):
        """The S2 hook marks review; it never replays into parameters."""
        seg = self._method_source("_self_correction_cycle")
        assert "run_sleep_replay" not in seg
        assert "SleepReplay" not in seg
        assert ".fit(" not in seg
        assert "train(" not in seg
        assert "reviewed" in seg  # audited review marking

    def _helpers(self):
        """Load _epoch_seconds/_hold_minutes from source, no heavy import."""
        import ast

        repo = Path(__file__).resolve().parents[1]
        src = (repo / "src" / "hanoon_prime" / "brain" / "consolidation.py").read_text()
        tree = ast.parse(src)
        wanted = {"_epoch_seconds", "_hold_minutes"}
        ns: dict = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                seg = ast.get_source_segment(src, node)
                assert seg is not None
                exec(
                    compile(
                        "from __future__ import annotations\n" + seg,
                        "<helpers>",
                        "exec",
                    ),
                    ns,
                )  # noqa: S102
        assert wanted <= set(ns), "helpers not found in consolidation.py"
        return ns["_epoch_seconds"], ns["_hold_minutes"]

    def test_hold_minutes_handles_floats(self):
        _, _hold_minutes = self._helpers()
        assert _hold_minutes(1000.0, 1060.0) == 1.0
        assert _hold_minutes(1000.0, 1000.0) == 0.0

    def test_hold_minutes_handles_datetimes(self):
        from datetime import datetime, timezone

        _, _hold_minutes = self._helpers()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
        assert _hold_minutes(start, end) == 30.0

    def test_hold_minutes_tolerates_garbage(self):
        _, _hold_minutes = self._helpers()
        assert _hold_minutes(None, 1060.0) is None
        assert _hold_minutes(1000.0, None) is None
        assert _hold_minutes("garbage", 1060.0) is None

    def test_epoch_seconds_contract(self):
        """Epoch numerics, numeric strings, and datetimes convert; else None."""
        from datetime import datetime, timezone

        _epoch_seconds, _ = self._helpers()
        assert _epoch_seconds(None) is None
        assert _epoch_seconds(1000.0) == 1000.0
        assert _epoch_seconds(1000) == 1000.0
        assert _epoch_seconds("1000.5") == 1000.5
        assert _epoch_seconds("not-a-time") is None
        assert _epoch_seconds("") is None
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _epoch_seconds(dt) == dt.timestamp()
        # ISO strings are not parsed — only epoch numerics are accepted.
        assert _epoch_seconds(dt.isoformat()) is None


# ── No automatic retraining ──────────────────────────────────────────


class TestNoAutomaticRetraining:
    def test_derate_to_zero_trains_nothing(self, tmp_path: Path):
        """Critical derating files a request but fits no model."""
        w, _ = evaluate_weight(_bad_snapshot(), 6.0, 100, directory=tmp_path)
        assert w == 0.0
        assert list((tmp_path / "retrain_requests").glob("*.json"))
        # No model artifact was created or touched by the module.
        assert not (tmp_path / "juli_meta_dnn.json").exists()
        model_files = [
            p
            for p in tmp_path.rglob("*")
            if p.suffix in (".json", ".pkl", ".pt")
            and "retrain" not in p.name
            and "derating_state" not in p.name
            and "predictions" not in p.name
            and "review_queue" not in p.name
            and "reviewed" not in p.name
        ]
        assert model_files == []

    def test_request_retrain_only_writes_a_file(self, tmp_path: Path):
        """request_retrain() performs no fitting of any kind."""
        path = request_retrain("test", _good_snapshot(), 0.0, directory=tmp_path)
        assert path is not None and path.exists()
        payload = json.loads(path.read_text())
        assert payload["status"] == "PENDING_HUMAN_APPROVAL"

    def test_module_contains_no_training_machinery(self):
        """The modules cannot train: no train/fit entry points exist."""
        import hanoon_prime.brain.self_correction_policy as scp

        for mod in (sc, scp):
            src = Path(mod.__file__).read_text()
            assert "def train" not in src
            assert ".fit(" not in src
            assert "gradient" not in src.lower()
