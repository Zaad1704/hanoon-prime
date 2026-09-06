"""test_coverage_reflection — distillation + retraining engines."""

from __future__ import annotations

from types import SimpleNamespace

from hanoon_prime.reflection.distill import DistillationEngine
from hanoon_prime.reflection.retrain import RetrainEngine


class _FakeMemory:
    """Minimal memory double for the distillation engine."""

    def __init__(self, win_rate=0.5, weights=None, total_trades=50):
        self._wr = win_rate
        self._weights = dict(weights or {})
        self._total = total_trades
        self.set_weights_calls: list = []

    def snapshot(self) -> dict:
        return {
            "win_rate": self._wr,
            "total_trades": self._total,
            "weights": dict(self._weights),
        }

    def set_weights(self, weights) -> None:
        self._weights = dict(weights)
        self.set_weights_calls.append(dict(weights))


def _fake_memory(win_rate: float = 0.5, weights=None, total_trades: int = 50):
    return _FakeMemory(win_rate=win_rate, weights=weights, total_trades=total_trades)


# ── DistillationEngine ──────────────────────────────────────────────────
class TestDistillationEngine:
    def test_distill_appends_digest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hanoon_prime.reflection.distill._DIGEST_PATH", tmp_path / "digest.json"
        )
        eng = DistillationEngine()
        eng.distill(_fake_memory(win_rate=0.6))
        assert len(eng._digests) == 1
        assert eng._digests[0]["win_rate"] == 0.6

    def test_history_capped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hanoon_prime.reflection.distill._DIGEST_PATH", tmp_path / "digest.json"
        )
        eng = DistillationEngine()
        mem = _fake_memory()
        for _ in range(7):
            eng.distill(mem)
        assert len(eng._digests) <= 5

    def test_guard_clean(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hanoon_prime.reflection.distill._DIGEST_PATH", tmp_path / "digest.json"
        )
        eng = DistillationEngine()
        eng.distill(_fake_memory(win_rate=0.5, total_trades=20))
        assert eng.guard(_fake_memory(win_rate=0.5, total_trades=20)) is True

    def test_guard_flags_corruption(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hanoon_prime.reflection.distill._DIGEST_PATH", tmp_path / "digest.json"
        )
        eng = DistillationEngine()
        eng.distill(_fake_memory(win_rate=0.2, total_trades=20))
        # big jump + enough trades -> suspicious
        assert eng.guard(_fake_memory(win_rate=0.9, total_trades=20)) is False

    def test_guard_no_digests(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hanoon_prime.reflection.distill._DIGEST_PATH", tmp_path / "digest.json"
        )
        eng = DistillationEngine()
        assert eng.guard(_fake_memory()) is True

    def test_rollback_restores_weights(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hanoon_prime.reflection.distill._DIGEST_PATH", tmp_path / "digest.json"
        )
        eng = DistillationEngine()
        weights = {"vpin": 0.08, "momentum": 0.10}
        eng.distill(_fake_memory(win_rate=0.6, weights=weights))
        mem = _fake_memory(win_rate=0.9, weights={})
        assert eng.rollback(mem) is True
        assert mem.set_weights_calls == [weights]

    def test_rollback_no_digests(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hanoon_prime.reflection.distill._DIGEST_PATH", tmp_path / "digest.json"
        )
        eng = DistillationEngine()
        mem = _fake_memory()
        assert eng.rollback(mem) is False

    def test_persists_across_reload(self, tmp_path, monkeypatch):
        path = tmp_path / "digest.json"
        monkeypatch.setattr("hanoon_prime.reflection.distill._DIGEST_PATH", path)
        eng = DistillationEngine()
        eng.distill(_fake_memory(win_rate=0.7))
        eng2 = DistillationEngine()
        assert len(eng2._digests) == 1
        assert eng2._digests[0]["win_rate"] == 0.7

    def test_load_corrupt_does_not_crash(self, tmp_path, monkeypatch):
        path = tmp_path / "digest.json"
        path.write_text("{bad json")
        monkeypatch.setattr("hanoon_prime.reflection.distill._DIGEST_PATH", path)
        eng = DistillationEngine()
        assert eng._digests == []


# ── RetrainEngine ───────────────────────────────────────────────────────
def _fake_buffer(n: int, win: bool = True):
    return SimpleNamespace(
        get_trades=lambda last_n=50: [SimpleNamespace(win=win) for _ in range(n)]
    )


class TestRetrainEngine:
    def test_should_retrain_no_buffer(self):
        eng = RetrainEngine(buffer=None)
        assert eng.should_retrain() is False

    def test_should_retrain_needs_trades(self):
        eng = RetrainEngine(buffer=_fake_buffer(10))
        assert eng.should_retrain() is False

    def test_should_retrain_true(self):
        eng = RetrainEngine(buffer=_fake_buffer(50))
        assert eng.should_retrain() is True

    def test_retrain_skipped(self):
        eng = RetrainEngine(buffer=_fake_buffer(10))
        rep = eng.retrain()
        assert rep["status"] == "skipped"

    def test_retrain_runs(self):
        calls = []
        recorder = lambda: calls.append("retrained")
        eng = RetrainEngine(buffer=_fake_buffer(60), on_retrained=recorder)
        rep = eng.retrain()
        assert rep["status"] == "done"
        assert rep["trades"] == 60
        assert len(calls) == 1
        assert eng._retrain_count == 1

    def test_retrain_callback_swallows_error(self):
        def boom():
            raise RuntimeError("nope")

        eng = RetrainEngine(buffer=_fake_buffer(60), on_retrained=boom)
        rep = eng.retrain()
        assert rep["status"] == "done"

    def test_telemetry(self):
        eng = RetrainEngine(buffer=None)
        tel = eng.get_telemetry()
        assert "last_retrain" in tel
        assert tel["retrain_count"] == 0
