"""tests/test_validation_honesty.py — Phase 3: honest-validation regressions.

Guards that the validation harness measures what it claims:
  * FIX-2026-09-23-09: the shipped WFA never fits anything, so purge/embargo
    are (correctly) NOT applied to the run; the sim scores via the standalone
    cortex path and never consults the trained MetaDNN artifact — the report
    must disclose both.
  * FIX-2026-09-23-10: the deflation trial count is calibrated to the same
    admissible ticker set that feeds the pooled Sharpe.
  * FIX-2026-09-23-11: run_paper enforces the pre-locked P5 protocol
    thresholds (deflated OOS Sharpe > 0.05 AND PBO < 0.05) as a separate
    gate, and actually evaluates P1–P6 (evaluate() was never called).
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hanoon_prime.hippocampus import Hippocampus
from hanoon_prime.immune import EDGE_LOOKBACK
from hanoon_prime.wfa import (
    MIN_TRADES,
    UniverseVerdict,
    _count_trials,
    deflated_sharpe,
    run_walk_forward,
    serialize,
    verdicts,
)
from scripts import paper_run


def _synthetic_ohlcv(n: int) -> dict[str, Any]:
    """Deterministic OHLCV panel for driving run_walk_forward / the sim."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0.02, 0.5, n))
    return {
        "datetime": [
            f"2026-09-{1 + i // 1440:02d} "
            f"{(9 + (i // 60)) % 24:02d}:{i % 60:02d}:00"
            for i in range(n)
        ],
        "open": close - 0.1,
        "high": close + np.abs(rng.normal(0, 0.3, n)),
        "low": close - np.abs(rng.normal(0, 0.3, n)),
        "close": close,
        "volume": np.full(n, 1000.0),
    }


def _wfa_fold(pnl: list[float], fold: int = 0):
    """FoldResult with trades populated to match pnl (kept in sync)."""
    from hanoon_prime.wfa import FoldResult

    mean = float(np.mean(pnl)) if pnl else 0.0
    std = float(np.std(pnl)) if pnl else 0.0
    return FoldResult(
        fold=fold,
        start=fold,
        end=fold + 1,
        trades=[object() for _ in pnl],
        ev_per_trade=mean,
        sharpe=float(mean / (std + 1e-12)),
        pnl=pnl,
    )


# ── FIX-2026-09-23-09: validation-scope disclosure ─────────────────────────


def test_serialize_carries_validation_caveats() -> None:
    """Regression: FIX-2026-09-23-09 — the report must disclose what the
    WFA does NOT validate (no DNN in the sim; purge/embargo not applied)."""
    blob = serialize(verdicts({}))
    caveats = blob["validation_caveats"]
    assert isinstance(caveats, list) and caveats
    assert any("MetaDNN" in c for c in caveats)
    assert any("purge" in c.lower() for c in caveats)


def test_run_walk_forward_never_calls_purge_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: FIX-2026-09-23-09 — the shipped WFA fits nothing, so
    the purge/embargo helpers must not be consulted by the run (wiring them
    in would be theater, not leakage control)."""
    import hanoon_prime.wfa as wfa_mod

    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("purge/embargo must not be called by the run")

    monkeypatch.setattr(wfa_mod, "purge_mask", _boom)
    monkeypatch.setattr(wfa_mod, "embargo_mask", _boom)
    monkeypatch.setattr(wfa_mod, "train_test_split", _boom)
    monkeypatch.setattr(wfa_mod, "purged_fold_windows", _boom)
    res = run_walk_forward("SYN", _synthetic_ohlcv(600), folds=2)
    assert len(res) == 2


def test_sim_does_not_consult_trained_dnn_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: FIX-2026-09-23-09 — even a present, veto-everything DNN
    artifact must not change the sim's trades (the sim never loads it)."""
    import hanoon_prime.brain.learning_config as lc
    from hanoon_prime.hands import simulate_ticker

    hostile = tmp_path / "juli_meta_dnn.json"
    hostile.write_text(
        json.dumps(
            {
                "input_dim": 9,
                "hidden": [32, 16],
                "p_win_threshold": 0.99,  # veto-everything gate
                "weights": [],
            }
        )
    )
    data = _synthetic_ohlcv(600)
    monkeypatch.setattr(lc, "META_DNN_FILE", hostile)
    trades_with_hostile = simulate_ticker("SYN", data, EDGE_LOOKBACK)
    monkeypatch.setattr(lc, "META_DNN_FILE", tmp_path / "missing.json")
    trades_without = simulate_ticker("SYN", data, EDGE_LOOKBACK)
    assert len(trades_with_hostile) == len(trades_without)


def test_sim_scoring_path_excludes_dnn_machinery() -> None:
    """Regression: FIX-2026-09-23-09 — pin the architectural boundary: the
    sim stack must not reference the DNN/orchestrator machinery. Honest
    disclosure strings (the VALIDATION_CAVEATS that name MetaDNN) are
    exempt — only real code references (imports, names, attributes) count.
    """
    import hanoon_prime.hands as hands_mod
    import hanoon_prime.wfa as wfa_mod

    hands_src = inspect.getsource(hands_mod)
    for token in ("MetaDNN", "meta_label_dnn", "orchestrator"):
        assert token not in hands_src, token

    code_tokens: list[str] = []

    class _RefVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            code_tokens.extend(a.asname or a.name for a in node.names)
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            code_tokens.extend(a.asname or a.name for a in node.names)
            if node.module:
                code_tokens.append(node.module)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            code_tokens.append(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            code_tokens.append(node.attr)
            self.generic_visit(node)

    _RefVisitor().visit(ast.parse(inspect.getsource(wfa_mod)))
    for token in ("MetaDNN", "meta_label_dnn"):
        assert not any(token in t for t in code_tokens), token


# ── FIX-2026-09-23-10: admissible-set trial counting ──────────────────────


def test_count_trials_admissible_only() -> None:
    """Regression: FIX-2026-09-23-10 — admissible ticker contributes its
    qualifying fold cells; a thin ticker contributes zero cells."""
    results = {
        # A: 2×MIN_TRADES OOS → admissible, 2 qualifying cells → 2 trials.
        "A": [_wfa_fold([0.01] * MIN_TRADES, 0), _wfa_fold([0.01] * MIN_TRADES, 1)],
        # B: qualifying cells but 5 OOS trades → inadmissible → 0 trials.
        "B": [_wfa_fold([0.01] * 5, 0)],
    }
    assert _count_trials(results) == 2


def test_deflation_less_conservative_with_fewer_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: FIX-2026-09-23-10 — pin the direction of effect: holding
    the pooled returns fixed, MORE trials → a harsher best-by-chance null →
    a lower (more conservative) deflated edge."""
    import hanoon_prime.wfa as wfa_mod

    # Non-degenerate pnl (sd > 0): a zero-variance panel would make the
    # observed Sharpe exactly 0 and the direction check vacuous.
    pnl = [0.06, -0.02, 0.05, 0.01, -0.03, 0.07, 0.02, -0.01] * 5
    results = {"A": [_wfa_fold(pnl)]}
    real_trials = wfa_mod._count_trials(results)
    monkeypatch.setattr(wfa_mod, "RNG", np.random.default_rng(0))
    few = wfa_mod.deflated_sharpe(results)
    monkeypatch.setattr(wfa_mod, "_count_trials", lambda r: real_trials * 10)
    monkeypatch.setattr(wfa_mod, "RNG", np.random.default_rng(0))
    many = wfa_mod.deflated_sharpe(results)
    assert few >= many


# ── FIX-2026-09-23-11: Phase-4 strict P5 + restored P1–P6 evaluation ───────


def _universe_verdict(deflated_edge: float, pbo: float) -> UniverseVerdict:
    return UniverseVerdict(
        admissible_tickers=[],
        pooled_sharpe=0.10,
        deflated_edge=deflated_edge,
        pbo=pbo,
        verdict="PASS",
        detail="synthetic",
    )


def _tiny_csv(tmp_path: Path) -> None:
    rows = []
    price = 100.0
    for i in range(300):
        price *= 1.0001
        rows.append(
            f"2026-09-{1 + i // 1440:02d} "
            f"{(9 + (i // 60)) % 24:02d}:{i % 60:02d}:00,"
            f"{price:.2f},{price * 1.001:.2f},{price * 0.999:.2f},"
            f"{price:.2f},1000"
        )
    (tmp_path / "SYN_1min.csv").write_text("\n".join(rows) + "\n")


def _stub_wfa(
    monkeypatch: pytest.MonkeyPatch, deflated_edge: float, pbo: float
) -> None:
    monkeypatch.setattr("hanoon_prime.wfa.run_wfa_universe", lambda *a, **k: {})
    monkeypatch.setattr(
        "hanoon_prime.wfa.verdicts", lambda r: _universe_verdict(deflated_edge, pbo)
    )


def test_p5_protocol_gate_blocks_weak_engine_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: FIX-2026-09-23-11 — deflated 0.03 passes wfa.verdicts()
    (deflated > 0) but must FAIL the pre-locked P5 protocol gate."""
    _tiny_csv(tmp_path)
    _stub_wfa(monkeypatch, 0.03, 0.03)
    report = paper_run.run_paper(tmp_path, ["SYN"], Hippocampus())
    assert report["criteria"]["P5_wfa_pass"] is True
    assert report["criteria"]["P5_protocol_thresholds"] is False
    assert report["verdict"] == "FAIL"


def test_p5_protocol_gate_passes_above_strict_thresholds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: FIX-2026-09-23-11 — deflated 0.06 and PBO 0.04 clear
    the strict protocol gate (equality at either boundary must FAIL)."""
    _tiny_csv(tmp_path)
    _stub_wfa(monkeypatch, 0.06, 0.04)
    report = paper_run.run_paper(tmp_path, ["SYN"], Hippocampus())
    assert report["criteria"]["P5_protocol_thresholds"] is True


@pytest.mark.parametrize(
    "deflated_edge,pbo",
    [(0.05, 0.04), (0.06, 0.05), (0.05, 0.05)],
)
def test_p5_protocol_gate_strict_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    deflated_edge: float,
    pbo: float,
) -> None:
    """Regression: FIX-2026-09-23-11 — the protocol thresholds are strict:
    equality at either boundary must FAIL the P5 gate."""
    _tiny_csv(tmp_path)
    _stub_wfa(monkeypatch, deflated_edge, pbo)
    report = paper_run.run_paper(tmp_path, ["SYN"], Hippocampus())
    assert report["criteria"]["P5_protocol_thresholds"] is False


def test_run_paper_evaluates_full_p1_to_p6(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: FIX-2026-09-23-11 — run_paper must actually evaluate
    P1–P6 (evaluate() was never called; the report held only P5_wfa_pass)."""
    _tiny_csv(tmp_path)
    _stub_wfa(monkeypatch, 0.06, 0.04)
    report = paper_run.run_paper(tmp_path, ["SYN"], Hippocampus())
    for key in (
        "P1_min_trades",
        "P2_min_sessions",
        "P3_no_kill_sessions",
        "P4_daily_loss_sessions",
        "P5_wfa_pass",
        "P5_protocol_thresholds",
        "P6_no_errors",
    ):
        assert key in report["criteria"], key
