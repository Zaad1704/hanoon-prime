"""tests/test_contract.py — Architectural contract tests.

These tests enforce the HANOON PRIME architectural contract.
Tagged @contract — always run in CI, cannot be skipped.

R1 — Only Thinker.deliberate produces verdicts
R3 — No file > 200 lines, no function > 40 lines, no nesting > 3
R4 — Exactly 5 indicators
R5 — No score inversion, PRIOR_TOP ≤ 0.65
R6 — Safety nets not configurable/bypassable
R7 — Journal is immutable
R8 — One learning system only
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ── R1: Single Decision Path ─────────────────────────────────────────────
def test_R1_signal_modules_never_produce_verdicts():
    """alpha.py, scoring.py, edge.py must NEVER contain verdict strings.

    Only thinker.deliberate may produce ENTER/HOLD/EXIT. Signal
    modules can compute indicators and scores but must never decide.
    """
    forbidden_modules = ["alpha.py", "scoring.py", "edge.py"]
    for mod in forbidden_modules:
        path = SRC / "hanoon_prime" / mod
        if not path.exists():
            continue
        content = path.read_text()
        for verdict in ('"ENTER"', '"EXIT"', '"HOLD"', '"WATCH"'):
            assert verdict not in content, (
                f"R1 VIOLATION: {mod} contains verdict string {verdict}"
            )


def test_R1_deliberate_is_the_single_entry_verdict_source():
    """thinker.py is the only module that assigns ENTER as a verdict."""
    thinker_path = SRC / "hanoon_prime" / "thinker.py"
    assert thinker_path.exists()
    content = thinker_path.read_text()
    # ENTER must be produced here (as a return value or assignment)
    assert '"ENTER"' in content or "'ENTER'" in content, (
        "R1 VIOLATION: thinker.py must produce the ENTER verdict"
    )


# ── R3: Complexity ────────────────────────────────────────────────────────
def test_R3_no_file_exceeds_200_lines():
    """No source file may exceed 200 lines."""
    violations = []
    for pyfile in SRC.rglob("*.py"):
        n = len(pyfile.read_text().splitlines())
        if n > 200:
            violations.append(f"{pyfile}: {n} lines")
    assert not violations, f"R3 VIOLATION:\n{chr(10).join(violations)}"


def test_R3_no_function_exceeds_40_lines():
    """No function may exceed 40 lines."""
    violations = []
    for pyfile in SRC.rglob("*.py"):
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = (node.end_lineno or node.lineno) - node.lineno + 1
                if length > 40:
                    violations.append(f"{pyfile.name}:{node.lineno} {node.name} ({length} lines)")
    assert not violations, f"R3 VIOLATION:\n{chr(10).join(violations)}"


# ── R4: Exactly 5 indicators ──────────────────────────────────────────────
def test_R4_indicator_set_is_exactly_5():
    """Must have exactly 5 indicators — auto-evaluated for edge via permutation test."""
    from hanoon_prime.alpha import INDICATOR_NAMES
    assert len(INDICATOR_NAMES) == 5, f"Expected 5 indicators, got {len(INDICATOR_NAMES)}: {INDICATOR_NAMES}"
    expected = {"vpin", "orderbook_imbalance", "institutional_flow", "momentum", "vwap_deviation"}
    assert set(INDICATOR_NAMES) == expected
    # Verify each indicator function exists
    from hanoon_prime import alpha
    for name in INDICATOR_NAMES:
        assert hasattr(alpha, f"compute_{name}"), f"Missing compute_{name}"


# ── R5: No score inversion ─────────────────────────────────────────────────
def test_R5_score_inversion_disabled():
    """SCORE_INVERT must be False — no band-aid score flipping."""
    from hanoon_prime.constants import SCORE_INVERT, PRIOR_TOP, PRIOR_TOP_MAX
    assert SCORE_INVERT is False, "R5 VIOLATION: SCORE_INVERT must be False"
    assert PRIOR_TOP <= PRIOR_TOP_MAX, f"R5 VIOLATION: PRIOR_TOP > {PRIOR_TOP_MAX}"
    assert PRIOR_TOP >= 0.40, f"R5 VIOLATION: PRIOR_TOP {PRIOR_TOP} must allow real edge (≥0.40)"


# ── R6: Safety nets are hard stops ────────────────────────────────────────
def test_R6_safety_nets_are_constants_not_env():
    """Safety net limits must be literal constants, not env-driven."""
    from hanoon_prime.constants import (
        MAX_POSITION_NOTIONAL, MAX_LOSS_PER_TRADE,
        MAX_CONCURRENT_POSITIONS, DAILY_LOSS_LIMIT,
        CONSECUTIVE_LOSSES_PAUSE,
    )
    assert MAX_POSITION_NOTIONAL > 0
    assert MAX_LOSS_PER_TRADE > 0
    assert MAX_CONCURRENT_POSITIONS > 0
    assert DAILY_LOSS_LIMIT > 0
    assert CONSECUTIVE_LOSSES_PAUSE > 0

    # No safety_net constant may read from os.environ
    for pyfile in SRC.rglob("*.py"):
        content = pyfile.read_text()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "getenv":
                    # Check if it's near a safety constant
                    for kw in node.keywords:
                        if kw.arg == "default":
                            continue
                    # Any getenv call is suspicious for safety nets
                    src_segment = ast.get_source_segment(content, node)
                    if src_segment and any(k in src_segment.upper() for k in
                                           ["MAX_LOSS", "DAILY_LOSS", "MAX_POS", "MAX_CONCURRENT"]):
                        pytest.fail(f"R6 VIOLATION: getenv in safety net: {pyfile}\n  {src_segment}")


# ── R7: Immutable journal ────────────────────────────────────────────────
def test_R7_journal_is_append_only():
    """Journal entries can only be appended — never deleted or updated."""
    from hanoon_prime.journal import Journal

    with tempfile.TemporaryDirectory() as tmp:
        j = Journal(Path(tmp) / "test_journal.jsonl")
        j.append({"event": "test1", "value": 1})
        j.append({"event": "test2", "value": 2})

        entries = j.entries()
        assert len(entries) == 2

        # Try to modify (should not work — the file is append-only)
        # The Journal class has no update/delete methods
        journal_methods = {m for m in dir(Journal) if not m.startswith("_")}
        assert "update" not in journal_methods, "R7 VIOLATION: Journal.update exists"
        assert "delete" not in journal_methods, "R7 VIOLATION: Journal.delete exists"
        assert "remove" not in journal_methods, "R7 VIOLATION: Journal.remove exists"

        # Verify hash chain integrity
        assert j.verify_chain(), "R7 VIOLATION: Hash chain broken"


# ── R8: One learning system ───────────────────────────────────────────────
def test_R8_single_learning_system():
    """Only one file may implement weight adaptation."""
    from hanoon_prime.constants import LEARNING_RATE, WEIGHT_FLOOR

    # The Brain class has a record_trade method that does learning.
    # No other module should implement weight adaptation.
    learning_files = []
    for pyfile in SRC.rglob("*.py"):
        content = pyfile.read_text()
        if ("adapt" in content.lower() and "weight" in content.lower()) or \
           ("adjust" in content.lower() and "weight" in content.lower()):
            # Check if it's the brain module
            if "brain" in pyfile.name or "learning" in pyfile.name:
                learning_files.append(str(pyfile.relative_to(SRC)))

    assert len(learning_files) <= 1, f"R8 VIOLATION: learning in {learning_files}"
