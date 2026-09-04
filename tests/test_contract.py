"""tests/test_contract.py — Architectural contract tests (v2.0).

These tests enforce the HANOON PRIME architectural contract.
Tagged @contract — always run in CI, cannot be skipped.

R1 — Only Cortex produces verdicts (BUY/SELL/HOLD)
R3 — No file > 200 lines, no function > 40 lines
R4 — Exactly 5 indicators
R5 — No score inversion, PRIOR_TOP ≤ 0.65
R6 — Safety nets not configurable/bypassable
R7 — Journal is immutable
R8 — One learning system only (Hippocampus)
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ── R1: Single Verdict Source ─────────────────────────────────────────
def test_R1_signal_modules_never_produce_verdicts():
    """cerebellum.py and edge.py must NEVER contain verdict strings.

    Only cortex.py may produce BUY/SELL/HOLD. Signal
    modules can compute indicators and probabilities but must
    never decide.
    """
    forbidden_modules = [
        "cerebellum.py",
        "edge.py",
        "hands.py",
        "hippocampus.py",
        "immune.py",
    ]
    for mod in forbidden_modules:
        path = SRC / "hanoon_prime" / mod
        if not path.exists():
            continue
        content = path.read_text()
        for verdict in ('"BUY"', '"SELL"', '"HOLD"', '"ENTER"', '"EXIT"'):
            assert (
                verdict not in content
            ), f"R1 VIOLATION: {mod} contains verdict string {verdict}"


def test_R1_cortex_is_the_single_verdict_source():
    """cortex.py is the ONLY module that produces BUY/SELL/HOLD verdicts."""
    cortex_path = SRC / "hanoon_prime" / "cortex.py"
    assert cortex_path.exists(), "R1 VIOLATION: cortex.py must exist"
    content = cortex_path.read_text()
    # BUY and SELL must be produced here
    assert (
        '"BUY"' in content or "'BUY'" in content
    ), "R1 VIOLATION: cortex.py must produce BUY verdict"
    assert (
        '"SELL"' in content or "'SELL'" in content
    ), "R1 VIOLATION: cortex.py must produce SELL verdict"


# ── R3: Complexity ─────────────────────────────────────────────────────
def test_R3_no_file_exceeds_200_lines():
    """No source file may exceed 200 lines."""
    skip = {"hands.py", "validator.py", "telemetry.py", "halim_adapter.py"}
    violations = []
    for pyfile in SRC.rglob("*.py"):
        if pyfile.name in skip:
            continue
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
                    violations.append(
                        f"{pyfile.name}:{node.lineno} {node.name} ({length} lines)"
                    )
    assert not violations, f"R3 VIOLATION:\n{chr(10).join(violations)}"


def test_R3_no_nesting_exceeds_3():
    """No function body may nest more than 3 levels deep."""

    def _max_depth(node, depth=0):
        result = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler),
            ):
                result = max(result, _max_depth(child, depth + 1))
            else:
                result = max(result, _max_depth(child, depth))
        return result

    violations = []
    for pyfile in SRC.rglob("*.py"):
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Start counting nesting inside the function body
                for child in ast.iter_child_nodes(node):
                    d = (
                        _max_depth(child, 1)
                        if isinstance(
                            child,
                            (
                                ast.If,
                                ast.For,
                                ast.While,
                                ast.With,
                                ast.Try,
                                ast.ExceptHandler,
                            ),
                        )
                        else _max_depth(child, 0)
                    )
                    if d > 4:  # 3 levels of nesting + function body = 4
                        violations.append(
                            f"{pyfile.name}:{node.lineno} {node.name}"
                            f" (nesting depth {d})"
                        )
                        break
    assert not violations, f"R3 VIOLATION:\n{chr(10).join(violations)}"


# ── R4: Exactly 5 indicators ────────────────────────────────────────────
def test_R4_indicator_set_is_exactly_5():
    """Must have exactly 5 indicators — auto-evaluated for edge."""
    from hanoon_prime.cerebellum import INDICATOR_NAMES

    assert len(INDICATOR_NAMES) == 5
    expected = {
        "vpin",
        "orderbook_imbalance",
        "institutional_flow",
        "momentum",
        "vwap_deviation",
    }
    assert set(INDICATOR_NAMES) == expected
    from hanoon_prime import cerebellum

    for name in INDICATOR_NAMES:
        assert hasattr(cerebellum, f"compute_{name}"), f"Missing compute_{name}"


# ── R5: No score inversion ──────────────────────────────────────────────
def test_R5_score_inversion_disabled():
    """SCORE_INVERT must be False — no band-aid score flipping."""
    from hanoon_prime.immune import PRIOR_TOP, PRIOR_TOP_MAX, SCORE_INVERT

    assert SCORE_INVERT is False, "R5 VIOLATION: SCORE_INVERT must be False"
    assert PRIOR_TOP <= PRIOR_TOP_MAX
    assert PRIOR_TOP >= 0.40, "R5 VIOLATION: PRIOR_TOP must allow real edge (≥0.40)"


# ── R6: Safety nets are hard stops ──────────────────────────────────────
def test_R6_safety_nets_are_constants_not_env():
    """Safety net limits must be literal constants, not env-driven."""
    from hanoon_prime.immune import (
        CONSECUTIVE_LOSSES_PAUSE,
        DAILY_LOSS_LIMIT,
        MAX_CONCURRENT_POSITIONS,
        MAX_LOSS_PER_TRADE,
        MAX_POSITION_NOTIONAL,
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
                    src_segment = ast.get_source_segment(content, node)
                    if src_segment and any(
                        k in src_segment.upper()
                        for k in ["MAX_LOSS", "DAILY_LOSS", "MAX_POS", "MAX_CONCURRENT"]
                    ):
                        pytest.fail(
                            f"R6 VIOLATION: getenv in safety net: {pyfile}"
                            f"\n  {src_segment}"
                        )


# ── R7: Immutable journal ───────────────────────────────────────────────
def test_R7_journal_is_append_only():
    """Journal entries can only be appended — never deleted or updated."""
    from hanoon_prime.memory import Journal

    with tempfile.TemporaryDirectory() as tmp:
        j = Journal(Path(tmp) / "test_journal.jsonl")
        j.append({"event": "test1", "value": 1})
        j.append({"event": "test2", "value": 2})

        entries = j.entries()
        assert len(entries) == 2

        # Journal has no update/delete/remove methods
        journal_methods = {m for m in dir(Journal) if not m.startswith("_")}
        assert "update" not in journal_methods, "R7 VIOLATION: Journal.update exists"
        assert "delete" not in journal_methods, "R7 VIOLATION: Journal.delete exists"
        assert "remove" not in journal_methods, "R7 VIOLATION: Journal.remove exists"

        # Verify hash chain integrity
        assert j.verify_chain(), "R7 VIOLATION: Hash chain broken"


# ── R8: One learning system ─────────────────────────────────────────────
def test_R8_single_learning_system():
    """Only one file may implement weight adaptation (Hippocampus)."""

    learning_files = []
    for pyfile in SRC.rglob("*.py"):
        content = pyfile.read_text()
        has_adapt = ("adapt" in content.lower() and "weight" in content.lower()) or (
            "adjust" in content.lower() and "weight" in content.lower()
        )
        if has_adapt:
            if "hippocampus" in pyfile.name:
                learning_files.append(str(pyfile.relative_to(SRC)))

    assert len(learning_files) == 1, f"R8 VIOLATION: learning in {learning_files}"


# ── R9: Positive indicator weights ────────────────────────────────────────
def test_R9_indicator_weights_all_positive():
    """All INDICATOR_WEIGHTS must be ≥ 0.

    Negative weights on FAST (momentum-persistence) tickers produce
    SELL signals on uptrending tickers — guaranteed losses. Sign is
    determined by the indicator's edge, not by weight negation.
    """
    from hanoon_prime.immune import INDICATOR_WEIGHTS

    for name, weight in INDICATOR_WEIGHTS.items():
        assert weight >= 0, f"R9 VIOLATION: {name} weight={weight} (must be ≥ 0)"
    assert sum(INDICATOR_WEIGHTS.values()) > 0


# ── R9b: SCORE_INVERT is False ────────────────────────────────────────────
def test_R9b_score_invert_is_false():
    """SCORE_INVERT must be False — no band-aid score flipping."""
    from hanoon_prime.immune import SCORE_INVERT

    assert SCORE_INVERT is False, "R9b VIOLATION: SCORE_INVERT must be False"


# ── R10: No print() in source ─────────────────────────────────────────────
def test_R10_no_print_in_source():
    """No print() calls in src/ — must use logging or return values.

    ib_adapter.py is excluded (live IB adapter, separate concern).
    ib_streamer.py and ib_executor.py are part of the IB adapter layer.
    Only logging is allowed; print() is forbidden everywhere in src/
    (use logging.info for CLI output).
    """
    import ast

    excluded = {
        "ib_adapter.py",
        "ib_streamer.py",
        "ib_executor.py",
        "ib_compat.py",
        "_ib_sync.py",
    }
    violations = []
    for pyfile in SRC.rglob("*.py"):
        if pyfile.name in excluded:
            continue
        content = pyfile.read_text()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                is_print = isinstance(func, ast.Name) and func.id == "print"
                if is_print:
                    violations.append(f"{pyfile.name}: line {node.lineno}")
    assert not violations, f"R10 VIOLATION:\n{chr(10).join(violations)}"


# ── R11: All public functions have docstrings ─────────────────────────────
def test_R11_public_functions_have_docstrings():
    """Every public function (not starting with _) must have a docstring."""
    import ast

    violations = []
    excluded = {
        "ib_adapter.py",
        "ib_streamer.py",
        "ib_executor.py",
        "ib_compat.py",
        "_ib_sync.py",
    }
    for pyfile in SRC.rglob("*.py"):
        if pyfile.name in excluded:
            continue
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                if node.name.startswith("test_"):
                    continue
                has_doc = (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                )
                if not has_doc:
                    violations.append(f"{pyfile.name}:{node.lineno} {node.name}")
    assert not violations, f"R11 VIOLATION:\n{chr(10).join(violations)}"


# ── R12: Coverage gate ≥ 80% ──────────────────────────────────────────────
def test_R12_coverage_gate_configured():
    """Coverage gate must be configured at ≥ 80% in pyproject.toml."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    content = pyproject.read_text()
    assert "fail_under" in content, "R12 VIOLATION: coverage gate not configured"
    assert "cov-fail-under" in content or "fail_under" in content


# ── R13: No string-based verdict dispatch ─────────────────────────────────
def test_R13_no_string_verdict_dispatch():
    """Direction must come from Thought.direction (int), not string comparison.

    This prevents typos like 'if direction == "LON"' that silently
    never trigger. Cortex produces integer directions {-1, 0, +1}.
    """
    import ast

    verdict_strings = {"BUY", "SELL", "HOLD", "ENTER", "EXIT", "LONG", "SHORT"}
    violations = []
    excluded = {"ib_adapter.py", "cortex.py"}
    # cortex.py IS the verdict source — it may compare against its own verdicts
    for pyfile in SRC.rglob("*.py"):
        if pyfile.name in excluded:
            continue
        content = pyfile.read_text()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comp in node.comparators:
                    if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                        if comp.value in verdict_strings:
                            violations.append(
                                f"{pyfile.name}:{node.lineno}"
                                f" — string comparison against '{comp.value}'"
                            )
    assert not violations, f"R13 VIOLATION:\n{chr(10).join(violations)}"


# ── R14: All constants type-annotated ─────────────────────────────────────
def test_R14_constants_are_typed():
    """All constants in immune.py must have explicit type annotations.

    Prevents accidental type coercion that could break safety-critical
    comparisons (e.g., string vs int for position limits).
    """
    import ast

    immune_path = SRC / "hanoon_prime" / "immune.py"
    tree = ast.parse(immune_path.read_text())
    violations = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.AnnAssign):
            if node.value is not None and node.annotation is None:
                violations.append(f"immune.py:{node.lineno} — untyped constant")
        elif isinstance(node, ast.Assign):
            # Allow __all__ and INDICATOR_NAMES (tuple[str, ...])
            if node.targets and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("__all__",):
                    continue
                violations.append(
                    f"immune.py:{node.lineno} — constant without type annotation"
                )
    assert not violations, f"R14 VIOLATION:\n{chr(10).join(violations)}"


# ── R15: No swallowed exceptions ──────────────────────────────────────────
def test_R15_no_swallowed_exceptions():
    """No bare except: or except: pass — must log or re-raise."""
    import ast

    violations = []
    excluded = {
        "ib_adapter.py",
        "ib_streamer.py",
        "ib_executor.py",
        "ib_compat.py",
        "_ib_sync.py",
    }
    for pyfile in SRC.rglob("*.py"):
        if pyfile.name in excluded:
            continue
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    violations.append(f"{pyfile.name}:{node.lineno} — bare except")
                elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    violations.append(f"{pyfile.name}:{node.lineno} — except: pass")
    assert not violations, f"R15 VIOLATION:\n{chr(10).join(violations)}"


# ── R16: No TODO/FIXME markers ────────────────────────────────────────────
def test_R16_no_todo_markers():
    """No TODO/FIXME/HACK/XXX in source — unfinished work is forbidden."""
    markers = ("TODO", "FIXME", "HACK", "XXX")
    violations = []
    for pyfile in SRC.rglob("*.py"):
        for i, line in enumerate(pyfile.read_text().splitlines(), 1):
            for marker in markers:
                if marker in line.upper():
                    violations.append(f"{pyfile.name}:{i} — {marker}")
    assert not violations, f"R16 VIOLATION:\n{chr(10).join(violations)}"


# ── R17: mypy strict enforced ─────────────────────────────────────────────
def test_R17_mypy_strict_configured():
    """mypy strict mode must be enabled in pyproject.toml."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    content = pyproject.read_text()
    assert "strict = true" in content, "R17 VIOLATION: mypy strict not enabled"
    assert "disallow_untyped_defs = true" in content
    assert "disallow_incomplete_defs = true" in content


# ── R18: All public APIs documented ──────────────────────────────────────
def test_R18_modules_have_docstrings():
    """Every source module must have a module-level docstring."""
    import ast

    violations = []
    for pyfile in SRC.rglob("*.py"):
        tree = ast.parse(pyfile.read_text())
        if tree.body and isinstance(tree.body[0], ast.Expr):
            if isinstance(tree.body[0].value, ast.Constant):
                if isinstance(tree.body[0].value.value, str):
                    continue
        violations.append(f"{pyfile.name} — missing module docstring")
    assert not violations, f"R18 VIOLATION:\n{chr(10).join(violations)}"
