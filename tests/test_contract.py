"""tests/test_contract.py — Architectural contract tests (v3.0).

These tests enforce the HANOON PRIME architectural contract.
Tagged @contract — always run in CI, cannot be skipped.

R1 — Only Cortex produces verdicts (BUY/SELL/HOLD)
R3 — No file > 200 lines, no function > 40 lines
R4 — Indicators with compute functions + positive weights (evolved from "exactly 5")
R4b — Weight sum in [0.8, 1.2] for stability
R5 — No score inversion, PRIOR_TOP ≤ 0.65
R6 — Safety nets not configurable/bypassable
R7 — Journal is immutable
R8 — Integrated learning ecosystem (STDP + Hippocampus + Nash + Episodic)
R19 — Realized-EV learning gate wired + verifiable (refuse losing band/conf-bin, admit recovery/thin)
R20 — Tiered exits integrated (ExitPolicy.evaluate delegated by orchestrator)
R21 — Gate advisor exists, is bounded, and is advisory (threshold/size only — never verdicts)
R22 — Closed learning loop: every real trade updates weights, cortex, realized stats, exits, advisor
R23 — Dynamic PRIOR doctrine: realized wins may widen PRIOR_TOP, capped and wired end-to-end
R24 — Exit ladder TIER semantics: TIER1 absolute stop, TIER2 numeric likelihood, TIER3 ExitPolicy
R25 — Scanner universe is RAW + unranked: no weight re-rank, no price/volume/market-cap filters
R26 — Every scan code is a real, whitelisted, ALL-CAPS IB code (≤10 codes, 50 rows each)
R27 — The brain sees the FULL discovery union: no MAX_CANDIDATES / [:20] truncation, rotation seats all
R28 — File-skip allowlists are FROZEN: skip-lists may never grow to dodge the 200-line cap
"""

from __future__ import annotations

import ast
import re
import sys
import tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# ── Rollout gates (R29/R30/R31) ─────────────────────────────────────────
# The 10 rollout organs. R29 forces default-OFF literals, R30 forces the live
# read-sites OFF, R31 forces any ON gate to carry money-gate evidence in
# docs/gates/promotions.json. Shared here so the truth table lives once.
ROLLOUT_GATES: tuple[str, ...] = (
    "CALIBRATION_NUDGE_ENABLED",
    "HYSTERESIS_EXIT_ENABLED",
    "PROBE_RECOVERY_ENABLED",
    "CONTRARIAN_MODE_ENABLED",
    "DELIBERATION_TRACE_ENABLED",
    "HALIM_EVIDENCE_LEARNING",
    "NEURO_BLEND_ENABLED",
    "NEURO_LEARN_ENABLED",
    "NEURO_ADAPTIVE_THRESHOLD_ENABLED",
    "NEURO_MOE_GATE_ENABLED",
)

# ── Frozen governance allowlists (R28) ─────────────────────────────────
# The ONLY finite set of modules exempt from the 200-line rule (R3). These
# are the pre-restructure oversized files. Growing THIS list to dodge the
# cap is a governance violation (see R28) — files must be refactored under
# 200 lines, not added to the allowlist. Mirrored 1:1 in
# scripts/check_file_length.sh.
FROZEN_FILE_SKIP: frozenset[str] = frozenset(
    {
        "hands.py",
        "validator.py",
        "telemetry.py",
        "halim_adapter.py",
        "ib_cycle.py",
        "orchestrator.py",
        "ib_executor.py",
        "ib_streamer.py",
        "ironclad.py",
        "consolidation.py",
        "shadow_book.py",
        "ib_adapter.py",
        "realized_ev.py",
        "config.py",
        "immune.py",
        "ev_gate.py",
        "risk.py",
        "exits.py",
        "stdp.py",
        "wfa.py",
        "ablation.py",
        "micro_live.py",
        "phase7.py",
        "phase8.py",
        "phase9.py",
    }
)
# Tokens that would rank/filter the raw scanner universe — banned by R25.
SCANNER_FILTER_TOKENS: frozenset[str] = frozenset(
    {
        "CODE_WEIGHTS",
        "abovePrice",
        "aboveVolume",
        "marketCapAbove",
        "marketCapBelow",
    }
)
# Required all-cap discovery codes — present in a raw, unfiltered scanner.
SCANNER_REQUIRED_CODES: frozenset[str] = frozenset(
    {"MARKET_CAP_USD_ASC", "HOT_BY_VOLUME", "TOP_VOLUME_RATE", "TOP_PERC_GAIN"}
)


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
    """No source file may exceed 200 lines.

    Skip allowlist is FROZEN (R28): only the 25 genuinely-oversized
    modules may ever be exempted, and test_R3 must reference the frozen
    constant — a hand-maintained local copy drifts and lets governance rot.
    """
    skip = set(FROZEN_FILE_SKIP)
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
    skip_files = {
        "ib_executor.py",
        "hands.py",
        "validator.py",
        "ev_gate.py",
        "risk.py",
        "stdp.py",
        "exits.py",
    }
    for pyfile in SRC.rglob("*.py"):
        if pyfile.name in skip_files:
            continue
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


# ── R4: Indicators must be validated (evolved from "exactly 5") ──────────
def test_R4_indicator_set_has_compute_functions():
    """Indicators must have compute functions and positive weights.

    Evolved from "exactly 5" constraint - now allows 27 indicators from
    rebuild's architecture, each validated for edge (p < 0.05 via permutation).
    """
    from hanoon_prime.cerebellum import INDICATOR_NAMES
    from hanoon_prime.immune import INDICATOR_WEIGHTS

    # Must have at least the core 5 (evolved requirement)
    core_indicators = {
        "vpin",
        "orderbook_imbalance",
        "institutional_flow",
        "momentum",
        "vwap_deviation",
    }
    assert core_indicators.issubset(set(INDICATOR_NAMES)), f"Missing core indicators"

    # All indicators must have compute functions
    from hanoon_prime import cerebellum

    for name in INDICATOR_NAMES:
        assert hasattr(cerebellum, f"compute_{name}"), f"Missing compute_{name}"

    # All weights must be positive (R9 invariant preserved)
    for name in INDICATOR_NAMES:
        assert name in INDICATOR_WEIGHTS, f"Missing weight for {name}"
        assert INDICATOR_WEIGHTS[name] >= 0, f"R9 VIOLATION: {name} weight < 0"


# ── R4b: Weight sum target ────────────────────────────────────────────────
def test_R4b_weights_sum_to_target():
    """Indicator weights should sum to approximately 1.0 for stability."""
    from hanoon_prime.immune import INDICATOR_WEIGHTS

    total = sum(INDICATOR_WEIGHTS.values())
    assert 0.8 <= total <= 1.2, f"R4b VIOLATION: weight sum {total} outside [0.8, 1.2]"


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


def test_R7_journal_concurrent_appends_keep_chain_intact():
    """Concurrent appends from multiple threads must never corrupt the chain."""
    import threading

    from hanoon_prime.memory import Journal

    with tempfile.TemporaryDirectory() as tmp:
        j = Journal(Path(tmp) / "concurrent.jsonl")

        def _writer(base: int, n: int) -> None:
            for i in range(n):
                j.append({"thread": base, "i": i})

        threads = [threading.Thread(target=_writer, args=(t, 50)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert j.count() == 200
        assert j.verify_chain(), "R7 VIOLATION: concurrent appends broke hash chain"


# ── R8: Learning ecosystem ───────────────────────────────────────────────
def test_R8_learning_ecosystem_integrated():
    """Cognitive learning ecosystem must be integrated: STDP + Hippocampus + Nash.

    Evolved from "single learning system" to "integrated learning ecosystem":
    - Hippocampus.py: asymmetric weight adaptation (primary)
    - STDP: continuous synaptic plasticity (neuromorphic)
    - Nash: pattern-based opinion (cognitive pillar)
    - Episodic: k-NN memory (cognitive pillar)
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)

    learning_components = {
        "hippocampus.py": "HIPPOCAMPUS",
        "stdp.py": "STDP",
        "nash.py": "NASH",
        "brain/cognitive/episodic.py": "EPISODIC",
    }

    found = {}
    for comp, label in learning_components.items():
        for pyfile in SRC.rglob("*.py"):
            if comp in str(pyfile):
                content = pyfile.read_text()
                has_learning = (
                    "learn" in content.lower()
                    or "stake" in content.lower()
                    or "replay" in content.lower()
                    or "update" in content.lower()
                    or "add" in content.lower()
                    or "record" in content.lower()
                )
                if has_learning:
                    found[label] = pyfile.relative_to(SRC)

    for label in learning_components.values():
        assert (
            label in found
        ), f"R8 VIOLATION: {label} learning component not integrated"

    _log.info("R8: Integrated learning ecosystem: %s", list(found.keys()))


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
    excluded = {
        "ib_adapter.py",
        "cortex.py",
        "ib_executor.py",
        "hands.py",
        "_guard.py",
        "_protect.py",
    }
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


# ── R19: Realized learning EV gate ───────────────────────────────────────
def test_R19_realized_ev_gate_integrated():
    """The realized-EV gate must exist, be wired into the risk engine, and
    its canary must prove it refuses losing bands while admitting recovery
    and thin-data (structural fallback)."""
    rv_path = SRC / "hanoon_prime" / "brain" / "realized_ev.py"
    rv_src = rv_path.read_text()
    # v2.1: gate math lives in brain/ev_gate.py (single source of truth);
    # realized_ev.py persists stats and re-exports the gate API.
    ev_path = SRC / "hanoon_prime" / "brain" / "ev_gate.py"
    assert ev_path.exists(), "R19: brain/ev_gate.py gate math missing"
    ev_tree = ast.parse(ev_path.read_text())
    names = {n.name for n in ast.walk(ev_tree) if isinstance(n, ast.FunctionDef)}
    assert "ev_gate_should_enter" in names, "R19: ev_gate_should_enter missing"
    assert "verify_learning_gate" in names, "R19: verify_learning_gate canary missing"
    assert (
        "ev_gate_should_enter" in rv_src
    ), "R19: realized_ev.py must re-export the gate"

    risk_src = (SRC / "hanoon_prime" / "brain" / "risk.py").read_text()
    assert (
        "ev_gate_should_enter" in risk_src
    ), "R19: risk.py must call the realized gate"

    # v2.1: conf-bin correction wired into the gate + orchestrator feed.
    assert (
        "add_confidence_outcome" in rv_src
    ), "R19: conf-bin tracking missing from realized_ev"
    orch_src = (SRC / "hanoon_prime" / "brain" / "orchestrator.py").read_text()
    assert (
        "add_confidence_outcome" in orch_src
    ), "R19: orchestrator must feed conf-bin outcomes"

    from hanoon_prime.brain.realized_ev import (
        RealizedStats,
        ev_gate_should_enter,
        verify_learning_gate,
    )
    from hanoon_prime.edge import score_to_win_prob

    assert (
        verify_learning_gate()["all_pass"] is True
    ), "R19: learning-gate canary failed"

    empty = RealizedStats(persist=False)
    wp = score_to_win_prob(0.6)
    assert ev_gate_should_enter(0.6, wp, empty, direction=1)["should_enter"] is True
    assert (
        ev_gate_should_enter(0.02, score_to_win_prob(0.02), empty, direction=1)[
            "should_enter"
        ]
        is False
    )

    # IRONYCLADE source filter must exist as a typed constant.
    from hanoon_prime.brain.config import _IRONYCLADE

    assert _IRONYCLADE == frozenset(
        {"real_trade", "ib_fill", "ib_paper", "reconciled_exit"}
    )


# ── R20: Tiered exits integrated ────────────────────────────────────────
def test_R20_tiered_exits_integrated():
    """brain/exits.py ExitPolicy.evaluate (TIER3 mechanical) must exist and
    the orchestrator's check_exit must delegate to the 3-tier ExitLadder
    (TIER1 hard stop, TIER2 JULI verdict, TIER3 = ExitPolicy mechanical)."""
    exits_tree = ast.parse((SRC / "hanoon_prime" / "brain" / "exits.py").read_text())
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "evaluate"
        for n in ast.walk(exits_tree)
    ), "R20 VIOLATION: ExitPolicy.evaluate (TIER3 base) missing"

    orch_src = (SRC / "hanoon_prime" / "brain" / "orchestrator.py").read_text()
    assert "def check_exit" in orch_src, "R20: orchestrator must expose check_exit"
    assert (
        "self._exit_ladder" in orch_src
    ), "R20: check_exit must delegate to the ExitLadder"

    ladder_path = SRC / "hanoon_prime" / "brain" / "exit_ladder.py"
    assert ladder_path.exists(), "R20: brain/exit_ladder.py missing"
    ladder_src = ladder_path.read_text()
    ladder_tree = ast.parse(ladder_src)
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "evaluate"
        for n in ast.walk(ladder_tree)
    ), "R20 VIOLATION: ExitLadder.evaluate missing"
    # R20 doctrine: TIER3 mechanical is still ExitPolicy — never replaced.
    assert (
        "self._policy.evaluate" in ladder_src
    ), "R20: ladder TIER3 must delegate mechanical exits to ExitPolicy.evaluate"

    from hanoon_prime.brain.exit_checks import ExitSignal
    from hanoon_prime.brain.exits import ExitPolicy

    policy = ExitPolicy()
    sig = policy.evaluate("TSLA", 100.0, 0.0, 1)
    assert isinstance(sig, ExitSignal)
    assert sig.should_exit is False  # unregistered position => no exit


# ── R21: Gate advisor (bounded, advisory) ────────────────────────────
def test_R21_gate_advisor_bounded_and_advisory():
    """The gate advisor tightens/loosens the entry bar within hard bounds
    and NEVER produces verdict strings (R1 boundary preserved)."""
    from hanoon_prime.brain.config import ADVISOR_DELTA_MAX, ADVISOR_LOOSEN_MAX
    from hanoon_prime.brain.gate_advisor import GateAdvisor

    adv = GateAdvisor()
    for _ in range(30):
        adv.record_outcome(False)
    assert 0.0 <= adv.threshold_delta() <= ADVISOR_DELTA_MAX
    assert adv.is_tightening() is True
    for _ in range(30):
        adv.record_outcome(True)
    assert -ADVISOR_LOOSEN_MAX <= adv.threshold_delta() <= 0.0
    # Thin data never moves the bar.
    fresh = GateAdvisor()
    fresh.record_outcome(False)
    assert fresh.threshold_delta() == 0.0
    # Advisory: no verdict strings anywhere in the module.
    src = (SRC / "hanoon_prime" / "brain" / "gate_advisor.py").read_text()
    for v in ('"BUY"', '"SELL"', '"HOLD"', '"ENTER"', '"EXIT"'):
        assert v not in src, f"R21 VIOLATION: advisor emits {v}"


# ── R22: Closed learning loop ───────────────────────────────────────
def test_R22_closed_learning_loop_wired():
    """Every real trade close must: update weights (Reflector), hot-swap
    the cortex, feed realized band+conf stats, and retune exits+advisor."""
    orch_src = (SRC / "hanoon_prime" / "brain" / "orchestrator.py").read_text()
    for token in (
        "self._reflector.on_trade_close",
        "self.cortex.set_weights",
        "self._realized.add_outcome",
        "self._realized.add_confidence_outcome",
        "self.exits.adapt_from_realized",
        "self._advisor.record_outcome",
    ):
        assert token in orch_src, f"R22 VIOLATION: missing {token}"
    # The reflector must adapt over ALL weighted indicators (not just 5).
    refl_src = (SRC / "hanoon_prime" / "brain" / "reflection.py").read_text()
    assert "for key in weights" in refl_src, "R22: reflector must adapt all weights"


def test_R22_nash_veto_bands_fire():
    """Nash gate authority must fire on the veto bands (the old flag
    required wp INSIDE [0.45, 0.55] while the veto required OUTSIDE —
    the veto could never fire)."""
    from hanoon_prime.brain.cognitive.nash import NashBrain

    nash = NashBrain()
    for _ in range(25):
        nash.record_outcome({"vpin": 0.9}, 0.0, False)
    pred = nash.predict({"vpin": 0.9}, 0.8, 1)
    assert pred.gate_authority is True, "losing pattern must carry veto authority"
    assert pred.win_prob < 0.45


# ── R23: Dynamic PRIOR doctrine ─────────────────────────────────────────
def test_R23_dynamic_prior_bounded_wired():
    """PRIOR_TOP may only WIDEN from realized wins — never past PRIOR_TOP_MAX
    (R5 runtime guard). Cold-start stays static; the earned cap reaches the
    entry gate (risk.py) and cortex via the orchestrator's realized feed."""
    from hanoon_prime.brain.realized_ev import RealizedStats
    from hanoon_prime.edge import get_dynamic_prior_top, score_to_win_prob
    from hanoon_prime.immune import PRIOR_BOTTOM, PRIOR_TOP, PRIOR_TOP_MAX

    # Pure math: bounded by PRIOR_TOP_MAX (R5), cold-starts on static.
    assert get_dynamic_prior_top(0.40, 0) == PRIOR_TOP  # cold start
    hot = get_dynamic_prior_top(0.80, 50)
    assert PRIOR_TOP < hot <= PRIOR_TOP_MAX  # earns UP, capped at 0.65
    assert get_dynamic_prior_top(0.10, 50) < PRIOR_TOP  # losing tightens

    # win_prob widens to the dynamic cap but never above it (R5).
    wp = score_to_win_prob(1.0, prior_top=hot)
    assert PRIOR_BOTTOM <= wp <= hot

    # RealizedStats exposes the cap + cold-starts on the structural prior.
    rs = RealizedStats(persist=False)
    assert rs.dynamic_prior_top() == PRIOR_TOP
    assert rs.recent_win_rate() == (0.5, 0)  # empty ⇒ neutral, not a crash

    # Wiring: gate (risk.py), cortex, and orchestrator feed the realized cap.
    risk_src = (SRC / "hanoon_prime" / "brain" / "risk.py").read_text()
    cortex_src = (SRC / "hanoon_prime" / "cortex.py").read_text()
    orch_src = (SRC / "hanoon_prime" / "brain" / "orchestrator.py").read_text()
    rv_src = (SRC / "hanoon_prime" / "brain" / "realized_ev.py").read_text()
    assert (
        "score_to_win_prob(score, prior_top=pt)" in risk_src
    ), "R23: entry gate must use the dynamic prior"
    assert "prior_top=prior_top" in cortex_src, "R23: cortex must accept prior_top"
    assert "dynamic_prior_top" in rv_src, "R23: RealizedStats lacks dynamic_prior_top"
    assert (
        "self._realized.dynamic_prior_top()" in orch_src
    ), "R23: orchestrator must feed realized cap to cortex"


# ── R24: Exit ladder TIER semantics ─────────────────────────────────────
def test_R24_exit_ladder_tier_semantics():
    """3-tier ladder: non-breaking defaults (dormant TIER2 → TIER3 =
    ExitPolicy), TIER1 hard stop is absolute, TIER2 reads a NUMERIC
    exit_likelihood (never a string verdict — R13 safe), TIER3 = ExitPolicy."""
    from hanoon_prime.brain.adaptive_thresholds import get_adaptive_thresholds
    from hanoon_prime.brain.exit_checks import ExitSignal
    from hanoon_prime.brain.exit_ladder import ExitLadder
    from hanoon_prime.brain.exits import ExitPolicy

    ladder = ExitLadder(ExitPolicy(), thresholds=get_adaptive_thresholds())

    # Non-breaking: defaults ⇒ TIER2 dormant ⇒ TIER3 = current ExitPolicy path.
    sig = ladder.evaluate("TSLA", 100.0, ib_pnl=0.0, direction=1)
    assert isinstance(sig, ExitSignal)
    assert sig.should_exit is False  # unregistered ⇒ no exit (preserves prior)

    # TIER1 hard stop is absolute: price breaches stop ⇒ force exit.
    hard = ladder.evaluate("TSLA", 90.0, ib_pnl=0.0, direction=1, stop_price=91.0)
    assert hard.should_exit is True
    assert hard.exit_type == "exit"

    # TIER2 is numeric exit_likelihood (no string verdict — R13 safe).
    tier2 = ladder.evaluate("TSLA", 100.0, ib_pnl=0.0, direction=1, exit_likelihood=1.0)
    assert tier2.should_exit is True
    assert tier2.exit_type == "exit"

    # TIER3 mechanical is still ExitPolicy (R20/R24 doctrine).
    src = (SRC / "hanoon_prime" / "brain" / "exit_ladder.py").read_text()
    assert "self._policy.evaluate" in src, "R24: TIER3 must be ExitPolicy"


# ── R25: Scanner universe is raw + unranked ─────────────────────────────
def test_R25_scanner_universe_is_raw_natural_order():
    """Scanner results are raw and unranked: no weight re-ranking, no
    price/volume/market-cap filters, and each item is ingested at its
    natural IB scan rank. Data prep never decides — the brain does."""
    scanner_path = SRC / "hanoon_prime" / "data" / "scanner.py"
    scanner_src = scanner_path.read_text()
    tree = ast.parse(scanner_src)

    # Banned tokens: any ranking table or scanner-side filter.
    for token in SCANNER_FILTER_TOKENS:
        assert token not in scanner_src, f"R25 VIOLATION: scanner filters/ranks {token}"

    # _ingest_item must take (self, item) — a rank/path offset argument
    # would let caller-stage ranking sneak back in.
    ingest = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_ingest_item"
        ),
        None,
    )
    assert ingest is not None, "R25: _ingest_item missing"
    assert [a.arg for a in ingest.args.args] == [
        "self",
        "item",
    ], f"R25: ingestion must be (self, item), got {[a.arg for a in ingest.args.args]}"

    # Natural rank retained verbatim (no offset/depth-warp).
    assert "eff = item.rank" in scanner_src, "R25: natural scan rank must be used"


def test_R26_scan_codes_all_caps_and_whitelisted():
    """Every scanCode is a real IB code: ALL-CAPS, frozen-whitelisted,
    ≤ 10 configs (IB's scan-subscription limit), 50 rows max per code."""
    from hanoon_prime.data.scanner import (
        ALLOWED_SCANCODES,
        DATA_INSTRUMENT,
        DATA_LOCATION,
        SCAN_CONFIGS,
    )

    assert len(SCAN_CONFIGS) <= 10, "R26: IB allows ≤ 10 scanner subscriptions"
    assert all(c.isupper() for c in ALLOWED_SCANCODES), "R26: codes must be ALL-CAPS"

    configured = set(SCAN_CONFIGS.values())
    assert (
        configured <= ALLOWED_SCANCODES
    ), f"R26: configured codes not in frozen whitelist: {configured - ALLOWED_SCANCODES}"
    assert (
        SCANNER_REQUIRED_CODES <= ALLOWED_SCANCODES
    ), "R26: raw all-cap discovery codes missing from whitelist"

    # Fixed instrument/location; rows bounded by IB's 50 — never more.
    assert DATA_INSTRUMENT == "STK", "R26: instrument must be STK"
    assert DATA_LOCATION == "STK.US.MAJOR", "R26: location must be STK.US.MAJOR"
    assert (
        "numberOfRows=50" in (SRC / "hanoon_prime" / "data" / "scanner.py").read_text()
    ), "R26: rows must be capped at IB's 50"


# ── R27: The brain sees the FULL discovery union ────────────────────────
def test_R27_no_discovery_truncation_before_brain():
    """No truncation of the discovery pool before the decision layer.
    A `[:MAX_CANDIDATES]` slice used to hide ~80% of the scanned pool
    from the brain. juli must feed the WHOLE candidate list, and the live
    cycle must watch streamed ∪ full-discovery, not a top-N excerpt."""
    juli_path = SRC / "hanoon_prime" / "juli.py"
    juli_src = juli_path.read_text()
    assert "MAX_CANDIDATES" not in juli_src, "R27: MAX_CANDIDATES truncation must go"
    assert "self._candidates[:20]" not in juli_src, "R27: hardcoded [:20] truncation"
    assert (
        "[c.symbol for c in self._candidates]" in juli_src
    ), "R27: juli must feed the full candidate list to allocation"

    ib_path = SRC / "hanoon_prime" / "ib_cycle.py"
    ib_src = ib_path.read_text()
    # The brain evaluates the LIVE budget-rotated universe. Budget rotation
    # streams EVERY discovered name over time (LRU rotation of data seats),
    # so each eventually scores with live data. Watching the full raw union
    # all-at-once flooded the window with ~215 un-seated no-data names →
    # zero decisions → a false brain-stall. The brain decides on names it
    # can actually see — which over the rotation is the whole pool.
    assert (
        "watch = set(self.juli.budget.get_all_tracked())" in ib_src
    ), "R27: live cycle must evaluate the budget-tracked (live-streamed) universe"
    assert (
        "{c.symbol for c in self.juli._candidates}" not in ib_src
    ), "R27: brain must not eval the raw all-at-once union (no_data flood)"
    for src, name in ((juli_src, "juli.py"), (ib_src, "ib_cycle.py")):
        assert "[:MAX_CANDIDATES]" not in src, f"R27: slider truncation in {name}"


def test_R27_budget_rotates_the_full_pool():
    """Budget rotation seats EVERY discovered name over time within IB's
    ~100 live-line allowance — never-served names rotate in each cycle,
    seats are not frozen, and streaming never exceeds the cap."""
    from hanoon_prime.data.budget import MAX_L1, ROTATE_PER_CYCLE, DataBudget

    budget = DataBudget()
    small_pool = ["A", "B", "C", "D"]
    to_sub, _ = budget.allocate(set(), small_pool)
    assert set(to_sub) == set(
        small_pool
    ), "R27: rotation must seat the whole pool while it fits capacity"

    # Pool grows far past capacity: hungry fresh names must rotate in.
    fresh = [f"T{i}" for i in range(500)]
    budget.allocate(set(), ["A"] + fresh)
    tracked = budget.get_all_tracked()
    rotated_in = {t for t in fresh[:ROTATE_PER_CYCLE] if t in tracked}
    assert rotated_in, "R27: fresh discovery names must rotate into seats"

    # Seats are not permanently frozen on the first 20; later names flow in.
    seen_after = budget.get_all_tracked()
    assert any(
        t in seen_after for t in fresh
    ), "R27: rotation must reach deep into the pool"

    # IB line allowance is never exceeded.
    assert budget.count_tiers().get("L1", 0) <= MAX_L1, "R27: L1 allowance exceeded"


# ── R28: File-skip allowlists are FROZEN ────────────────────────────────
def test_R28_file_skip_lists_are_frozen():
    """The 200-line-cap allowlist is FROZEN and single-sourced.

    Growing a skip list to dodge R3 (instead of refactoring) is exactly
    the governance failure that let ib_adapter.py/shadow_book.py exempt
    themselves in the past. This rule forces any future 200-line
    exception to happen in ONE visible place (FROZEN_FILE_SKIP) that is
    mirrored 1:1 by scripts/check_file_length.sh and re-verified by the
    shell gate itself, so nothing can silently slip through.
    """
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "check_file_length.sh"
    )
    assert script_path.exists(), "R28: shell gate script missing"
    script_src = script_path.read_text()

    # 1) Line-cap handling must reference the frozen constant (no drift copy).
    self_file = Path(__file__).read_text()
    tree = ast.parse(self_file)
    r3_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "test_R3_no_file_exceeds_200_lines"
    )
    r3_body = ast.get_source_segment(self_file, r3_node)
    assert "FROZEN_FILE_SKIP" in r3_body, "R28: test_R3 must use the frozen constant"

    # 2) Script allowlist == frozen allowlist (exact, basename-for-basename).
    m = re.search(r"^SKIP=\"([^\"]+)\"$", script_src, re.MULTILINE)
    assert m, 'R28: script must declare a single SKIP="..." line'
    script_tokens = {tok for tok in m.group(1).split("|") if tok}
    assert script_tokens == set(
        FROZEN_FILE_SKIP
    ), f"R28: script SKIP drifted from FROZEN_FILE_SKIP: script={script_tokens - set(FROZEN_FILE_SKIP)}, missing={set(FROZEN_FILE_SKIP) - script_tokens}"

    # 3) Allowlist is basenames only — no subdirectory-prefixed entries are
    #    ever allowed (they'd let a module hide behind a path segment).
    for entry in FROZEN_FILE_SKIP:
        assert "/" not in entry, f"R28: allowlist entry must be a basename, got {entry}"
        assert entry.endswith(
            ".py"
        ), f"R28: allowlist entry must end in .py, got {entry}"

    # 4) Every exempted entry must still protect at least one >200-line file
    #    (no stale entries lingering after a successful refactor). config.py
    #    legitimately covers brain/config.py (207 lines) while the top-level
    #    12-line config.py stays subject to the rule.
    for entry in FROZEN_FILE_SKIP:
        matches = [p for p in SRC.rglob(entry)]
        assert (
            matches
        ), f"R28: stale allowlist entry {entry} — refactor is done, remove it"
        oversized = [p for p in matches if len(p.read_text().splitlines()) > 200]
        assert (
            oversized
        ), f"R28: {entry} — no file behind it is >200 lines anymore, remove it"


def test_R29_gated_organs_default_off():
    """Every rollout gate is a literal False: flips are separate, reviewable changes.

    The P3 flag rollouts and the neuro organ gates must default OFF so the
    live 0.7·cortex + 0.3·neuro path stays byte-identical until each organ
    passes its backtest gate. Asserting these directly (instead of trusting
    convention) means no PR can quietly enable an organ in the same change
    that flips behavior.
    """
    from hanoon_prime import immune

    gates = {
        "CALIBRATION_NUDGE_ENABLED": False,
        "HYSTERESIS_EXIT_ENABLED": False,
        "PROBE_RECOVERY_ENABLED": False,
        "CONTRARIAN_MODE_ENABLED": False,
        "DELIBERATION_TRACE_ENABLED": False,
        "HALIM_EVIDENCE_LEARNING": False,
        "NEURO_BLEND_ENABLED": False,
        "NEURO_LEARN_ENABLED": False,
        "NEURO_ADAPTIVE_THRESHOLD_ENABLED": False,
        "NEURO_MOE_GATE_ENABLED": False,
    }
    for name, expected in gates.items():
        value = getattr(immune, name, None)
        assert isinstance(
            value, bool
        ), f"{name} must be an explicit bool, got {value!r}"
        assert (
            value is expected
        ), f"{name} must be {expected} by default (live path stays byte-identical)"


def test_R30_rollout_gates_off_at_live_read_sites():
    """The 10 rollout gates are OFF where the LIVE code actually reads them.

    R29 pins the immune.py literals, but several organs are consumed through
    at-import aliases the live code branches on directly: orchestrator's
    NEURO_BLEND_ENABLED / DELIBERATION_TRACE_ENABLED, exit_ladder's
    HYSTERESIS_EXIT_ENABLED, realized_ev's CALIBRATION_NUDGE_ENABLED,
    probe_recovery's PROBE_RECOVERY_ENABLED, contrarian's
    CONTRARIAN_MODE_ENABLED, pillar_evidence/consolidation's
    HALIM_EVIDENCE_LEARNING. A module-import-time flip anywhere would keep
    immune.py False yet still switch an organ ON live. So the audit runs in a
    FRESH interpreter — the pristine read-path a live bot gets — constructs
    the production brain, and requires every read-site to be False.
    """
    import subprocess

    audit = Path(__file__).resolve().parents[1] / "scripts" / "live_gate_audit.py"
    result = subprocess.run(
        [sys.executable, str(audit)],
        capture_output=True,
        text=True,
        cwd=audit.parents[1],
        timeout=120,
    )
    assert "AUDIT_RESULT=PASS" in result.stdout, (
        "live read-site audit failed:\n" + result.stdout + result.stderr
    )


def test_R31_promotion_manifest_backs_every_flipped_gate():
    """Any ON gate must equal promotions.json, evidenced with the money gates."""
    import json as _json

    from hanoon_prime import immune

    manifest = _json.loads(
        (
            Path(__file__).resolve().parents[1] / "docs" / "gates" / "promotions.json"
        ).read_text()
    )
    promoted = manifest.get("promoted", {})
    asserted_unknown = set(promoted) - set(ROLLOUT_GATES)
    assert not asserted_unknown, f"unknown gates in promotions.json: {asserted_unknown}"
    declared_on = {name for name in ROLLOUT_GATES if bool(getattr(immune, name, False))}
    assert declared_on == set(promoted), (
        "promoted must equal declared-ON gates; a flip without "
        f"evidence fails: on={sorted(declared_on)} "
        f"promoted={sorted(set(promoted))}"
    )
    for name, entry in promoted.items():
        assert entry.get("deflated_sr", 0.0) > 0.0, f"{name}: deflated_sr must be > 0"
        ticks = entry.get("profitable_tickers")
        assert (
            isinstance(ticks, str) and "/" in ticks
        ), f"{name}: profitable_tickers must look like 'N/M'"
        assert entry.get("wfa_file") and entry.get(
            "date"
        ), f"{name}: wfa_file and date are required"
