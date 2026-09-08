"""tests/test_architecture.py — Brain-First architecture enforcement.

Doctrine: IB → JULI (the whole brain — all processing/decisions) → execution
→ learning. Every decision organ lives inside ``hanoon_prime/brain/``; no new
top-level organs; the fast/slow cortices are brain-owned; orchestration layers
(juli, ib_cycle, telemetry) never carry decision vocabulary or own decision
engines. Violations here are standing failures — fixing them is the absorption
contract, and these assertions stay binding after the refactor lands.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

# Decision organs that only the brain may own/instantiate. Value types
# (Verdict/ENTER/HOLD/VETOED) are data and may travel freely.
DECISION_ENGINES = (
    "Governor",
    "PortfolioRiskManager",
    "SafetyProducer",
    "ProbeRecovery",
    "TradingPolicy",
)

# Names that orchestration may import from brain/policy — they are data or a
# read-only config surface, NOT decision engines.
ALLOWED_OUTSIDE_IMPORTS = frozenset(
    {"Verdict", "ENTER", "HOLD", "VETOED", "TradingConfig", "TRADING_CONFIG"}
)

# The canonical decision vocabulary. A gate reason/stage may ONLY appear (as a
# quoted string) inside hanoon_prime/brain/.
DECISION_REASONS = (
    "no_data",
    "eval_error",
    "no_signal",
    "session_disabled",
    "direction_rejected",
    "low_penny_score",
    "halted",
    "daily_loss_limit",
    "too_many_positions",
    "cycle_budget",
    "reuse_cooldown",
    "not_sized",
    "sized_to_zero",
    "equity_unsynced",
    "exposure_cap",
    "max_positions",
    "risk_scalar",
    "portfolio_giveback",
    "rotation_off",
)
DECISION_STAGES = ("validity", "governor", "portfolio_risk", "trading_policy")
DECISION_VOCABULARY = tuple(set(DECISION_REASONS) | set(DECISION_STAGES))


def _source_files(home: str) -> list[Path]:
    """All .py files under SRC/home."""
    return sorted((SRC / "hanoon_prime" / home).rglob("*.py"))


def _quoted(token: str) -> re.Pattern[str]:
    """Match the token as a string literal in source."""
    return re.compile(rf'["\']{re.escape(token)}["\']')


def test_brain_modules_never_touch_ib():
    """Decision modules under brain/ must never import the IB layer.

    The IB connection surface lives exclusively outside brain/: ib_compat,
    ib_insync/ibapi, and the IB adapters are reachable only from the
    IB orchestration layer, never from where decisions are made.
    """
    ib_tokens = ("ib_insync", "ib_compat", "ibapi", "from .ib_compat")
    violations = []
    for pyfile in _source_files("brain"):
        content = pyfile.read_text()
        for token in ib_tokens:
            if token in content:
                violations.append(f"{pyfile.name}: contains {token!r}")
    assert (
        not violations
    ), "ARCH VIOLATION: brain/ must never touch the IB layer:\n" + "\n".join(violations)


def test_probe_recovery_is_brain_owned():
    """Probe recovery (a decision/policy organ) lives inside brain/ only."""
    brain_probe = SRC / "hanoon_prime" / "brain" / "probe_recovery.py"
    top_probe = SRC / "hanoon_prime" / "probe_recovery.py"
    assert brain_probe.exists(), "ARCH: brain/probe_recovery.py missing"
    assert not top_probe.exists(), "ARCH: probe_recovery.py must not live at top level"


def test_portfolio_risk_is_brain_owned():
    """Portfolio risk (a decision organ) lives inside brain/policy only."""
    brain_pr = SRC / "hanoon_prime" / "brain" / "policy" / "portfolio_risk.py"
    monitor_pr = SRC / "hanoon_prime" / "monitor" / "portfolio_risk.py"
    assert brain_pr.exists(), "ARCH: brain/policy/portfolio_risk.py missing"
    assert not monitor_pr.exists(), "ARCH: monitor/portfolio_risk.py must not exist"


def test_brain_policy_modules_not_leaked_outside_brain():
    """No module outside brain/ may import a brain policy decision engine.

    Orchestration may read brain VALUE types (Verdict, action constants) and
    the read-only config singleton — but must never import or own a decision
    engine (Governor, PortfolioRiskManager, SafetyProducer, TradingPolicy).
    """
    policy_dir = SRC / "hanoon_prime" / "brain" / "policy"
    if not policy_dir.exists():
        return  # not yet absorbed; becomes active the moment it exists
    outside = [
        p
        for p in (SRC / "hanoon_prime").rglob("*.py")
        if "brain" not in p.relative_to(SRC / "hanoon_prime").parts[:1]
    ]
    violations = []
    owned = _brain_defined_engines()
    for pyfile in outside:
        tree = ast.parse(pyfile.read_text(), str(pyfile))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if "brain.policy" not in module and "brain.probe_recovery" not in module:
                continue
            for alias in node.names:
                if alias.name in DECISION_ENGINES and alias.name in owned:
                    violations.append(
                        f"{pyfile.name}:{node.lineno} imports {alias.name}"
                    )
    assert (
        not violations
    ), "ARCH VIOLATION: brain policy engines leaked outside brain/:\n" + "\n".join(
        violations
    )


def test_brain_policy_engines_never_instantiated_outside_brain():
    """Decision engines may only be constructed by brain-owned modules.

    Ownership home is where the class is DEFINED: once a decision engine's
    class lives under brain/, no outside module may construct it. This turns
    red the moment an engine moves into brain/ (until its outside usages are
    absorbed) and stays binding forever after.
    """
    owned = _brain_defined_engines()
    if not owned:
        return
    outside = [
        p
        for p in (SRC / "hanoon_prime").rglob("*.py")
        if "brain" not in p.relative_to(SRC / "hanoon_prime").parts[:1]
    ]
    violations = []
    for pyfile in outside:
        tree = ast.parse(pyfile.read_text(), str(pyfile))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in owned:
                    violations.append(f"{pyfile.name}:{node.lineno} {node.func.id}()")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in owned:
                    violations.append(
                        f"{pyfile.name}:{node.lineno} {node.func.attr}() "
                        f"(method on {pyfile.name})"
                    )
    assert (
        not violations
    ), "ARCH VIOLATION: decision engine instantiated outside brain/:\n" + "\n".join(
        violations
    )


def _brain_defined_engines() -> set[str]:
    """Decision engine names whose class is DEFINED under brain/."""
    owned: set[str] = set()
    for pyfile in (SRC / "hanoon_prime" / "brain").rglob("*.py"):
        tree = ast.parse(pyfile.read_text(), str(pyfile))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in DECISION_ENGINES:
                owned.add(node.name)
    return owned


def test_juli_is_decision_free():
    """juli.py is the JULI facade — it delegates, it never gate-keeps.

    Decision vocabulary (gate reasons/stages) is the brain's language; if a
    quoted decision token ever appears in juli.py a gate has leaked upward.
    """
    content = (SRC / "hanoon_prime" / "juli.py").read_text()
    hits = [t for t in DECISION_VOCABULARY if _quoted(t).search(content)]
    assert not hits, f"ARCH VIOLATION: juli.py carries decision vocabulary: {hits}"


def test_external_boundary_keeps_ib_out_of_decisions():
    """The brain decision pipeline stays executable IB-free.

    Decision modules must be importable without the IB adapter — a hard import
    of ib_compat inside brain/policy would break the architecture (decisions
    must never depend on order routing).
    """
    for pyfile in _source_files("brain/policy"):
        content = pyfile.read_text()
        tree = ast.parse(content, str(pyfile))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(
                    (node.lineno, ast.get_source_segment(content, node) or "")
                )
        for lineno, seg in imports:
            assert (
                "ib_compat" not in seg and "ib_insync" not in seg
            ), f"ARCH VIOLATION: {pyfile.name}:{lineno} imports IB layer"
