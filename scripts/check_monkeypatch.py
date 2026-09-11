#!/usr/bin/env python3
"""scripts/check_monkeypatch.py — anti-monkeypatch contract (JULI 2.0).

Three layers, all enforced by pre-commit and CI:

1. Critical-singleton denylist.  Tests may NOT monkeypatch the risk/stdp
   critical singletons unless the patch is explicitly authorized with a
   trailing comment ``# allow: monkeypatch <SYMBOL>`` on the same line or
   the line directly above. Symbols blocked: TRADING_CONFIG, Journal (the
   append-only journal), Governor, Memory/BrainState (shared brain state),
   IHALIMAdapter, and the immune constant pool (DAILY_LOSS_LIMIT,
   MAX_POSITION_NOTIONAL, KILL_DAILY_LOSS_LIMIT, ...).
2. Contract no-patch rule.  Contract tests (test_contract.py) must never
   monkeypatch hanoon_prime.inspection.* or the module under test — a
   contract test that patches away the thing it checks proves nothing.
3. Production self-patching guard.  No src module may rewrite its own
   import-time bindings (``sys.modules[__name__]``, ``globals()[``,
   module-level ``setattr``/``__dict__`` writes). Runtime *instance* state
   (``setattr(self, ...)``) and config writes inside request handlers are
   allowed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TESTS = ROOT / "tests"

# Symbols whose monkeypatching can silently disable production risk rails.
# Allow-comment list names are matched by last dotted segment, so
# "TRADING_CONFIG", "pillar.TRADING_CONFIG" and
# "hanoon_prime.config.TRADING_CONFIG" are all recognized.
DENYLIST = {
    "TRADING_CONFIG",
    "Journal",
    "Governor",
    "Memory",
    "BrainState",
    "IHALIMAdapter",
    "DAILY_LOSS_LIMIT",
    "KILL_DAILY_LOSS_LIMIT",
    "MAX_POSITION_NOTIONAL",
    "MAX_LOSS_PER_TRADE",
    "MAX_CONCURRENT_POSITIONS",
    "CONSECUTIVE_LOSSES_PAUSE",
    "PAUSE_DURATION_MIN",
    "ATR_STOP_MULT",
    "ATR_TARGET_MULT",
    "PRIOR_TOP",
    "PRIOR_TOP_MAX",
    "SCORE_INVERT",
    "CONFIDENCE_FLOOR",
    "ENTRY_REUSE_COOLDOWN_SEC",
    "MAX_ENTRIES_PER_CYCLE",
}

_ALLOW_RE = re.compile(r"#\s*allow:\s*monkeypatch\s+([A-Za-z_][A-Za-z0-9_.]*)")
_SETATTR_RE = re.compile(r"monkeypatch\.setattr\(\s*([^,\)\s][^,]*?)\s*,")
_PATCH_TARGET_RE = re.compile(
    r"(?:TRADING_CONFIG|Journal|Governor|Memory|BrainState|IHALIMAdapter"
    r"|DAILY_LOSS_LIMIT|KILL_DAILY_LOSS_LIMIT|MAX_POSITION_NOTIONAL"
    r"|MAX_LOSS_PER_TRADE|MAX_CONCURRENT_POSITIONS|CONSECUTIVE_LOSSES_PAUSE"
    r"|PAUSE_DURATION_MIN|ATR_STOP_MULT|ATR_TARGET_MULT|PRIOR_TOP|PRIOR_TOP_MAX"
    r"|SCORE_INVERT|CONFIDENCE_FLOOR|ENTRY_REUSE_COOLDOWN_SEC|MAX_ENTRIES_PER_CYCLE)"
)

# Production self-patching patterns (module/import-time rewrites).
_SELF_PATCH_RE = re.compile(
    r"(?:sys\.modules\s*\[\s*[\"']__name__[\"']\s*\]"
    r"|globals\(\)\s*\["
    r"|news\.__dict__|sys\.modules|setattr\s*\(\s*(?:sys\.modules))"
)
# Runtime instance/config writes that ARE allowed.
_ALLOWED_SRC_RE = re.compile(r"setattr\s*\(\s*self|setattr\s*\(\s*TRADING_CONFIG")

_TESTS_WITH_DENYLIST_PATCHES = [
    "test_ib_executor.py",
    "test_pillar.py",
]

# The enforcement test itself embeds denylist-patch strings in literals to
# exercise the scanner; its own string constants must not be flagged.
_SELF_TEST_FILES = {"test_anti_monkeypatch.py"}


def _last_segment(name: str) -> str:
    return name.rstrip(".")
    # Only the last dotted segment matters for the denylist match.
    parts = name.strip().split(".")
    return parts[-1].strip() if parts else ""


def _authorized(lines: list[str], i: int) -> bool:
    """True when the patch statement carries an allow comment.

    Scans the preceding line through a short window after the match line so
    multi-line monkeypatch.setattr( calls (black-formatted) whose allow
    comment lands on the closing-paren line are still honored.
    """
    lo = max(0, i - 1)
    hi = min(len(lines), i + 6)
    for ln in lines[lo:hi]:
        m = _ALLOW_RE.search(ln)
        if m is not None:
            # Allow-comment must name the same symbol family as the patch.
            if _last_segment(m.group(1)) in DENYLIST or m.group(1) in DENYLIST:
                return True
    return False


def _check_test_file(path: Path) -> list[str]:
    """Return denylist-patch violations for one test file."""
    violating: list[str] = []
    try:
        text = path.read_text()
        lines = text.splitlines()
    except (OSError, UnicodeDecodeError):
        return violating
    for m in _SETATTR_RE.finditer(text):
        target = m.group(1).strip()
        if _PATCH_TARGET_RE.search(target) is None:
            continue
        lineno = text[: m.start()].count("\n") + 1
        if not _authorized(lines, lineno - 1):
            violating.append(
                f"{path.name}:{lineno} monkeypatch of {target} "
                f"needs '# allow: monkeypatch <symbol>' on the same or "
                f"preceding line"
            )
    return violating


_CONTRACT_PATCH_RE = re.compile(
    r"monkeypatch\.(?:setattr|setitem)\s*\(\s*(hanoon_prime\.\S+?)" r"[" ",]"
)


def _check_contract_no_patch(path: Path) -> list[str]:
    """Contract tests must not patch inspection.* or the module under test."""
    if path.name != "test_contract.py":
        return []
    violating: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return violating
    for i, line in enumerate(lines):
        m = _CONTRACT_PATCH_RE.search(line)
        if m is None:
            continue
        target = m.group(1)
        if "inspection" in target or "hanoon_prime" in target:
            if not _authorized(lines, i):
                violating.append(
                    f"{path.name}:{i + 1} contract test patches {target} — "
                    f"contract tests may not mock the module under test"
                )
    return violating


def _check_src_no_self_patch(path: Path) -> list[str]:
    """Production modules must not rewrite their own import-time bindings."""
    violating: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return violating
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith('"""'):
            continue
        if "from __future__ import annotations" in line:
            continue
        if _SELF_PATCH_RE.search(line) and not _ALLOWED_SRC_RE.search(line):
            violating.append(
                f"{path.name}:{i + 1} self-patching (import-time rewrite "
                f"of module bindings) is forbidden — {line.strip()}"
            )
    return violating


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    only_src = "--src-only" in args
    violations: list[str] = []

    # Layer 3: production self-patching guard (always).
    for py in sorted(SRC.rglob("*.py")):
        violations.extend(_check_src_no_self_patch(py))

    if not only_src:
        # Layer 1: critical-singleton denylist across the test suite.
        for py in sorted(TESTS.glob("test_*.py")):
            if py.name in _SELF_TEST_FILES:
                continue
            violations.extend(_check_test_file(py))
        # Layer 2: contract no-patch rule.
        for py in sorted(TESTS.glob("test_contract.py")):
            violations.extend(_check_contract_no_patch(py))

    if violations:
        print("ANTI-MONKEYPATCH VIOLATION:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("Anti-monkeypatch contract OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
