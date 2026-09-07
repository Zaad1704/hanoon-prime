"""tests/test_fixes_journal.py — FIXES.md is a live, enforced contract.

Enforces the rules documented in FIXES.md:
1. Every entry has a Class and a Guard; Class letters must be defined
   in the Bug Classes section (analyze-before-fixing rule).
2. Every `test:` guard names an existing test, and that test's file
   carries a `Regression: <FIX-ID>` marker pointing back at the entry.
3. Every `smoke:` guard names a check that exists in the smoke harness.
4. Class C static watchdog: `x.get(...) or []/ {}` patterns in src are
   banned unless annotated `# array-safe` (typed-container truthiness
   killed every live entry evaluation once).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "FIXES.md"
SRC = ROOT / "src" / "hanoon_prime"
TESTS = ROOT / "tests"
SMOKE = ROOT / "scripts" / "smoke_live.py"

FIX_ID = re.compile(r"FIX-\d{4}-\d{2}-\d{2}-\d{2}")
GUARD_TEST = re.compile(r"`test: ([^`]+)`")
GUARD_SMOKE = re.compile(r"`smoke: ([^`]+)`")
MARKER = re.compile(r"Regression: (FIX-\d{4}-\d{2}-\d{2}-\d{2})")
TRUTHY = re.compile(r"\.get\([^)]*\)\s+or\s+(\[\]|\{\})")


def entries(text: str) -> list[tuple[str, str]]:
    """Split FIXES.md into (id, body) entry pairs."""
    found: list[tuple[str, str]] = []
    for m in FIX_ID.finditer(text):
        start = m.start()
        nxt = FIX_ID.search(text, m.end())
        body_start = text.find("\n", start)
        body = text[body_start : nxt.start() if nxt else len(text)]
        found.append((m.group(0), body))
    return found


def defined_classes(text: str) -> set[str]:
    """Class letters defined in the Bug Classes section."""
    return set(re.findall(r"### Class ([A-Z])", text))


def test_every_entry_has_class_and_guard():
    """Each FIX entry classifies its bug and names a guard."""
    text = JOURNAL.read_text()
    ids = entries(text)
    assert len(ids) >= 6, "journal lost entries"
    classes = defined_classes(text)
    assert classes >= {"A", "B", "C", "D", "E"}, classes
    for fid, body in ids:
        cls = re.search(r"- \*\*Class:\*\* ([A-Z](?:, ?[A-Z])*)", body)
        assert cls, f"{fid}: missing **Class:** line (analyze-before-fixing)"
        for letter in re.findall(r"[A-Z]", cls.group(1)):
            assert letter in classes, f"{fid}: class {letter} undefined"
        assert "- **Guard:**" in body, f"{fid}: missing **Guard:** line"


def test_test_guards_exist_and_mark_back():
    """`test:` guards point at real tests marked Regression: <FIX-ID>."""
    text = JOURNAL.read_text()
    for fid, body in entries(text):
        for ref in GUARD_TEST.findall(body):
            path, _, name = ref.partition("::")
            f = ROOT / path.strip()
            assert f.exists(), f"{fid}: guard file missing: {path}"
            content = f.read_text()
            if name:
                assert name in content, f"{fid}: guard test missing: {name}"
                assert (
                    f"Regression: {fid}" in content
                ), f"{fid}: guard test {name} lacks the Regression marker"


def test_smoke_guards_name_real_checks():
    """`smoke:` guards name check strings present in the harness."""
    text = JOURNAL.read_text()
    smoke_src = SMOKE.read_text()
    for fid, body in entries(text):
        for ref in GUARD_SMOKE.findall(body):
            path, _, check = ref.partition("::")
            assert (ROOT / path.strip()).exists(), f"{fid}: {path} missing"
            assert (
                check.strip().strip('"') in smoke_src
            ), f"{fid}: smoke check not found: {check}"


def test_regression_markers_point_at_real_entries():
    """Every Regression marker in tests references a journal entry."""
    ids = {fid for fid, _ in entries(JOURNAL.read_text())}
    marked = []
    for f in TESTS.glob("*.py"):
        marked += [(f.name, m) for m in MARKER.findall(f.read_text())]
    assert marked, "no Regression markers found — journal went stale"
    for fname, fid in marked:
        assert fid in ids, f"{fname}: marker {fid} not in FIXES.md"


def test_class_c_watchdog_no_array_truthiness():
    """Static ban on `get(...) or []/{}` unless annotated array-safe."""
    offenders = []
    for f in SRC.rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if TRUTHY.search(line) and "# array-safe" not in line:
                offenders.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
    assert (
        not offenders
    ), "array-truthiness hazard (see FIXES.md Class C):\n" + "\n".join(offenders)
