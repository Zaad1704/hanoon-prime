#!/usr/bin/env python3
"""scripts/check_no_todo.py — Enforce R16: no TODO/FIXME/HACK/XXX markers.

These markers indicate unfinished work. In a production system
they represent known liabilities. If it needs doing, do it.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
MARKERS = ("TODO", "FIXME", "HACK", "XXX", "NOQA")


def main() -> int:
    violations: list[str] = []
    for pyfile in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(pyfile.read_text().splitlines(), 1):
            for marker in MARKERS:
                if marker in line.upper():
                    violations.append(f"  {pyfile.name}:{i} — {marker}")
    if violations:
        print("R16 VIOLATION — TODO/FIXME markers found in src/:")
        for v in violations:
            print(v)
        return 1
    print("R16 OK — no TODO/FIXME markers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
