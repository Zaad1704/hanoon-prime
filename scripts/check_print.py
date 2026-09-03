#!/usr/bin/env python3
"""scripts/check_print.py — Enforce R10: no print() in production source code.

Only scripts/ and tests/ may use print(). The src/ tree must use
logging or return values. This prevents debug noise from leaking
into production and forces structured output.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def main() -> int:
    violations: list[str] = []
    excluded = {"ib_adapter.py", "ib_streamer.py", "ib_executor.py", "ib_compat.py"}
    for pyfile in sorted(SRC.rglob("*.py")):
        if pyfile.name in excluded:
            continue
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Name) and func.id == "print") or (
                    isinstance(func, ast.Attribute) and func.attr == "print"
                ):
                    violations.append(
                        f"  {pyfile}: line {node.lineno} — print() in source"
                    )
    if violations:
        print("R10 VIOLATION — print() calls in src/ (use logging instead):")
        for v in violations:
            print(v)
        return 1
    print("R10 OK — no print() in src/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
