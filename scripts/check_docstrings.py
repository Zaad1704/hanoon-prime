#!/usr/bin/env python3
"""scripts/check_docstrings.py — Enforce R11: all public functions have docstrings.

Every function not starting with _ must have a docstring. This
prevents the accumulation of undocumented code paths.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def main() -> int:
    violations: list[str] = []
    for pyfile in sorted(SRC.rglob("*.py")):
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                if node.name.startswith("test_"):
                    continue
                if not (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    violations.append(
                        f"  {pyfile.name}:{node.lineno} {node.name}"
                        " — missing docstring"
                    )
    if violations:
        print("R11 VIOLATION — public functions missing docstrings:")
        for v in violations:
            print(v)
        return 1
    print("R11 OK — all public functions have docstrings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
