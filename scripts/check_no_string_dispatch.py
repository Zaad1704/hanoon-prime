#!/usr/bin/env python3
"""scripts/check_no_string_dispatch.py — Enforce R13: no string-based verdict dispatch.

Direction must come from enum-like integers (Thought.direction ∈ {-1, 0, +1}),
never from string comparison. Prevents typos like:
  if thought.direction == "LON"  ← would silently never trigger
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
VERDICT_STRINGS = {
    "BUY",
    "SELL",
    "HOLD",
    "ENTER",
    "EXIT",
    "LONG",
    "SHORT",
    "long",
    "short",
}


def main() -> int:
    violations: list[str] = []
    excluded = {
        "ib_adapter.py",
        "ib_streamer.py",
        "ib_executor.py",
        "ib_compat.py",
        "cortex.py",
    }
    for pyfile in sorted(SRC.rglob("*.py")):
        if pyfile.name in excluded:
            continue
        content = pyfile.read_text()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comp in node.comparators:
                    if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                        if comp.value in VERDICT_STRINGS:
                            violations.append(
                                f"  {pyfile.name}:{node.lineno}"
                                f" — string comparison against '{comp.value}'"
                            )
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if isinstance(kw.value, ast.Constant) and isinstance(
                        kw.value.value, str
                    ):
                        if kw.value.value in VERDICT_STRINGS and kw.arg in (
                            "direction",
                            "verdict",
                            "signal",
                        ):
                            violations.append(
                                f"  {pyfile.name}:{node.lineno}"
                                f" — string passed as {kw.arg}='{kw.value.value}'"
                            )
    if violations:
        print("R13 VIOLATION — string-based verdict dispatch detected:")
        for v in violations:
            print(v)
        return 1
    print("R13 OK — no string-based verdict dispatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
