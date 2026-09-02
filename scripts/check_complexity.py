#!/usr/bin/env python3
"""scripts/check_complexity.py — Enforce R3: no function > 40 lines, no nesting > 3."""
from __future__ import annotations
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

MAX_FUNC_LINES = 40
MAX_NESTING = 3


def check_nesting(node: ast.AST) -> int:
    depth = 0
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try,
                             ast.ExceptHandler, ast.AsyncFor, ast.AsyncWith)):
            depth = max(depth, 1 + check_nesting(child))
        else:
            depth = max(depth, check_nesting(child))
    return depth


def main() -> int:
    violations = []
    for pyfile in sorted(SRC.rglob("*.py")):
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = (node.end_lineno or node.lineno) - node.lineno + 1
                if length > MAX_FUNC_LINES:
                    violations.append(
                        f"  {pyfile.name}:{node.lineno} {node.name} = {length} lines (max {MAX_FUNC_LINES})"
                    )
                nesting = check_nesting(node)
                if nesting > MAX_NESTING:
                    violations.append(
                        f"  {pyfile.name}:{node.lineno} {node.name} nesting={nesting} (max {MAX_NESTING})"
                    )
    if violations:
        print("R3 VIOLATION — functions exceeding complexity limits:")
        for v in violations:
            print(v)
        return 1
    print("R3 OK — all functions within limits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
