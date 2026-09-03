#!/usr/bin/env python3
"""scripts/check_no_swallow.py — Enforce R15: no swallowed exceptions.

Bare `except:` or `except: pass` silently hides bugs. In a trading
system, this can mean a position is never closed because an
exception was swallowed. Every except must either log or re-raise.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def main() -> int:
    violations: list[str] = []
    excluded = {
        "ib_adapter.py",
        "ib_streamer.py",
        "ib_executor.py",
        "ib_compat.py",
        "_ib_sync.py",
    }
    for pyfile in sorted(SRC.rglob("*.py")):
        if pyfile.name in excluded:
            continue
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    violations.append(
                        f"  {pyfile.name}:{node.lineno}"
                        " — bare except (must specify exception type)"
                    )
                    continue
                # Check for pass-only handler body
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    violations.append(
                        f"  {pyfile.name}:{node.lineno}"
                        " — except: pass (must log or re-raise)"
                    )
    if violations:
        print("R15 VIOLATION — swallowed exceptions detected:")
        for v in violations:
            print(v)
        return 1
    print("R15 OK — no swallowed exceptions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
