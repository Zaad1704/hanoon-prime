#!/usr/bin/env python3
"""scripts/check_verdict_strings.py — Enforce R1: no verdict strings in signal modules.

The string literals "BUY", "SELL", "HOLD", "ENTER", "EXIT" must
ONLY appear in cortex.py. Signal modules (cerebellum, edge, hands,
hippocampus, immune) must never produce verdicts — they return
Thought objects, not strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "hanoon_prime"
VERDICT_STRINGS = (
    '"BUY"',
    '"SELL"',
    '"HOLD"',
    '"ENTER"',
    '"EXIT"',
    "'BUY'",
    "'SELL'",
    "'HOLD'",
    "'ENTER'",
    "'EXIT'",
)

# These modules are allowed to produce verdict strings:
ALLOWED = {"cortex.py"}
# These modules must NEVER contain verdict strings:
FORBIDDEN = {
    "cerebellum.py",
    "edge.py",
    "hands.py",
    "hippocampus.py",
    "immune.py",
    "eyes.py",
    "memory.py",
    "metrics.py",
    "types.py",
    "calibrate.py",
    "validator.py",
    "brain.py",
}


def main() -> int:
    violations: list[str] = []
    for mod in FORBIDDEN:
        path = SRC / mod
        if not path.exists():
            continue
        content = path.read_text()
        for verdict in VERDICT_STRINGS:
            if verdict in content:
                violations.append(f"  {mod} contains {verdict}")
    if violations:
        print("R1 VIOLATION — verdict strings in non-cortex modules:")
        for v in violations:
            print(v)
        return 1
    print("R1 OK — no verdict strings outside cortex.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
