#!/usr/bin/env python3
"""scripts/check_positive_weights.py — Enforce R9: indicator weights must all be positive.

Negative weights on FAST tickers produce SELL signals on uptrending
tickers, causing guaranteed losses. All 5 indicator weights must be
positive (or zero) — sign is determined by the indicator's own
edge, not by weight negation.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "hanoon_prime"


def main() -> int:
    try:
        from hanoon_prime.immune import INDICATOR_WEIGHTS
    except ImportError as e:
        print(f"R9 FAIL — cannot import: {e}")
        return 1

    violations: list[str] = []
    for name, weight in INDICATOR_WEIGHTS.items():
        if weight < 0:
            violations.append(f"  {name}: weight={weight} (must be ≥ 0)")

    if violations:
        print("R9 VIOLATION — negative indicator weights found:")
        for v in violations:
            print(v)
        return 1
    print("R9 OK — all indicator weights are positive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
