"""hanoon_prime.brain.telemetry_summaries — compact dashboard snapshot builders.

Aggregation helpers that stay small enough to keep the source modules
inside the R3 file-length budget. Feed read-only dashboard panels — no
decision behavior lives here.
"""

from __future__ import annotations

from typing import Any


def extinction_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts plus strongest-active cells for the live dashboard.

    Ships aggregate counts and only the top cells (by inhibition, then
    pattern mass) so telemetry never streams the whole signature map.
    """
    return {
        "size": len(cells),
        "contexts": len({c.get("context", "") for c in cells}),
        "inhibited": sum(1 for c in cells if (c.get("inhibition") or 0) > 0),
        "total_patterns": sum(int(c.get("patterns", 0)) for c in cells),
        "cells": sorted(
            cells,
            key=lambda c: (c.get("inhibition", 0.0), c.get("patterns", 0)),
            reverse=True,
        )[:16],
    }


__all__ = ["extinction_summary"]
