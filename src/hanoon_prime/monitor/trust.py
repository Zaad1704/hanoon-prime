"""monitor.trust — weighted composite trust score for at-a-glance health."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

HEALTHY_FLOOR: float = 80.0
DEGRADED_FLOOR: float = 50.0

# Pipeline check name → relative weight (max total ~sum below).
_WEIGHTS: dict[str, int] = {
    "ib_connected": 5,
    "bars_fresh": 4,
    "brain_advancing": 4,
    "entry_evals": 3,
    "subs_present": 2,
}


@dataclass(frozen=True)
class TrustCheck:
    """One named, weighted, annotated health signal."""

    name: str
    ok: bool
    weight: int = 1
    detail: str = ""


def status_for(score: float) -> str:
    """Map a 0-100 composite score to a status label."""
    if score >= HEALTHY_FLOOR:
        return "HEALTHY"
    if score >= DEGRADED_FLOOR:
        return "DEGRADED"
    return "CRITICAL"


def checks_from_snapshot(snap: dict[str, Any]) -> list[TrustCheck]:
    """Build the weighted check list from a PipelineMonitor snapshot."""
    failing = snap.get("failing", {})
    return [
        TrustCheck(
            name=f"{name}",
            ok=name not in failing,
            weight=int(_WEIGHTS.get(name, 1)),
            detail=str(failing.get(name) or "ok"),
        )
        for name in _WEIGHTS
    ]


def score_checks(checks: list[TrustCheck]) -> dict[str, Any]:
    """Weighted composite score + failing list from a check list."""
    total = sum(c.weight for c in checks) or 1
    got = sum(c.weight for c in checks if c.ok)
    score = round(100.0 * got / total)
    return {
        "score": score,
        "status": status_for(score),
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
        "failing": [c.name for c in checks if not c.ok],
        "last_updated": time.time(),
    }


def assess(snap: dict[str, Any]) -> dict[str, Any]:
    """Weighted composite trust from a pipeline snapshot (telemetry /trust)."""
    return score_checks(checks_from_snapshot(snap))
