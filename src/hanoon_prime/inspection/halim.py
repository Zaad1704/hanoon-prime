"""halim joint checks — liveness + decision integration."""

from __future__ import annotations

from .checks import FAIL, OK, WARN, CheckResult
from .ctx import InspectionContext
from .probe import halim_probe
from .probe import snapshot as snapshot_probe


def halim_state_matches_clock(ctx: InspectionContext) -> CheckResult:
    """halim sessions are either up (or asleep) when they should be."""
    st = halim_probe(ctx)
    if st == "down":
        return CheckResult(
            "halim", "halim_state_matches_clock", FAIL, detail="down (degraded)"
        )
    suffix = " (expected post-market)" if st == "asleep" else ""
    return CheckResult(
        "halim", "halim_state_matches_clock", OK, detail=f"state={st}{suffix}"
    )


_JOINT = "halim"
_NAME = "halim_engaged_in_decisions"


def halim_engaged_in_decisions(ctx: InspectionContext) -> CheckResult:
    """HALIM modifier + recommendations flowing into brain state."""
    snap = snapshot_probe(ctx)
    mod = snap.get("halim_modifier")
    recs = snap.get("halim_recommendations", [])
    insight = snap.get("halim_last_insight")
    ev = {"modifier": mod, "recommendations": len(recs), "has_insight": bool(insight)}
    if mod is None:
        return CheckResult(
            _JOINT,
            _NAME,
            WARN,
            detail="not published yet (System 2 warming up)",
            evidence=ev,
        )
    n = float(mod or 0)
    if abs(n) > 0:
        return CheckResult(
            _JOINT,
            _NAME,
            OK,
            detail=f"mod={n:.3f}, recs={len(recs)}, insights={bool(insight)}",
            evidence=ev,
        )
    return CheckResult(
        _JOINT,
        _NAME,
        WARN,
        detail="polled but modifier is zero — no advisory signal",
        evidence=ev,
    )


__all__ = ["halim_state_matches_clock", "halim_engaged_in_decisions"]
