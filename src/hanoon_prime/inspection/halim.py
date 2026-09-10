"""halim joint check."""

from __future__ import annotations

from .checks import FAIL, OK, CheckResult
from .ctx import InspectionContext
from .probe import halim_probe


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
