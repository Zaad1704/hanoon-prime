"""safety joint checks."""

from __future__ import annotations

import re

from .checks import FAIL, OK, UNVERIFIABLE, WARN, CheckResult
from .ctx import InspectionContext
from .probe import (
    GUARD_MARKER,
    LEARN_BLOCKED_MARKER,
    SAFETY_HALT_MARKER,
    TRACE_MARKER,
    health,
    runtime_state,
    session_lines,
)

ERROR_BURST_WARN = 10
ERROR_BURST_FAIL = 50
ERROR_MARKER = re.compile(r" (ERROR|CRITICAL) ")


def _counts(ctx: InspectionContext) -> dict[str, int]:
    """Count marker hits in the current session's log lines."""
    lines = session_lines(ctx)
    return {
        "guard": sum(1 for ln in lines if GUARD_MARKER.search(ln)),
        "tracebacks": sum(1 for ln in lines if TRACE_MARKER.search(ln)),
        "safety_halt": sum(1 for ln in lines if SAFETY_HALT_MARKER.search(ln)),
        "learn_blocked": sum(1 for ln in lines if LEARN_BLOCKED_MARKER.search(ln)),
    }


def _zero_check(
    ctx: InspectionContext, name: str, value: int, label: str
) -> CheckResult:
    """Build a FAIL/OK result from a count of blocked markers."""
    if value:
        return CheckResult(
            "safety", name, FAIL, detail=f"{value} {label}(s) since last start"
        )
    return CheckResult("safety", name, OK, detail="none")


def no_netting_guard(ctx: InspectionContext) -> CheckResult:
    """No NETTING GUARD blocks fired since last start."""
    return _zero_check(ctx, "no_netting_guard", _counts(ctx)["guard"], "NETTING GUARD")


def no_traceback(ctx: InspectionContext) -> CheckResult:
    """No Python tracebacks since last start."""
    return _zero_check(ctx, "no_traceback", _counts(ctx)["tracebacks"], "Traceback")


def no_safety_halt(ctx: InspectionContext) -> CheckResult:
    """No SAFETY HALT assertions since last start."""
    return _zero_check(
        ctx, "no_safety_halt", _counts(ctx)["safety_halt"], "SAFETY HALT"
    )


def no_learn_blocked(ctx: InspectionContext) -> CheckResult:
    """No juli LEARN BLOCKED stalls since last start."""
    return _zero_check(
        ctx, "no_learn_blocked", _counts(ctx)["learn_blocked"], "LEARN BLOCKED"
    )


def policy_flags(ctx: InspectionContext) -> CheckResult:
    """Safety net enabled and bot authorized while a session is active."""
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult(
            "safety", "policy_flags", UNVERIFIABLE, detail=str(h["_unreachable"])
        )
    pol = runtime_state(ctx).get("brain_state", {}).get("policy_state", {})
    if not isinstance(pol, dict):
        return CheckResult(
            "safety", "policy_flags", UNVERIFIABLE, detail="policy_state missing"
        )
    enabled = bool(pol.get("enabled", False))
    authorized = bool(pol.get("authorized", True))
    if not h.get("session_active"):
        return CheckResult(
            "safety", "policy_flags", OK, detail="session inactive — idle expected"
        )
    if not enabled:
        return CheckResult(
            "safety",
            "policy_flags",
            WARN,
            detail="safety net disabled while session active",
        )
    if not authorized:
        return CheckResult("safety", "policy_flags", WARN, detail="bot not authorized")
    return CheckResult("safety", "policy_flags", OK)


def no_error_burst(ctx: InspectionContext) -> CheckResult:
    """No ERROR/CRITICAL log flood since the last start."""
    errors = [ln for ln in session_lines(ctx) if ERROR_MARKER.search(ln)]
    if len(errors) >= ERROR_BURST_FAIL:
        return CheckResult(
            "safety",
            "no_error_burst",
            FAIL,
            detail=f"{len(errors)} error lines since last start",
        )
    if len(errors) >= ERROR_BURST_WARN:
        return CheckResult(
            "safety",
            "no_error_burst",
            WARN,
            detail=f"{len(errors)} error lines since last start",
        )
    return CheckResult("safety", "no_error_burst", OK, detail="none")


def drawdown_bound(ctx: InspectionContext) -> CheckResult:
    """Daily PnL stays within the -1% drawdown floor."""
    pol = runtime_state(ctx).get("brain_state", {}).get("policy_state", {})
    if not isinstance(pol, dict):
        return CheckResult(
            "safety", "drawdown_bound", UNVERIFIABLE, detail="policy_state missing"
        )
    equity = float(pol.get("equity", 0.0) or 0.0)
    if equity <= 0:
        return CheckResult(
            "safety",
            "drawdown_bound",
            OK,
            detail="equity 0 — rule armed after first equity sync",
        )
    dpnl = float(pol.get("daily_pnl", 0.0) or 0.0)
    pct = -100.0 * dpnl / equity
    ok = pct <= 1.0
    detail = f"daily_pnl {dpnl:.2f}" + (
        f" = -{pct:.2f}% (floor -1.0%)" if not ok else f" within floor (-{pct:.2f}%)"
    )
    return CheckResult("safety", "drawdown_bound", OK if ok else FAIL, detail=detail)
