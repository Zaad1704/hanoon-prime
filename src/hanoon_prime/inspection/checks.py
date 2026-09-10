"""Check model + runner harness for the Inside Man.

Every joint is verified through CheckSpec functions returning CheckResult.
The harness wraps each fn so an instrument failure is reported as
UNVERIFIABLE — never a raise, and never a false FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AbstractSet, Callable

if TYPE_CHECKING:
    from .ctx import InspectionContext

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
UNVERIFIABLE = "UNVERIFIABLE"

CheckFn = Callable[["InspectionContext"], "CheckResult"]

VALID_ACTIONS = {"BUY", "SELL", "HOLD", "VETOED", "PASS", "OPEN", "CLOSE", "ENTER"}


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check. status is OK|WARN|FAIL|UNVERIFIABLE."""

    joint: str
    name: str
    status: str
    detail: str = ""
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckSpec:
    """Declarative registration of a single check against a joint."""

    joint: str
    name: str
    fn: CheckFn
    hard: bool = False  # FAIL => guardian hard violation (exit 2, streak reset)
    report: bool = False  # non-hard finding => bug-catcher anomaly feed (exit 4)


def run_check(spec: CheckSpec, ctx: "InspectionContext") -> CheckResult:
    """Run one check; convert any instrument failure to UNVERIFIABLE."""
    try:
        return spec.fn(ctx)
    except Exception as exc:  # instrument failure ≠ verdict
        return CheckResult(
            spec.joint,
            spec.name,
            UNVERIFIABLE,
            detail=f"instrument error: {exc}",
            evidence={"error": str(exc)},
        )


def MANIFEST_STATUS(
    results: "list[CheckResult]", hard_keys: "AbstractSet[tuple[str, str]]"
) -> str:
    """Any hard FAIL => FAIL; else any WARN/UNVERIFIABLE => WARN; else OK."""
    hard_fails = [
        r for r in results if (r.joint, r.name) in hard_keys and r.status == FAIL
    ]
    if hard_fails:
        return FAIL
    if any(r.status in (WARN, UNVERIFIABLE) for r in results):
        return WARN
    return OK
