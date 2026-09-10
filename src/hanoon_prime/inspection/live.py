"""live fidelity joint checks — marks, holdings, and bar progression.

These checks reconcile the two surfaces that broke together: the PnL the
bot reports for its positions and the closed-bar feed its entry gate needs.
Coarse liveness (CYCLE lines, connected flag) passes while both are wrong,
so each check compares a number against IB ground truth instead.
"""

from __future__ import annotations

from typing import Any

from .checks import FAIL, OK, UNVERIFIABLE, WARN, CheckResult
from .ctx import InspectionContext
from .probe import account, health, positions, recent_cycle_bars


def _unreachable(ctx: InspectionContext, joint: str, name: str) -> CheckResult | None:
    """Flag a probe instrument failure as UNVERIFIABLE."""
    if health(ctx).get("_unreachable"):
        return CheckResult(joint, name, UNVERIFIABLE, detail="health unreachable")
    return None


def _mark_verdict(
    name: str, listed: list[dict[str, Any]], total: float, ctx: InspectionContext
) -> CheckResult:
    """Decide FAIL/WARN/OK by cross-checking the positions surface vs IB."""
    summary = account(ctx).get("account_summary")
    if not isinstance(summary, dict):
        summary = {}
    account_pnl = float(
        summary.get("unrealized_pnl") or summary.get("UnrealizedPnL") or 0.0
    )
    if total == 0.0 and abs(account_pnl) > 0.5:
        return CheckResult(
            "telemetry",
            name,
            FAIL,
            detail=f"surface PnL 0.00 vs IB {account_pnl:.2f}",
            evidence={"total": total, "ib": account_pnl, "count": len(listed)},
        )
    stale = [
        p["ticker"] for p in listed if p.get("market_price") == p.get("entry_price")
    ]
    if stale:
        all_stuck = len(stale) == len(listed)
        detail = (
            f"all {len(listed)} mark(s) stuck at entry"
            if all_stuck
            else f"{len(stale)} mark(s) still at entry: {','.join(stale)}"
        )
        return CheckResult(
            "telemetry",
            name,
            FAIL if all_stuck else WARN,
            detail=detail,
        )
    return CheckResult(
        "telemetry",
        name,
        OK,
        detail=f"total {total:.2f} across {len(listed)} positions",
    )


def positions_marked_live(ctx: InspectionContext) -> CheckResult:
    """Open positions must carry a live mark, not a zero-PnL lie."""
    bad = _unreachable(ctx, "telemetry", "positions_marked_live")
    if bad:
        return bad
    shown = positions(ctx)
    if shown.get("_unreachable"):
        return CheckResult(
            "telemetry",
            "positions_marked_live",
            UNVERIFIABLE,
            detail="positions unreachable",
        )
    listed = shown.get("positions")
    if not isinstance(listed, list):
        listed = []
    if not listed:
        held = int(health(ctx).get("position_count", 0) or 0)
        return CheckResult(
            "telemetry",
            "positions_marked_live",
            WARN if held else OK,
            detail="no marks listed" if held else "flat",
        )
    return _mark_verdict(
        "positions_marked_live", listed, float(shown.get("total_pnl") or 0.0), ctx
    )


def bars_advance_when_active(ctx: InspectionContext) -> CheckResult:
    """An active session must be closing bars, not a stuck bars=0 feed."""
    bad = _unreachable(ctx, "pipeline", "bars_advance_when_active")
    if bad:
        return bad
    if not health(ctx).get("session_active"):
        return CheckResult("pipeline", "bars_advance_when_active", OK, detail="idle")
    counts = recent_cycle_bars(ctx)
    if not counts:
        return CheckResult(
            "pipeline", "bars_advance_when_active", FAIL, detail="no CYCLE lines"
        )
    # Cycle runs every ~1s; 1-minute bars close at the minute boundary.
    # Only tickers in pendingTickers() that cross a boundary in this cycle
    # are counted, so the expected rate is ~1.7% (1/60). Thresholds are
    # calibrated to catch a genuinely stuck feed (0% closure) while
    # tolerating the natural burst pattern of minute-bar aggregation.
    closed = sum(1 for n in counts if n > 0)
    rate = closed / len(counts) if counts else 0.0
    if rate < 0.01:
        status = FAIL
    elif rate < 0.05:
        status = WARN
    else:
        status = OK
    return CheckResult(
        "pipeline",
        "bars_advance_when_active",
        status,
        detail=f"{closed}/{len(counts)} cycles closed a bar",
        evidence={"bars": counts[-10:], "window": len(counts)},
    )
