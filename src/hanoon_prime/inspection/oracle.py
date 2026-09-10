"""execution oracle joint checks — verdict->fill reconciliation."""

from __future__ import annotations

from typing import Any

from .checks import OK, UNVERIFIABLE, WARN, CheckResult
from .ctx import InspectionContext
from .probe import health, journal_tail, runtime_state

ENTER_ACTIONS = {"ENTER"}


def _events(ctx: InspectionContext, event: str) -> list[dict[str, Any]]:
    """Rows in the journal tail matching the requested event type."""
    return [r for r in journal_tail(ctx, 200) if r.get("event") == event]


def enters_minted(ctx: InspectionContext) -> CheckResult:
    """Every ENTER verdict is either filling an open IB position or closed."""
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult(
            "execution_oracle",
            "enters_minted",
            UNVERIFIABLE,
            detail=str(h["_unreachable"]),
        )
    positions = set(h.get("positions") or [])
    enters = [r for r in _events(ctx, "verdict") if r.get("action") in ENTER_ACTIONS]
    closed = {r.get("ticker") for r in _events(ctx, "position_closed")}
    unmatched = [
        r.get("ticker")
        for r in enters
        if r.get("ticker") not in positions and r.get("ticker") not in closed
    ]
    if not unmatched:
        return CheckResult(
            "execution_oracle",
            "enters_minted",
            OK,
            evidence={"enters": len(enters), "unmatched": 0},
        )
    return CheckResult(
        "execution_oracle",
        "enters_minted",
        WARN,
        detail=f"{len(unmatched)} ENTER without fill or IB position",
        evidence={"unmatched": unmatched[:8]},
    )


def closes_reconciled(ctx: InspectionContext) -> CheckResult:
    """Position closes carry the pnl/entry_price/shares trio."""
    closed = _events(ctx, "position_closed")
    missing = sum(
        1 for r in closed if not all(k in r for k in ("pnl", "entry_price", "shares"))
    )
    if not missing:
        return CheckResult(
            "execution_oracle",
            "closes_reconciled",
            OK,
            evidence={"closes": len(closed)},
        )
    return CheckResult(
        "execution_oracle",
        "closes_reconciled",
        WARN,
        detail=f"{missing} close(s) missing pnl/entry_price/shares",
    )


def equity_synced(ctx: InspectionContext) -> CheckResult:
    """Policy equity is synced and non-zero."""
    pol = runtime_state(ctx).get("brain_state", {}).get("policy_state", {})
    if not isinstance(pol, dict):
        return CheckResult(
            "execution_oracle",
            "equity_synced",
            UNVERIFIABLE,
            detail="policy_state missing",
        )
    synced = bool(pol.get("equity_synced", False))
    equity = float(pol.get("equity", 0.0) or 0.0)
    if synced and equity > 0:
        return CheckResult(
            "execution_oracle", "equity_synced", OK, detail=f"equity {equity:.2f}"
        )
    if not synced:
        return CheckResult(
            "execution_oracle",
            "equity_synced",
            WARN,
            detail="post-restart equity not yet synced",
        )
    return CheckResult(
        "execution_oracle", "equity_synced", WARN, detail="synced but equity zero"
    )
