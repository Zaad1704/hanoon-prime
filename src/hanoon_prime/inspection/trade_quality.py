"""Trade-quality joint checks — P&L health, HALIM insights, sign consistency.

Reads recent ``position_closed`` journal events and the brain state's
``halim_last_insight`` to answer: *is the brain making money, and can
HALIM explain why (or why not)?*
"""

from __future__ import annotations

from typing import Any

from .checks import OK, WARN, CheckResult
from .ctx import InspectionContext
from .probe import runtime_state

CLOSED_EVENTS: tuple[str, ...] = ("position_closed", "exit")


def _recent_closed(ctx: InspectionContext, limit: int = 20) -> list[dict[str, Any]]:
    """Last *limit* closed-trade journal entries (newest first)."""
    from .probe_chain import journal_tail

    closed = [r for r in journal_tail(ctx, 500) if r.get("event") in CLOSED_EVENTS]
    closed.reverse()
    return closed[:limit]


def _pnl(t: dict[str, Any]) -> float:
    """Coerce journal P&L to float."""
    try:
        return float(t.get("pnl") or 0)
    except (TypeError, ValueError):
        return 0.0


def _gross(trades: list[dict[str, Any]]) -> tuple[float, float]:
    """Return (gross_wins, gross_losses) in absolute terms."""
    gw = sum(max(0, _pnl(t)) for t in trades)
    gl = abs(sum(min(0, _pnl(t)) for t in trades))
    return gw, gl


def win_rate(ctx: InspectionContext) -> CheckResult:
    """Recent win rate from closed trades. WARN < 50 %."""
    trades = _recent_closed(ctx, 20)
    if not trades:
        return CheckResult(
            "trade_quality",
            "win_rate",
            OK,
            detail="no closed trades — warming up",
            evidence={"trades": 0},
        )
    wins = sum(1 for t in trades if _pnl(t) > 0)
    rate = wins / len(trades)
    status = "WARN" if rate < 0.50 else OK
    return CheckResult(
        "trade_quality",
        "win_rate",
        status,
        detail=f"{wins}/{len(trades)} won ({rate:.0%})"
        + ("" if rate >= 0.50 else " — below 50 %"),
        evidence={"wins": wins, "total": len(trades), "win_rate": round(rate, 3)},
    )


def profit_factor(ctx: InspectionContext) -> CheckResult:
    """Profit factor (gross wins / gross losses). WARN if < 1.0."""
    trades = _recent_closed(ctx, 20)
    if not trades:
        return CheckResult(
            "trade_quality",
            "profit_factor",
            OK,
            detail="no closed trades — warming up",
            evidence={"trades": 0},
        )
    gw, gl = _gross(trades)
    if gl == 0:
        return CheckResult(
            "trade_quality",
            "profit_factor",
            OK,
            detail=f"all wins (gross {gw:.2f})",
            evidence={"gross_wins": round(gw, 2), "gross_losses": 0.0, "pf": "inf"},
        )
    pf = gw / gl
    status = OK if pf >= 1.0 else "WARN"
    return CheckResult(
        "trade_quality",
        "profit_factor",
        status,
        detail=f"pf={pf:.2f}" + (" — unprofitable tail" if pf < 1.0 else ""),
        evidence={
            "gross_wins": round(gw, 2),
            "gross_losses": round(gl, 2),
            "pf": round(pf, 3),
        },
    )


def halim_postmortem(ctx: InspectionContext) -> CheckResult:
    """HALIM post-trade insight present after a closed trade."""
    trades = _recent_closed(ctx, 5)
    if not trades:
        return CheckResult(
            "trade_quality",
            "halim_postmortem",
            OK,
            detail="no closed trades — N/A",
            evidence={"trades": 0},
        )
    state = runtime_state(ctx).get("brain_state", {})
    insight = state.get("halim_last_insight") if isinstance(state, dict) else None
    if isinstance(insight, dict) and insight.get("insight"):
        return CheckResult(
            "trade_quality",
            "halim_postmortem",
            OK,
            detail="HALIM provided trade analysis",
            evidence={"has_insight": True},
        )
    return CheckResult(
        "trade_quality",
        "halim_postmortem",
        "WARN",
        detail="closed trades exist but no HALIM postmortem insight",
        evidence={"trades": len(trades), "has_insight": False},
    )


def pnl_sign_consistency(ctx: InspectionContext) -> CheckResult:
    """Closed trades carry meaningful non-zero P&L."""
    trades = _recent_closed(ctx, 20)
    if not trades:
        return CheckResult(
            "trade_quality",
            "pnl_sign_consistency",
            OK,
            detail="no closed trades — warming up",
            evidence={"trades": 0},
        )
    zeros = sum(1 for t in trades if _pnl(t) == 0)
    if zeros == 0:
        return CheckResult(
            "trade_quality",
            "pnl_sign_consistency",
            OK,
            detail="all closed trades carry meaningful P&L",
            evidence={"checked": len(trades)},
        )
    return CheckResult(
        "trade_quality",
        "pnl_sign_consistency",
        "WARN",
        detail=f"{zeros}/{len(trades)} closed trades logged with zero P&L",
        evidence={"zeros": zeros, "checked": len(trades)},
    )


__all__ = [
    "win_rate",
    "profit_factor",
    "halim_postmortem",
    "pnl_sign_consistency",
]
