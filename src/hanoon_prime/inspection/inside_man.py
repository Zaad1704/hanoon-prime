"""Inside Man checks — monitor the three critical runtime fixes.

Verifies at runtime that:
  1. HALIM modifier is bounded (±0.03) in the fast-path score.
  2. ib_pnl flows to exit logic (profit-lock / giveback actually fire).
  3. Losing confidence bins trigger aggressive learning adaptation.
"""

from __future__ import annotations

from typing import Any

from .checks import FAIL, OK, WARN, CheckResult
from .ctx import InspectionContext
from .probe import runtime_state

# Must match brain.config.HALIM_MOD_BOUND
_HALIM_BOUND: float = 0.03

# Must match brain.realized_ev.CONF_LOSS_STREAK_WARN
_CONF_LOSS_WARN: int = 10

# Exit-reason substrings that indicate ib_pnl-dependent pillars fired.
_PNL_EXIT_REASONS: tuple[str, ...] = ("profit_lock", "giveback")


def brain_halim_bounded(ctx: InspectionContext) -> CheckResult:
    """HALIM modifier stays within ±0.03 in brain_state (fast-path bound)."""
    bs = runtime_state(ctx).get("brain_state", {})
    if not isinstance(bs, dict):
        return CheckResult(
            "inside_man", "brain_halim_bounded", WARN, detail="brain_state missing"
        )
    hm = bs.get("halim_modifier", 0.0)
    if not isinstance(hm, (int, float)):
        return CheckResult(
            "inside_man",
            "brain_halim_bounded",
            FAIL,
            detail=f"halim_modifier not numeric: {hm!r}",
            evidence={"halim_modifier": hm},
        )
    if abs(hm) > _HALIM_BOUND:
        return CheckResult(
            "inside_man",
            "brain_halim_bounded",
            FAIL,
            detail=f"halim_modifier {hm:.4f} exceeds ±{_HALIM_BOUND}",
            evidence={"halim_modifier": round(hm, 4), "bound": _HALIM_BOUND},
        )
    return CheckResult(
        "inside_man",
        "brain_halim_bounded",
        OK,
        evidence={"halim_modifier": round(hm, 4)},
    )


def exits_ib_pnl_fed(ctx: InspectionContext) -> CheckResult:
    """Exit reasons driven by ib_pnl (profit-lock/giveback) appear in journal.

    Before the fix ``check_exit`` was called with ``ib_pnl=0.0``, so
    profit-lock/giveback never fired. Verify they appear in closed-trade
    journal rows now that unrealized P&L is computed from live price.
    """
    from .probe import journal_tail

    rows = journal_tail(ctx, 500)
    closed = [r for r in rows if r.get("event") in ("position_closed", "exit")]
    if not closed:
        return CheckResult(
            "inside_man",
            "exits_ib_pnl_fed",
            OK,
            detail="no closed trades — warming up",
            evidence={"checked": len(rows)},
        )
    pnl_exits = [
        r
        for r in closed
        if any(sub in str(r.get("reason", "")) for sub in _PNL_EXIT_REASONS)
    ]
    if not pnl_exits:
        return CheckResult(
            "inside_man",
            "exits_ib_pnl_fed",
            WARN,
            detail="no profit-lock/giveback exits in recent journal — ib_pnl may not be flowing",
            evidence={"checked": len(rows)},
        )
    return CheckResult(
        "inside_man",
        "exits_ib_pnl_fed",
        OK,
        detail=f"{len(pnl_exits)} profit-lock/giveback exit(s) found",
        evidence={"pnl_exits": len(pnl_exits)},
    )


def conf_bin_loss_streak(ctx: InspectionContext) -> CheckResult:
    """Losing confidence bins (0 wins, ≥10 losses) are detected for aggressive learning.

    Aggressive learning kicks in when a conf bin has zero wins but ≥10
    losses — the threshold should rise faster than the normal
    CONF_MIN_SAMPLES=20 gate. This check verifies the data path exists.
    """
    realized = runtime_state(ctx).get("realized", {})
    if not isinstance(realized, dict):
        return CheckResult(
            "inside_man",
            "conf_bin_loss_streak",
            WARN,
            detail="realized stats missing from runtime state",
        )
    conf_wins = realized.get("conf_wins", {})
    conf_losses = realized.get("conf_losses", {})
    losing_bins: list[dict[str, Any]] = []
    if isinstance(conf_losses, dict):
        for b, losses in conf_losses.items():
            wins = conf_wins.get(b, 0) if isinstance(conf_wins, dict) else 0
            if wins == 0 and losses >= _CONF_LOSS_WARN:
                losing_bins.append({"bin": b, "losses": losses, "wins": wins})
    if losing_bins:
        return CheckResult(
            "inside_man",
            "conf_bin_loss_streak",
            WARN,
            detail=f"{len(losing_bins)} losing conf bin(s) — aggressive learning should raise threshold",
            evidence={"losing_bins": losing_bins},
        )
    return CheckResult(
        "inside_man",
        "conf_bin_loss_streak",
        OK,
        detail="no active loss streaks in conf bins",
        evidence={
            "conf_bins": len(conf_losses) if isinstance(conf_losses, dict) else 0
        },
    )


__all__ = [
    "brain_halim_bounded",
    "exits_ib_pnl_fed",
    "conf_bin_loss_streak",
]
