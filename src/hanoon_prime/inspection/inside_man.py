"""Inside Man checks — runtime guards for the Sep-11 fixes.

1. HALIM modifier stays within ±HALIM_MOD_BOUND (default 0.03) — the
   brain publishes a bounded additive advisory, not a multiplier.
2. ib_pnl-dependent exits (profit_lock/giveback) actually fire.
3. Losing confidence bins surface for aggressive threshold learning.
4. No zero-conviction |score| verdict is labelled direction_rejected.
"""

from __future__ import annotations

import re
from typing import Any

from .checks import FAIL, OK, WARN, CheckResult
from .ctx import InspectionContext
from .probe import EVAL_MARKER, _log_lines, journal_tail, runtime_state

# halim_modifier is a ±0.03 additive bounded modulator (HALIM_MOD_BOUND),
# clamped *before* it enters shared state by ConsolidationEngine._update_halim.
# The old code exposed a [0.5, 1.5] multiplier that was removed in the Sep-11
# fix (commit range): the brain now publishes the additive form.
_HALIM_MOD_BOUND: float = 0.03

# Must match brain.realized_ev.CONF_LOSS_STREAK_WARN
_CONF_LOSS_WARN: int = 10

# Exit-reason substrings indicating ib_pnl-driven exits fired.
_PNL_EXIT_REASONS: tuple[str, ...] = ("profit_lock", "giveback")

# Must match immune.DIRECTION_MIN_SCORE — |score| below this = no_signal.
_DIRECTION_MIN_SCORE: float = 0.02

# Verdict token inside an EVAL line: ticker:ACTION(score,side)[stage:reason]
_VETO_REASON_RE = re.compile(
    r":VETOED\((?P<score>-?\d+\.\d+),[^)]*\)\[(?P<stage>[^:\]]*):(?P<reason>[^\]]*)\]"
)


def _cr(name: str, status: str, detail: str = "", **ev: Any) -> CheckResult:
    """Build a CheckResult on the inside_man joint (DRY within this module)."""
    return CheckResult("inside_man", name, status, detail, dict(ev) if ev else {})


def brain_halim_bounded(ctx: InspectionContext) -> CheckResult:
    """HALIM modifier stays within ±HALIM_MOD_BOUND (default 0.03).

    The brain publishes a bounded additive advisory clamped before it
    enters shared state.  A value beyond the bound indicates a regression
    in ConsolidationEngine._update_halim's clamping logic.
    """
    bs = runtime_state(ctx).get("brain_state", {})
    if not isinstance(bs, dict):
        return _cr("brain_halim_bounded", WARN, detail="brain_state missing")
    hm = bs.get("halim_modifier", 0.0)
    if not isinstance(hm, (int, float)):
        return _cr(
            "brain_halim_bounded",
            FAIL,
            f"halim_modifier not numeric: {hm!r}",
            halim_modifier=hm,
        )
    bound = _HALIM_MOD_BOUND
    if abs(float(hm)) <= bound + 1e-9:
        detail = "HALIM advisory cold (cache miss)" if hm == 0.0 else "within bound"
        return _cr(
            "brain_halim_bounded", OK, detail, halim_modifier=round(hm, 4), bound=bound
        )
    return _cr(
        "brain_halim_bounded",
        FAIL,
        f"halim_modifier {hm:.4f} outside ±{bound} — clamping regression",
        halim_modifier=round(hm, 4),
        bound=bound,
    )


def exits_ib_pnl_fed(ctx: InspectionContext) -> CheckResult:
    """Profit-lock/giveback (ib_pnl-driven exits) appear in the journal."""
    rows = journal_tail(ctx, 500)
    closed = [r for r in rows if r.get("event") in ("position_closed", "exit")]
    if not closed:
        return _cr(
            "exits_ib_pnl_fed", OK, "no closed trades — warming up", checked=len(rows)
        )
    pnl_exits = [
        r
        for r in closed
        if any(s in str(r.get("reason", "")) for s in _PNL_EXIT_REASONS)
    ]
    if not pnl_exits:
        return _cr(
            "exits_ib_pnl_fed",
            WARN,
            "no profit-lock/giveback exits — ib_pnl may not be flowing",
            checked=len(rows),
        )
    return _cr(
        "exits_ib_pnl_fed",
        OK,
        f"{len(pnl_exits)} profit-lock/giveback exit(s)",
        pnl_exits=len(pnl_exits),
    )


def conf_bin_loss_streak(ctx: InspectionContext) -> CheckResult:
    """Losing conf bins (0 wins, ≥10 losses) feed aggressive threshold learning."""
    realized = runtime_state(ctx).get("realized", {})
    if not isinstance(realized, dict):
        return _cr(
            "conf_bin_loss_streak", WARN, "realized stats missing from runtime state"
        )
    conf_wins = realized.get("conf_wins", {})
    conf_losses = realized.get("conf_losses", {})
    losing: list[dict[str, Any]] = []
    if isinstance(conf_losses, dict):
        for b, losses in conf_losses.items():
            wins = conf_wins.get(b, 0) if isinstance(conf_wins, dict) else 0
            if wins == 0 and losses >= _CONF_LOSS_WARN:
                losing.append({"bin": b, "losses": losses, "wins": wins})
    if losing:
        return _cr(
            "conf_bin_loss_streak",
            WARN,
            f"{len(losing)} losing conf bin(s) — threshold should rise faster",
            losing_bins=losing,
        )
    n_bins = len(conf_losses) if isinstance(conf_losses, dict) else 0
    return _cr(
        "conf_bin_loss_streak",
        OK,
        "no active loss streaks in conf bins",
        conf_bins=n_bins,
    )


def veto_conviction_integrity(ctx: InspectionContext) -> CheckResult:
    """No ``direction_rejected`` veto carries near-zero |score| (fix 22b60b1).

    A direction verdict with |score| < DIRECTION_MIN_SCORE means the cortex
    had no conviction; it must surface as ``no_signal`` HOLD — never as a
    directional VETOED. Such vetoes are the "no winning decisions" signature.
    """
    bogus: list[dict[str, Any]] = []
    eval_lines = 0
    for line in _log_lines(ctx):
        if not EVAL_MARKER.search(line):
            continue
        eval_lines += 1
        for m in _VETO_REASON_RE.finditer(line):
            raw = m.group("score")
            reason = m.group("reason")
            if raw is None or reason is None:
                continue
            score = float(raw)
            if abs(score) < _DIRECTION_MIN_SCORE and reason == "direction_rejected":
                bogus.append({"score": round(score, 3), "reason": reason})
    if bogus:
        return _cr(
            "veto_conviction_integrity",
            FAIL,
            f"{len(bogus)} zero-conviction direction_rejected veto(s) — regression",
            bogus_vetoes=len(bogus),
            sample=bogus[:5],
        )
    return _cr(
        "veto_conviction_integrity",
        OK,
        "no zero-conviction direction_rejected vetoes found",
        scanned_eval_lines=eval_lines,
    )


__all__ = [
    "brain_halim_bounded",
    "conf_bin_loss_streak",
    "exits_ib_pnl_fed",
    "veto_conviction_integrity",
]
