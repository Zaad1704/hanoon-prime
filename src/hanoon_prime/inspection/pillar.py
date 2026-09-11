"""hanoon_prime.inspection.pillar — directional conviction balance.

LONG vs SHORT verdict-conviction geometry from EVAL lines; evidence feeds the
juli webapp see-saw. Healthy learning = imbalance_ratio trending to 0.
"""

from __future__ import annotations

from typing import Any

from ..brain.policy.trading_policy import TRADING_CONFIG
from ..immune import (
    PILLAR_IMBALANCE_OK,
    PILLAR_IMBALANCE_WARN,
    PILLAR_VETO_SKEW_FAIL,
    PILLAR_VETO_SKEW_WARN,
)
from .checks import FAIL, OK, WARN, CheckResult
from .ctx import InspectionContext
from .pillar_evidence import _BAND, _full_evidence, _parse_eval_lines, _warm_evidence
from .probe import runtime_state, snapshot


def _cr(name: str, status: str, detail: str, **ev: Any) -> CheckResult:
    """Build an inside_man CheckResult."""
    return CheckResult("inside_man", name, status, detail, dict(ev) if ev else {})


def _status(
    ratio: float, skew: float, v_long: int, v_short: int, long_c: float, short_c: float
) -> tuple[str, str]:
    """Map geometry -> (status, detail) using immune thresholds."""
    if ratio > PILLAR_IMBALANCE_WARN or skew >= PILLAR_VETO_SKEW_FAIL:
        side = "SHORT" if (v_short >= v_long or short_c > long_c) else "LONG"
        return (
            FAIL,
            f"pillar fallen — {side}-dominated (ratio={ratio:.3f}, skew={skew:.2f})",
        )
    if ratio > PILLAR_IMBALANCE_OK or skew >= PILLAR_VETO_SKEW_WARN:
        return WARN, f"pillar tipping (ratio={ratio:.3f}, skew={skew:.2f})"
    return OK, f"pillar upright (ratio={ratio:.3f}, skew={skew:.2f})"


def _policy_adjust(
    status: str,
    detail: str,
    mode: str,
    v_long: int,
    v_short: int,
    pv_long: int,
    pv_short: int,
    skew: float,
) -> tuple[str, str]:
    """Downgrade FAIL→WARN when vetoes are policy-driven, not a cortex defect.

    Two checks: (1) reason-aware — gate vetoes (low_penny_score, etc.) block
    both directions equally; (2) direction_mode — direction_rejected only
    counts as policy when the mode actually blocks that side.
    """
    if status == FAIL and pv_short > 0 and pv_short >= v_short * 0.5:
        return WARN, (
            f"SHORT vetoes policy-gated ({pv_short}/{v_short} are "
            f"low_penny_score/daily_loss_limit/etc); skew {skew:.2f} — {detail}"
        )
    if status == FAIL and pv_long > 0 and pv_long >= v_long * 0.5:
        return WARN, (
            f"LONG vetoes policy-gated ({pv_long}/{v_long} are "
            f"low_penny_score/daily_loss_limit/etc); skew {skew:.2f} — {detail}"
        )
    if mode == "long_only" and v_short > v_long and status == FAIL:
        return (
            WARN,
            f"SHORT vetoes policy-driven (long_only blocks SHORT); skew {skew:.2f} — {detail}",
        )
    if mode == "short_only" and v_long > v_short and status == FAIL:
        return (
            WARN,
            f"LONG vetoes policy-driven (short_only blocks LONG); skew {skew:.2f} — {detail}",
        )
    return status, detail


def pillar_balance(ctx: InspectionContext) -> CheckResult:
    """Directional conviction balance — the no-win signature's pillar axis."""
    lc, sc, vl, vs, el, pvl, pvs = _parse_eval_lines(ctx)
    if el == 0:
        return _cr(
            "pillar_balance", OK, "warming up (no EVAL lines yet)", **_warm_evidence(el)
        )
    if lc + sc == 0:
        return _cr(
            "pillar_balance", WARN, "no scored verdicts in window", **_warm_evidence(el)
        )
    total = lc + sc
    ratio = abs(lc - sc) / total if total > 0 else 0.0
    mx, mn = (vl, vs) if vl >= vs else (vs, vl)
    skew = mx / mn if mn > 0 else float(mx)
    bs = runtime_state(ctx).get("brain_state", {})
    hm = bs.get("halim_modifier", 0.0) if isinstance(bs, dict) else 0.0
    cfg = snapshot(ctx).get("config", {})
    mode = cfg.get("direction_mode", TRADING_CONFIG.direction_mode)
    status, detail = _status(ratio, skew, vl, vs, lc, sc)
    status, detail = _policy_adjust(status, detail, mode, vl, vs, pvl, pvs, skew)
    ev = _full_evidence(lc, sc, vl, vs, pvl, pvs, ratio, skew, hm, el)
    ev["direction_mode"] = mode
    return _cr("pillar_balance", status, detail, **ev)


__all__ = ["pillar_balance"]
