"""purity joint checks: hermetic learning, weights, brain fields."""

from __future__ import annotations

from .checks import FAIL, OK, CheckResult
from .ctx import InspectionContext
from .probe import journal_tail, juli_state, runtime_state

POLLUTED = {"T", "TEST"}


def no_test_episodes(ctx: InspectionContext) -> CheckResult:
    """No T/TEST tickers in juli episodes or journal rows."""
    js = juli_state(ctx)
    eps = [ep for ep in js.get("episodes", []) if isinstance(ep, dict)]
    bad = [ep.get("ticker") for ep in eps if ep.get("ticker") in POLLUTED]
    if bad:
        return CheckResult(
            "purity",
            "no_test_episodes",
            FAIL,
            detail=f"{len(bad)} test episode(s)",
            evidence={"tickers": bad[:5]},
        )
    rows = journal_tail(ctx, 400)
    bad_rows = [r.get("ticker") for r in rows if r.get("ticker") in POLLUTED]
    if bad_rows:
        return CheckResult(
            "purity",
            "no_test_episodes",
            FAIL,
            detail=f"{len(bad_rows)} journal test row(s)",
            evidence={"tickers": bad_rows[:5]},
        )
    return CheckResult("purity", "no_test_episodes", OK)


def weights_finite_in_band(ctx: InspectionContext) -> CheckResult:
    """juli weights are finite and within [-2, 2]."""
    w = juli_state(ctx).get("weights")
    if not isinstance(w, dict):
        return CheckResult(
            "purity", "weights_finite_in_band", FAIL, detail="weights missing/untyped"
        )
    vals = [v for v in w.values() if isinstance(v, (int, float))]
    if len(vals) < 10:
        return CheckResult(
            "purity",
            "weights_finite_in_band",
            FAIL,
            detail=f"sparse weights ({len(vals)})",
        )
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        return CheckResult(
            "purity", "weights_finite_in_band", FAIL, detail="NaN/inf weight"
        )
    if not all(-2.0 <= v <= 2.0 for v in vals):
        return CheckResult(
            "purity", "weights_finite_in_band", FAIL, detail="weight outside [-2,2]"
        )
    return CheckResult(
        "purity", "weights_finite_in_band", OK, evidence={"n": len(vals)}
    )


def brain_fields_bounded(ctx: InspectionContext) -> CheckResult:
    """brain_state fields sit in their declared safe ranges."""
    bs = runtime_state(ctx).get("brain_state", {})
    if not isinstance(bs, dict):
        return CheckResult(
            "purity", "brain_fields_bounded", FAIL, detail="brain_state missing"
        )
    issues: list[str] = []
    t = bs.get("threshold")
    if not (isinstance(t, (int, float)) and 0.45 <= t <= 0.70):
        issues.append(f"threshold={t!r}")
    pe = bs.get("pred_error")
    if pe is not None and not (isinstance(pe, (int, float)) and 0.0 <= pe <= 1.0):
        issues.append(f"pred_error={pe!r}")
    rc = bs.get("risk_ceiling")
    if not (isinstance(rc, (int, float)) and rc > 0):
        issues.append(f"risk_ceiling={rc!r}")
    po = bs.get("positions_open")
    if not (isinstance(po, int) and po >= 0):
        issues.append(f"positions_open={po!r}")
    if issues:
        return CheckResult(
            "purity", "brain_fields_bounded", FAIL, detail="; ".join(issues)
        )
    return CheckResult("purity", "brain_fields_bounded", OK)
