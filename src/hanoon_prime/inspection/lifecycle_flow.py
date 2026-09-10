"""inspection.lifecycle_flow — stages 4–6 of the signal lifecycle.

Split out from ``lifecycle`` to keep each file within the 200-line
contract.  Imports the shared helpers (``_inactive``, ``_get``,
``_stage_result``, ``VERDICT_MAX_AGE_S``) from ``lifecycle``; this is a
bottom-import in that module so the cycle resolves cleanly.
"""

from __future__ import annotations

import time
from typing import Any

from .checks import CheckResult
from .ctx import InspectionContext
from .lifecycle import VERDICT_MAX_AGE_S, _get, _inactive, _stage_result

# ── Stage 4: VERDICT ──────────────────────────────────────────────────


def lifecycle_verdict(ctx: InspectionContext) -> CheckResult:
    """Verdicts are minted recently enough for an active session."""
    if _inactive(ctx):
        return _stage_result("verdict", True, "session inactive", {})
    v = _get(ctx, "/verdicts")
    rows = v.get("verdicts") or []  # array-safe
    if not rows:
        return _stage_result(
            "verdict",
            False,
            "no verdicts yet this session",
            {},
            warn_only=True,
        )
    newest = rows[0] if isinstance(rows[0], dict) else {}
    ts = newest.get("ts")
    age = time.time() - float(ts) if isinstance(ts, (int, float)) else None
    stale = age is not None and age > VERDICT_MAX_AGE_S
    return _stage_result(
        "verdict",
        not stale,
        f"newest verdict {age:.0f}s ago" if age is not None else "verdict ts missing",
        {
            "count": v.get("count", len(rows)),
            "newest_action": newest.get("action"),
            "age_s": round(age, 1) if age is not None else None,
        },
        warn_only=True,
    )


# ── Stage 5: EXECUTION ────────────────────────────────────────────────


def lifecycle_execution(ctx: InspectionContext) -> CheckResult:
    """Open orders / executions reconcile with the bot's position surface."""
    if _inactive(ctx):
        return _stage_result("execution", True, "session inactive", {})
    ib = _get(ctx, "/ib")
    pos = _get(ctx, "/positions")
    ib_pos_raw = ib.get("positions") or []  # array-safe
    ib_positions = [p for p in ib_pos_raw if abs(p.get("position") or 0) > 0]
    surface = pos.get("positions") or []  # array-safe
    ib_syms = {p.get("symbol") for p in ib_positions if p.get("symbol")}
    surface_syms = {
        p.get("ticker") for p in surface if isinstance(p, dict) and p.get("ticker")
    }
    drift = ib_syms ^ surface_syms
    fills = ib.get("executions") or []  # array-safe
    sorted_drift = sorted(drift, key=str)
    return _stage_result(
        "execution",
        not drift,
        f"{len(ib_positions)} IB position(s), {len(fills)} session fill(s)"
        + (f", symbol drift: {sorted_drift}" if drift else ""),
        {
            "ib_positions": len(ib_positions),
            "surface_positions": len(surface),
            "drift": sorted_drift,
            "fills": len(fills),
        },
        warn_only=True,
    )


# ── Stage 6: LEARNING ─────────────────────────────────────────────────


def lifecycle_learning(ctx: InspectionContext) -> CheckResult:
    """Learning organs receive closes: brain memory + regime vectors grow."""
    if _inactive(ctx):
        return _stage_result("learning", True, "session inactive", {})
    br = _get(ctx, "/brain")
    decisions = br.get("decision_count") or 0
    episodic = br.get("episodic_size") or 0
    w = br.get("weights")
    weights = w if isinstance(w, dict) else {}
    learned = sum(1 for v in weights.values() if isinstance(v, (int, float)))
    if decisions == 0 and episodic == 0:
        return _stage_result(
            "learning",
            False,
            "no decisions or episodes recorded",
            {},
            warn_only=True,
        )
    return _stage_result(
        "learning",
        True,
        f"{decisions} decisions, {episodic} episodes, {learned} weights",
        {"decisions": decisions, "episodes": episodic, "weights": learned},
    )
