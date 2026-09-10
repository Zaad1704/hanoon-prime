"""memory joint checks — journal growth, seq, verdict validity, chain anchor."""

from __future__ import annotations

from .checks import FAIL, OK, VALID_ACTIONS, WARN, CheckResult
from .ctx import InspectionContext
from .probe import journal, journal_chain_state, journal_tail


def journal_grows(ctx: InspectionContext) -> CheckResult:
    """Journal count is not below the previous tick's count."""
    cur = journal(ctx).count()
    prev = ctx.prev_journal_count
    if prev is None:
        return CheckResult("memory", "journal_grows", OK, evidence={"count": cur})
    ok = cur >= prev
    return CheckResult(
        "memory",
        "journal_grows",
        OK if ok else FAIL,
        detail=f"count {cur} vs prev {prev}",
    )


def seq_forward(ctx: InspectionContext) -> CheckResult:
    """Recent journal seq values advance without gaps."""
    rows = journal_tail(ctx, 200)
    gaps = 0
    prev_seq: int | None = None
    for r in rows:
        s = r.get("seq")
        if isinstance(s, int):
            if prev_seq is not None and s != prev_seq + 1:
                gaps += 1
            prev_seq = s
    return CheckResult(
        "memory",
        "seq_forward",
        OK if gaps == 0 else WARN,
        detail=f"{gaps} seq gap(s) in last {len(rows)}",
    )


def verdicts_valid(ctx: InspectionContext) -> CheckResult:
    """Recent verdicts carry valid actions and finite scores."""
    rows = journal_tail(ctx, 60)
    sample = [r for r in rows if r.get("event") == "verdict"]
    bad_actions = [
        r for r in sample if str(r.get("action", "")).upper() not in VALID_ACTIONS
    ]
    nan_scores = [
        r
        for r in sample
        if not isinstance(r.get("score"), (int, float))
        or r.get("score") != r.get("score")
    ]
    if bad_actions or nan_scores:
        return CheckResult(
            "memory",
            "verdicts_valid",
            FAIL,
            detail=f"{len(bad_actions)} bad action, {len(nan_scores)} NaN score",
        )
    return CheckResult("memory", "verdicts_valid", OK, evidence={"sample": len(sample)})


def chain_intact_from_anchor(ctx: InspectionContext) -> CheckResult:
    """No journal chain gaps after the bootstrap anchor."""
    cs = journal_chain_state(ctx)
    ok = bool(cs.get("gaps_after_anchor") == 0)
    return CheckResult(
        "memory",
        "chain_intact_from_anchor",
        OK if ok else FAIL,
        detail=f"anchor_seq={cs.get('anchor_seq')} gaps={cs.get('gaps_after_anchor')}",
        evidence=cs,
    )
