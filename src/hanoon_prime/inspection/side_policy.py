"""Inside Man diagnostic — which policy gate vetoes recent verdicts, and did
the cortex have real conviction behind the rejection.

A direction_rejected veto carrying a high |cortex score| means real conviction
is being discarded by a side policy (e.g. SHORT disallowed on a cash account)
— the "why are we not winning?" signature. A direction_rejected veto with a
near-zero score is a genuine zero-conviction failure.

Verdict tokens are only score-bearing once orchestrator surfaces thought.score
onto the direction_rejected Verdict; otherwise every veto logs 0.000 and this
guard would false-positive. See orchestrator._apply_fast_gates.
"""
from __future__ import annotations

import re
from typing import Any

from .checks import OK, WARN, CheckResult
from .ctx import InspectionContext
from .probe import EVAL_MARKER, _log_lines

# TICKER:ACTION(score,side)[stage:reason]
_VERDICT_RE = re.compile(
    r"(?P<ticker>[A-Z0-9.]+):(?P<action>VETOED|HOLD|BUY|SELL|PRESSED|ENTER)"
    r"\((?P<score>-?\d+\.\d+),[^)]*\)\[(?P<stage>[^:\]]*):(?P<reason>[^\]]*)\]"
)
# Mean |cortex score| on direction_rejected vetoes above which the veto is
# flagged as "real conviction discarded by the side policy" rather than noise.
_REAL_CONVICTION: float = 0.5
# Scan only the most recent verdicts to keep the diagnostic responsive.
_WINDOW: int = 1000
_DIRECTION_REJECTED: str = "direction_rejected"


def _scan_verdicts(
    ctx: InspectionContext,
) -> tuple[int, dict[str, int], list[float]]:
    """Parse recent EVAL verdicts into (eval_lines, vetoes_by_reason, dir_scores)."""
    by_reason: dict[str, int] = {}
    dir_scores: list[float] = []
    eval_lines = 0
    for line in _log_lines(ctx)[-_WINDOW:]:
        if not EVAL_MARKER.search(line):
            continue
        eval_lines += 1
        for m in _VERDICT_RE.finditer(line):
            if m.group("action") != "VETOED":
                continue
            reason = m.group("reason")
            by_reason[reason] = by_reason.get(reason, 0) + 1
            if reason == _DIRECTION_REJECTED:
                dir_scores.append(abs(float(m.group("score"))))
    return eval_lines, by_reason, dir_scores


def side_policy_conviction(ctx: InspectionContext) -> CheckResult:
    """Aggregate recent vetoes by reason + direction_rejected conviction."""
    eval_lines, by_reason, dir_scores = _scan_verdicts(ctx)
    total = sum(by_reason.values())
    n_dir = len(dir_scores)
    mean_dir = (sum(dir_scores) / n_dir) if dir_scores else 0.0
    ev: dict[str, Any] = {
        "checked_eval_lines": eval_lines,
        "vetoes": total,
        "by_reason": by_reason,
        "direction_rejected": n_dir,
        "mean_abs_score": round(mean_dir, 3),
    }
    if n_dir and mean_dir >= _REAL_CONVICTION:
        return CheckResult(
            "inside_man",
            "side_policy_conviction",
            WARN,
            f"{n_dir} direction_rejected vetoes carry mean|score|={mean_dir:.3f} "
            "- real conviction discarded by the side policy "
            "(SHORT disallowed on a cash account); vetoes={by_reason}",
            ev,
        )
    if not eval_lines:
        return CheckResult(
            "inside_man",
            "side_policy_conviction",
            OK,
            "no recent EVAL verdicts - warming up",
            ev,
        )
    return CheckResult(
        "inside_man",
        "side_policy_conviction",
        OK,
        f"no real-conviction short-side vetoes discarded "
        f"(direction_rejected mean|score|={mean_dir:.3f}); "
        f"vetoes={by_reason or 'none'}",
        ev,
    )


__all__ = ["side_policy_conviction"]
