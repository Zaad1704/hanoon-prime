"""hanoon_prime.inspection.pillar_evidence — EVAL-window statistics.

Collects LONG/SHORT verdict-conviction geometry from the recent EVAL log
window so the pillar check can judge it.  Living here keeps the pillar
public gate lean; healthy learning = imbalance_ratio trending to 0.
"""

from __future__ import annotations

import re
from typing import Any

from ..immune import HALIM_EVIDENCE_LEARNING, PILLAR_IMBALANCE_OK, PILLAR_IMBALANCE_WARN
from .ctx import InspectionContext
from .probe import EVAL_MARKER, _log_lines

# Captures: ticker:ACTION(score,side)[stage:reason]
_VERDICT_RE = re.compile(
    r":([A-Z]+)\((?P<score>-?\d+\.\d+),(?P<side>[^)]*)\)\[(?P<stage>[^:\]]*):(?P<reason>[^\]]*)\]"
)

# Trading/safety gates that block both directions equally — NOT brain defects.
# direction_rejected is excluded: only the direction_mode fallback demotes it.
_POLICY_VETO_REASONS = frozenset(
    "low_penny_score daily_loss_limit session_disabled sized_to_zero".split()
)

_BAND: list[float] = [PILLAR_IMBALANCE_OK, PILLAR_IMBALANCE_WARN]
# Recent EVAL lines only: stale direction_mode-era vetoes don't tip the pillar.
_EVAL_WINDOW_LINES: int = 1000


def _classify(
    m: re.Match[str],
    score: float,
    conv: dict[str, float],
    vetoes: dict[str, int],
    policy_vetoes: dict[str, int],
) -> None:
    """Bucket a verdict into conviction + veto counters; see _policy_adjust."""
    if score > 0:
        conv["long"] += score
    elif score < 0:
        conv["short"] += -score
    if m.group(1) != "VETOED" or score == 0:
        return
    side = "long" if score > 0 else "short"
    vetoes[side] += 1
    if m.group("reason") in _POLICY_VETO_REASONS:
        policy_vetoes[side] += 1


def _parse_eval_lines(
    ctx: InspectionContext,
) -> tuple[float, float, int, int, int, int, int]:
    """Accumulate conviction + veto counts from the recent EVAL window."""
    conv = {"long": 0.0, "short": 0.0}
    vetoes = {"long": 0, "short": 0}
    policy_vetoes = {"long": 0, "short": 0}
    recent: list[str] = []
    for line in reversed(_log_lines(ctx)):
        if EVAL_MARKER.search(line):
            recent.append(line)
            if len(recent) >= _EVAL_WINDOW_LINES:
                break
    recent.reverse()
    eval_lines = 0
    for line in recent:
        eval_lines += 1
        for m in _VERDICT_RE.finditer(line):
            _classify(m, float(m.group("score")), conv, vetoes, policy_vetoes)
    return (
        conv["long"],
        conv["short"],
        vetoes["long"],
        vetoes["short"],
        eval_lines,
        policy_vetoes["long"],
        policy_vetoes["short"],
    )


def _full_evidence(
    long_c: float,
    short_c: float,
    v_long: int,
    v_short: int,
    pv_long: int,
    pv_short: int,
    ratio: float,
    skew: float,
    hm: Any,
    eval_lines: int,
) -> dict[str, Any]:
    """Evidence shape for a scored evaluation window."""
    return {
        "long_conviction": round(long_c, 4),
        "short_conviction": round(short_c, 4),
        "net": round(long_c - short_c, 4),
        "imbalance_ratio": round(ratio, 4),
        "vetoes_long": v_long,
        "vetoes_short": v_short,
        "policy_vetoes_long": pv_long,
        "policy_vetoes_short": pv_short,
        "veto_skew": round(skew, 3),
        "halim_modifier": hm,
        "learning_active": HALIM_EVIDENCE_LEARNING,
        "eval_lines": eval_lines,
        "band": list(_BAND),
    }


def _warm_evidence(eval_lines: int) -> dict[str, Any]:
    """Evidence for early-exit cases (no lines / no scored verdicts)."""
    return {
        "eval_lines": eval_lines,
        "imbalance_ratio": 0.0,
        "veto_skew": 0.0,
        "band": list(_BAND),
    }


__all__ = ["_BAND", "_parse_eval_lines", "_full_evidence", "_warm_evidence"]
