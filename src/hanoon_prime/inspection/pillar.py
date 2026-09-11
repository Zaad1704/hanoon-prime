"""hanoon_prime.inspection.pillar — directional conviction balance.

The 'why not winning' signature is a tipped pillar: one direction's vetoes
dominate (today: SHORT direction_rejected @ mean conviction 0.675).
pillar_balance measures the LONG vs SHORT verdict-conviction geometry from EVAL
lines and confirms (via the inside_man joint) when the pillar is upright vs
fallen. Its evidence is the see-saw the juli webapp renders, and the learning
loop is healthy when imbalance_ratio trends toward 0 under HALIM recs.

Split into narrow helpers so every function stays under R3's 40-line / depth-3
limits while keeping the full decision chain readable.
"""

from __future__ import annotations

import re
from typing import Any

from ..immune import (
    HALIM_EVIDENCE_LEARNING,
    PILLAR_IMBALANCE_OK,
    PILLAR_IMBALANCE_WARN,
    PILLAR_VETO_SKEW_FAIL,
    PILLAR_VETO_SKEW_WARN,
)
from .checks import FAIL, OK, WARN, CheckResult
from .ctx import InspectionContext
from .probe import EVAL_MARKER, _log_lines, runtime_state

# Captures every verdict token: ticker:ACTION(score,side)[stage:reason]
_VERDICT_RE = re.compile(
    r":([A-Z]+)\((?P<score>-?\d+\.\d+),(?P<side>[^)]*)\)\[(?P<stage>[^:\]]*):(?P<reason>[^\]]*)\]"
)

# Conviction + veto thresholds (see immune.py) — kept local as a value object.
_BAND: list[float] = [PILLAR_IMBALANCE_OK, PILLAR_IMBALANCE_WARN]


def _cr(name: str, status: str, detail: str, **ev: Any) -> CheckResult:
    """Build an inside_man CheckResult (DRY within this module)."""
    return CheckResult("inside_man", name, status, detail, dict(ev) if ev else {})


def _classify(
    m: re.Match[str], score: float, conv: dict[str, float], vetoes: dict[str, int]
) -> None:
    """Bucket one verdict into directional conviction + veto counters."""
    if score > 0:
        conv["long"] += score
    elif score < 0:
        conv["short"] += -score
    if m.group(1) == "VETOED":
        if score > 0:
            vetoes["long"] += 1
        elif score < 0:
            vetoes["short"] += 1


def _parse_eval_lines(ctx: InspectionContext) -> tuple[float, float, int, int, int]:
    """Accumulate long/short conviction + veto counts from EVAL log lines."""
    conv = {"long": 0.0, "short": 0.0}
    vetoes = {"long": 0, "short": 0}
    eval_lines = 0
    for line in _log_lines(ctx):
        if not EVAL_MARKER.search(line):
            continue
        eval_lines += 1
        for m in _VERDICT_RE.finditer(line):
            _classify(m, float(m.group("score")), conv, vetoes)
    return conv["long"], conv["short"], vetoes["long"], vetoes["short"], eval_lines


def _geometry(
    long_c: float, short_c: float, v_long: int, v_short: int
) -> tuple[float, float, float]:
    """Return (imbalance_ratio, veto_skew, total_conviction)."""
    total = long_c + short_c
    ratio = abs(long_c - short_c) / total if total > 0 else 0.0
    mx, mn = (v_long, v_short) if v_long >= v_short else (v_short, v_long)
    # max/min veto ratio; one-sided (mn==0) -> raw majority count = inf skew.
    skew = mx / mn if mn > 0 else float(mx)
    return ratio, skew, total


def _halim_modifier(ctx: InspectionContext) -> Any:
    """Current HALIM regime modifier (0.0 when the slow cortex hasn't spoken)."""
    bs = runtime_state(ctx).get("brain_state", {})
    if isinstance(bs, dict):
        return bs.get("halim_modifier", 0.0)
    return 0.0


def _warm_evidence(eval_lines: int, band: list[float]) -> dict[str, Any]:
    """Evidence shape for the no-signal / no-lines early exits."""
    return {
        "eval_lines": eval_lines,
        "imbalance_ratio": 0.0,
        "veto_skew": 0.0,
        "band": list(band),
    }


def _full_evidence(
    long_c: float,
    short_c: float,
    v_long: int,
    v_short: int,
    ratio: float,
    skew: float,
    hm: Any,
    eval_lines: int,
    band: list[float],
) -> dict[str, Any]:
    """Evidence shape for a scored, evaluated window."""
    return {
        "long_conviction": round(long_c, 4),
        "short_conviction": round(short_c, 4),
        "net": round(long_c - short_c, 4),
        "imbalance_ratio": round(ratio, 4),
        "vetoes_long": v_long,
        "vetoes_short": v_short,
        "veto_skew": round(skew, 3),
        "halim_modifier": hm,
        "learning_active": HALIM_EVIDENCE_LEARNING,
        "eval_lines": eval_lines,
        "band": list(band),
    }


def _status(
    ratio: float, skew: float, v_long: int, v_short: int, long_c: float, short_c: float
) -> tuple[str, str]:
    """Map geometry -> (status, detail) using immune thresholds."""
    if ratio > PILLAR_IMBALANCE_WARN or skew >= PILLAR_VETO_SKEW_FAIL:
        if skew >= PILLAR_VETO_SKEW_FAIL:
            side = "SHORT" if v_short >= v_long else "LONG"
        else:
            side = "SHORT" if short_c > long_c else "LONG"
        return (
            FAIL,
            f"pillar fallen — {side}-dominated (ratio={ratio:.3f}, skew={skew:.2f})",
        )
    if ratio > PILLAR_IMBALANCE_OK or skew >= PILLAR_VETO_SKEW_WARN:
        return WARN, f"pillar tipping (ratio={ratio:.3f}, skew={skew:.2f})"
    return OK, f"pillar upright (ratio={ratio:.3f}, skew={skew:.2f})"


def pillar_balance(ctx: InspectionContext) -> CheckResult:
    """Directional conviction balance — the no-win signature's pillar axis."""
    long_c, short_c, v_long, v_short, eval_lines = _parse_eval_lines(ctx)
    if eval_lines == 0:
        return _cr(
            "pillar_balance",
            OK,
            "warming up (no EVAL lines yet)",
            **_warm_evidence(eval_lines, _BAND),
        )
    if long_c + short_c == 0:
        return _cr(
            "pillar_balance",
            WARN,
            "no scored verdicts in window",
            **_warm_evidence(eval_lines, _BAND),
        )
    ratio, skew, _total = _geometry(long_c, short_c, v_long, v_short)
    hm = _halim_modifier(ctx)
    ev = _full_evidence(
        long_c, short_c, v_long, v_short, ratio, skew, hm, eval_lines, _BAND
    )
    status, detail = _status(ratio, skew, v_long, v_short, long_c, short_c)
    return _cr("pillar_balance", status, detail, **ev)


__all__ = ["pillar_balance"]
