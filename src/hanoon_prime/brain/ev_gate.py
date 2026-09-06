"""brain.ev_gate — Realized-EV entry gate (rebuild ev_gate.py gate math).

Pure gate computation: blends structural EV with realized band WR,
realized R:R and confidence-bin WR, then decides entry. Split out of
``realized_ev.py`` so the stats store and the gate math each stay under
the 200-line R3b limit.
"""

from __future__ import annotations

from typing import Any, Optional

from ..edge import score_to_win_prob
from ..immune import TARGET_R_R
from .config import CONSERVATIVE_EV_MIN, DIRECTION_EXP_BOUND, ENTRY_EV_THRESHOLD
from .realized_ev import RealizedStats


def _direction_mod(direction: int) -> float:
    """Shorts face a bounded expectancy penalty (DIRECTION_EXP_BOUND)."""
    return 1.0 - DIRECTION_EXP_BOUND if direction < 0 else 1.0


def _blend_priors(
    score: float,
    win_prob: float,
    realized: Optional[RealizedStats],
    confidence: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Two-stage prior pull: structural → band → confidence-bin.

    Returns ``(p, r, band_wr, band_rel, conf_wr, conf_rel, rr, rr_rel)``
    where ``p``/``r`` are the pulled win-probability and reward:risk.
    """
    if realized is None:
        p = float(win_prob)
        return p, float(TARGET_R_R), p, 0.0, p, 0.0, float(TARGET_R_R), 0.0
    b_wr, b_rel, _n = realized.band_wr(score)
    rr, rr_rel0, _n2 = realized.realized_rr()
    c_wr, c_rel, _n3 = realized.conf_band_wr(confidence)
    mod = 1.0 - confidence
    band_rel = b_rel * mod
    rr_rel = rr_rel0 * mod
    conf_rel = c_rel * mod
    p0 = win_prob * (1.0 - band_rel) + b_wr * band_rel
    p = p0 * (1.0 - conf_rel) + c_wr * conf_rel
    r = float(TARGET_R_R) * (1.0 - rr_rel) + rr * rr_rel
    return p, r, b_wr, band_rel, c_wr, conf_rel, rr, rr_rel


def compute_ev_and_entry(
    score: float,
    win_prob: float,
    realized: Optional[RealizedStats],
    direction: int = 1,
    confidence: float = 0.5,
) -> dict[str, Any]:
    """Blend structural EV with realized band/RR/confidence-bin data.

    ``confidence`` scales the realized pull (high-confidence => trust the
    model). A proven-losing confidence band drags EV down until the gate
    refuses, mirroring rebuild's calibration-curve correction but inside
    the bounded single gate.
    """
    p, r, band_wr, band_rel, conf_wr, conf_rel, rr, rr_rel = _blend_priors(
        score, win_prob, realized, confidence
    )
    ev = (p * r - (1.0 - p)) * _direction_mod(direction)
    gate = realized.is_gate_closed(score) if realized is not None else False
    return {
        "ev": ev,
        "p": p,
        "r": r,
        "p_struct": win_prob,
        "r_struct": float(TARGET_R_R),
        "band_wr": band_wr,
        "band_rel": band_rel,
        "conf_wr": conf_wr,
        "conf_rel": conf_rel,
        "realized_rr": rr,
        "rr_rel": rr_rel,
        "direction_mod": _direction_mod(direction),
        "gate_closed": gate,
        "direction": direction,
    }


def ev_gate_should_enter(
    score: float,
    win_prob: float,
    realized: Optional[RealizedStats],
    direction: int = 1,
    confidence: float = 0.5,
) -> dict[str, Any]:
    """Full realized-EV entry gate: learned gate + realized EV + direction."""
    info = compute_ev_and_entry(score, win_prob, realized, direction, confidence)
    ev = info["ev"]
    floor = CONSERVATIVE_EV_MIN if info["band_rel"] > 0.0 else ENTRY_EV_THRESHOLD
    if info["gate_closed"]:
        should, reason = False, "learned gate closed (loss band)"
    elif ev <= floor:
        tag = "conservative" if info["band_rel"] > 0.0 else "structural"
        should, reason = False, f"EV {ev:.4f} < {floor:.3f} ({tag})"
    else:
        should, reason = True, "ok"
    info["should_enter"] = should
    info["reason"] = reason
    return info


def _seeded(
    trades: list[tuple[float, bool]], conf_trades: list[tuple[float, bool]]
) -> RealizedStats:
    """Build a non-persisted RealizedStats pre-loaded with probe outcomes."""
    stats = RealizedStats(persist=False)
    for score, won in trades:
        stats.add_outcome(score, won=won, pnl_pct=0.02 if won else -0.01)
    for conf, won in conf_trades:
        stats.add_confidence_outcome(conf, won)
    return stats


def verify_learning_gate() -> dict[str, Any]:
    """5-probe canary: refuse losing band/conf-bin, admit recovery, admit thin.

    Probes 1-3 are the original band probes; probe 4 proves the
    confidence-bin correction refuses a losing confidence band; probe 5
    proves a losing conf-bin with winning band data still trades (the
    conf pull must not veto on its own when the band is proven).
    """
    score, wp = 0.62, score_to_win_prob(0.62)
    cases: dict[str, tuple[RealizedStats, float, bool]] = {
        "losing_band_refused": (_seeded([(score, False)] * 30, []), 0.5, False),
        "recovery_band_admitted": (_seeded([(score, True)] * 25, []), 0.5, True),
        "thin_data_admitted": (RealizedStats(persist=False), 0.5, True),
        "losing_conf_band_refused": (
            _seeded([(score, False)] * 30, [(0.50, False)] * 30),
            0.50,
            False,
        ),
        "conf_pull_not_veto_alone": (
            _seeded([(score, True)] * 25, [(0.50, False)] * 25),
            0.50,
            True,
        ),
    }
    out: dict[str, Any] = {}
    for name, (stats, conf, want) in cases.items():
        got = ev_gate_should_enter(score, wp, stats, direction=1, confidence=conf)[
            "should_enter"
        ]
        out[name] = got is want
    out["all_pass"] = all(out.values())
    return out
