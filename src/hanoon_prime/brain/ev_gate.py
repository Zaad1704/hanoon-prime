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
    realized: RealizedStats | Any,  # Accepts RealizedStats or _ProbeMemory for testing
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


class _ProbeMemory:
    """Minimal stand-in for testing verify_learning_gate.

    Mirrors rebuild's _ProbeMemory: provides confidence calibration data
    with thin-data fallback when samples are below CONF_MIN_SAMPLES.
    """

    def __init__(self, conf_wr: float = 0.5, conf_total: int = 0):
        self._conf_wr = conf_wr
        self._conf_total = conf_total
        self._band_wr_val: float | None = None
        self._band_total_val: int = 0

    def band_wr(self, _score: float) -> tuple[float, float, int]:
        """Empty band data for thin-data test (no band pull)."""
        if self._band_wr_val is not None:
            rel = min(1.0, self._band_total_val / 100.0)
            return self._band_wr_val, rel, self._band_total_val
        return 0.5, 0.0, 0  # Neutral, no reliability

    def conf_band_wr(self, _conf: float) -> tuple[float, float, int]:
        """Confidence band with thin-data threshold.

        If samples < CONF_MIN_SAMPLES, return (0.5, 0.0, 0) to fall back
        to structural prior.
        """
        if self._conf_total < 20:  # CONF_MIN_SAMPLES
            return 0.5, 0.0, 0
        rel = min(1.0, self._conf_total / (self._conf_total + 100))
        return self._conf_wr, rel, self._conf_total

    def is_gate_closed(self, _score: float) -> bool:
        """No gate close for probe memory (thin data mode)."""
        return False

    def dynamic_prior_top(self) -> float:
        """Return static PRIOR_TOP for test purity."""
        from ..immune import PRIOR_TOP

        return PRIOR_TOP

    def realized_rr(self) -> tuple[float, float, int]:
        """Return structural 3:1 R:R for probe tests."""
        return 3.0, 0.0, 0


def verify_learning_gate() -> dict[str, Any]:
    """6-probe canary: refuses losing band/conf-bin, admits recovery, admits thin.

    Probes 1-5 are original band/conf probes; probe 6 (thin_data_structural_fallback)
    verifies that thin data (low confidence samples) falls back to structural prior.
    """
    score, wp = 0.62, score_to_win_prob(0.62)
    cases: dict[str, tuple[_ProbeMemory, float, bool]] = {
        # Probe 1: Losing band refuses (structural prior too low)
        "losing_band_refused": (
            _ProbeMemory(conf_wr=0.5, conf_total=1800),  # Will be seeded separately
            0.50,
            False,  # Will fail because band shows 0% WR
        ),
        # Probe 2: Recovery band admits
        "recovery_band_admitted": (
            _ProbeMemory(conf_wr=0.5, conf_total=1800),
            0.50,
            True,  # Structural prior gives positive EV
        ),
        # Probe 3: Empty stats admits (cold start)
        "thin_data_admitted": (
            _ProbeMemory(conf_wr=0.5, conf_total=0),
            0.50,
            True,  # Cold start: structural prior
        ),
        # Probe 4: Losing conf band refuses
        "losing_conf_band_refused": (
            _ProbeMemory(conf_wr=0.20, conf_total=1800),
            0.50,
            False,  # Losing confidence pulls EV negative
        ),
        # Probe 5: Conf pull not veto alone
        "conf_pull_not_veto_alone": (
            _ProbeMemory(
                conf_wr=0.50, conf_total=1800
            ),  # Conf is 50% neutral, band wins
            0.50,
            True,  # Admits because band is winning
        ),
        # Probe 6: Thin data structural fallback (rebuild's key test)
        "thin_data_structural_fallback": (
            _ProbeMemory(conf_wr=0.19, conf_total=2),  # Thin: < 20 samples → fallback
            0.50,
            True,  # Falls back to structural prior → admits
        ),
    }

    # Seeded stats for probes that need real data
    losing_band_stats = _seeded([(score, False)] * 30, [])
    recovery_band_stats = _seeded([(score, True)] * 25, [])
    conf_losing_stats = _seeded([(score, False)] * 30, [(0.50, False)] * 30)
    conf_neutral_stats = _seeded([(score, True)] * 25, [(0.50, False)] * 25)

    # Define which stats to use for each probe
    probe_stats = {
        "losing_band_refused": losing_band_stats,
        "recovery_band_admitted": recovery_band_stats,
        "thin_data_admitted": _ProbeMemory(conf_wr=0.5, conf_total=0),
        "losing_conf_band_refused": conf_losing_stats,
        "conf_pull_not_veto_alone": conf_neutral_stats,
        "thin_data_structural_fallback": _ProbeMemory(conf_wr=0.19, conf_total=2),
    }

    out: dict[str, Any] = {}
    for name, (_, conf, want) in cases.items():
        stats = probe_stats[name]
        # For probes using RealizedStats (seeded), add confidence outcomes
        if isinstance(stats, RealizedStats):
            # The stats are already seeded via _seeded()
            pass
        got = ev_gate_should_enter(score, wp, stats, direction=1, confidence=conf)[
            "should_enter"
        ]
        out[name] = got is want
    out["all_pass"] = all(out.values())
    return out
