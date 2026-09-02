"""hanoon_prime.edge — score → win probability → EV → entry decision.

Keeps the score-to-win-probability mapping HONEST and HONESTLY BOUNDED.

  win_prob = PRIOR_BOTTOM + (score_norm * (PRIOR_TOP - PRIOR_BOTTOM))

  score normalized [0,1] within [THRESHOLD_MIN, THRESHOLD_MAX]
  → win_prob in [0.25, 0.55]

  EV per unit = p * R - (1 - p)
  Entry iff EV > 0 and gross > 0 (before fees, fees checked at sizing).
"""
from __future__ import annotations

import math
from .constants import (
    FIXED_FEE,
    FEE_RATE,
    PRIOR_BOTTOM,
    PRIOR_TOP,
    TARGET_R_R,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
)


def score_to_win_prob(score: float) -> float:
    """Map score ∈ [THRESHOLD_MIN, THRESHOLD_MAX] to [PRIOR_BOTTOM, PRIOR_TOP].

    Honest linear map. No band-aid, no dynamic trickery at the score→prob
    boundary. The dynamic prior_top adjustment (if any) happens at the
    Brain level, not here.
    """
    s = float(max(THRESHOLD_MIN, min(THRESHOLD_MAX, score)))
    span = THRESHOLD_MAX - THRESHOLD_MIN
    norm = (s - THRESHOLD_MIN) / span if span > 0 else 0.5
    return float(PRIOR_BOTTOM + norm * (PRIOR_TOP - PRIOR_BOTTOM))


def compute_fee_drag(position_notional: float) -> float:
    """Round-trip fee drag in dollars: 2 × (fixed + rate × notional)."""
    return 2.0 * (FIXED_FEE + FEE_RATE * position_notional)


def compute_ev(
    win_prob: float,
    reward_risk: float = TARGET_R_R,
    position_notional: float = 0.0,
    stop_pct: float = 0.021,
) -> dict[str, float]:
    """Compute gross and net EV.

    gross_ev = p * R - (1 - p)            (per unit risk)
    net_ev   = gross_ev - fee_drag_in_R   (if notional provided)

    fee_drag_in_R = fees / (notional * stop_pct)
    """
    p = float(max(0.0, min(1.0, win_prob)))
    R = float(max(reward_risk, 1.0))
    gross_ev = p * R - (1.0 - p)

    net_ev = gross_ev
    if position_notional > 0.0 and stop_pct > 0.0:
        risk_amount = position_notional * stop_pct
        fee_drag = compute_fee_drag(position_notional)
        net_ev = gross_ev - (fee_drag / risk_amount if risk_amount > 0 else 0.0)

    return {
        "win_prob": round(p, 4),
        "loss_prob": round(1.0 - p, 4),
        "gross_ev": round(gross_ev, 4),
        "net_ev": round(net_ev, 4),
        "ev_enter": gross_ev > 0.0,  # decision uses gross; sizing uses net
        "fee_drag": round(compute_fee_drag(position_notional), 2) if position_notional > 0 else 0.0,
    }


def kelly_fraction(win_prob: float, reward_risk: float = TARGET_R_R) -> float:
    """Half-Kelly fraction for sizing. f* = (p*(R+1) - 1) / R."""
    p = float(max(0.0, min(1.0, win_prob)))
    R = float(max(reward_risk, 1.0))
    if p <= 0.0:
        return 0.0
    f = (p * (R + 1.0) - 1.0) / R
    return float(max(0.0, min(0.5, f)))
