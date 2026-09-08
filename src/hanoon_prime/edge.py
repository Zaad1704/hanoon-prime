"""hanoon_prime.edge — score → win probability → EV → Kelly.

R5: No score inversion. The mapping is direction-agnostic —
|score| maps to win probability, while the sign only indicates
LONG vs SHORT direction. This allows both directions to benefit
equally from high-confidence signals.

  win_prob = PRIOR_BOTTOM + |score| × (PRIOR_TOP - PRIOR_BOTTOM)

  EV per unit = p * R - (1 - p)
  Entry iff EV > 0 (before fees; fees checked at sizing).
"""

from __future__ import annotations

from .immune import (
    DYNAMIC_PRIOR_TOP_BLEND,
    DYNAMIC_PRIOR_TOP_ENABLED,
    DYNAMIC_PRIOR_TOP_MAX,
    DYNAMIC_PRIOR_TOP_MIN,
    DYNAMIC_PRIOR_TOP_MIN_TRADES,
    DYNAMIC_PRIOR_TOP_SCALE,
    FEE_RATE,
    FIXED_FEE,
    PRIOR_BOTTOM,
    PRIOR_TOP,
    PRIOR_TOP_MAX,
    TARGET_R_R,
)


def score_to_win_prob(score: float, prior_top: float | None = None) -> float:
    """Map |tanh score| ∈ [0, 1] → win probability ∈ [PRIOR_BOTTOM, prior_top].

    Direction-agnostic: both extreme positive (LONG) and extreme
    negative (SHORT) scores map to high win probability. The sign
    only indicates direction, not confidence (R5: no inversion, |score|).

    ``prior_top`` widens/tightens the cap from realized performance
    (see ``get_dynamic_prior_top`` — faithful port of
    brain.memory.get_dynamic_prior_top, memory.py:356-391). None
    (default) uses the structural PRIOR_TOP — cold-start safe, R5-bounded
    (never above PRIOR_TOP_MAX = 0.65).
    """
    pt = PRIOR_TOP if prior_top is None else float(prior_top)
    s = abs(max(-1.0, min(1.0, score)))
    return float(PRIOR_BOTTOM + s * (pt - PRIOR_BOTTOM))


def get_dynamic_prior_top(win_rate: float, n: int) -> float:
    """Dynamic PRIOR_TOP: EARN a higher confidence cap from realized WR.

    Faithful port of rebuild ``brain.memory.Memory.get_dynamic_prior_top``
    (hanoon/juli/memory.py:356-391): cold-start uses the static PRIOR_TOP
    until ``DYNAMIC_PRIOR_TOP_MIN_TRADES`` real trades prove otherwise.
    Deviation from a 0.45 reference is blended 70/30 with the static cap
    and hard-clamped to ``[DYNAMIC_PRIOR_TOP_MIN, DYNAMIC_PRIOR_TOP_MAX]``;
    the ceiling equals PRIOR_TOP_MAX, so the entry gate (R5) never
    over-believes past 0.65.

    WR=0.55 → +0.06 delta → blended ~0.64 (caps at 0.65);
    a weak WR tightens the cap below the static PRIOR_TOP.
    """
    if not DYNAMIC_PRIOR_TOP_ENABLED or n < DYNAMIC_PRIOR_TOP_MIN_TRADES:
        return PRIOR_TOP
    wr = float(max(0.0, min(1.0, win_rate)))
    deviation = wr - 0.45
    dynamic_prior = PRIOR_TOP + deviation * DYNAMIC_PRIOR_TOP_SCALE
    blended = dynamic_prior * DYNAMIC_PRIOR_TOP_BLEND + PRIOR_TOP * (
        1.0 - DYNAMIC_PRIOR_TOP_BLEND
    )
    return float(max(DYNAMIC_PRIOR_TOP_MIN, min(DYNAMIC_PRIOR_TOP_MAX, blended)))


def compute_fee_drag(position_notional: float) -> float:
    """Round-trip fee drag in dollars: 2 × (fixed + rate × notional)."""
    return 2.0 * (FIXED_FEE + FEE_RATE * position_notional)


def compute_ev(
    win_prob: float,
    reward_risk: float = TARGET_R_R,
    position_notional: float = 0.0,
    stop_pct: float = 0.0,
) -> dict[str, float]:
    """Compute gross and net EV per unit of risk.

    gross_ev = p * R - (1 - p)          (per unit risk)
    net_ev   = gross_ev - fee_drag_in_R  (if notional provided)
    """
    p = float(max(0.0, min(1.0, win_prob)))
    r = float(max(reward_risk, 1.0))
    gross_ev = p * r - (1.0 - p)

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
        "ev_enter": gross_ev > 0.0,
        "fee_drag": (
            round(compute_fee_drag(position_notional), 2)
            if position_notional > 0
            else 0.0
        ),
    }


def kelly_fraction(win_prob: float, reward_risk: float = TARGET_R_R) -> float:
    """Fractional Kelly for position sizing: f* = (p*(R+1) - 1) / R."""
    p = float(max(0.0, min(1.0, win_prob)))
    r = float(max(reward_risk, 1.0))
    if p <= 0.0:
        return 0.0
    f = (p * (r + 1.0) - 1.0) / r
    return float(max(0.0, min(1.0, f)))
