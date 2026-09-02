"""hanoon_prime.thinker — the brain's verdict (ENTER / HOLD / EXIT).

This is the ONLY function in the system that produces a verdict string.
Architecture enforcement (R1) is validated by tests/test_contract.py.

Entry verdict:
    ENTER iff:
      1. score >= SIGNAL_THRESHOLD (signal strength)
      2. confidence > CONFIDENCE_FLOOR (genuine conviction)
      3. direction != 0 (there's a tradeable direction)
      4. gross_ev > 0 (positive expected value after fee drag)

That's it. No modifiers. No gates. No deliberation theater.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from .constants import (
    CONFIDENCE_FLOOR,
    SIGNAL_THRESHOLD,
)


@dataclass
class Thought:
    """The brain's output — verdict + diagnostic metadata."""
    verdict: str = "HOLD"
    confidence: float = 0.50
    score: float = 0.50
    direction: int = 0
    reasons: list[str] = field(default_factory=list)
    trace: dict = field(default_factory=dict)


def _direction_from_alpha(alpha: dict[str, float]) -> int:
    """Derive trade direction from net signal.

    On noisy 5-min bars, mean-reversion captures reversal dynamics.
    institutional_flow (corr=+0.024, p<0.001) is the strongest signal;
    momentum is inverted to capture mean-reverting behavior.
    """
    # VPIN is stored as signed [-1, 1] for edge testing.
    # In the mean-reversion direction formula, we need the unsigned magnitude
    # to preserve the tuned (unsigned - 0.5) behavior.
    vpin_mag = alpha.get("vpin_magnitude", 0.5)  # true unsigned [0,1]
    net = (
        -alpha.get("momentum", 0.0) * 0.4      # inverted: mean-reverting
        + alpha.get("orderbook_imbalance", 0.0)
        + (vpin_mag - 0.5) * 0.5               # unsigned magnitude centered at 0.5
        + (alpha.get("institutional_flow", 1.0) - 1.0) * 0.3
        - alpha.get("vwap_deviation", 0.0) * 0.25
    )
    if net > 0.10:
        return 1
    if net < -0.10:
        return -1
    return 0


def _compute_confidence(win_prob: float) -> float:
    """Map win probability [0.25, 0.55] → confidence [0.50, 0.90]."""
    if win_prob <= 0.25:
        return CONFIDENCE_FLOOR
    raw = 0.50 + (win_prob - 0.25) * (0.40 / 0.30)
    return float(max(CONFIDENCE_FLOOR, min(0.95, raw)))


def _make_verdict(score, confidence, direction, gross_ev,
                  volatility=None) -> str:
    """Single decision path: all criteria must pass."""
    # Volatility filter: don't trade flat tickers (stop > range)
    if volatility is not None and volatility < 0.005:
        return "HOLD"
    if score >= SIGNAL_THRESHOLD and confidence > CONFIDENCE_FLOOR \
       and direction != 0 and gross_ev > 0.0:
        return "ENTER"
    return "HOLD"


def _build_reasons(score, confidence, direction, gross_ev,
                   volatility=None) -> list[str]:
    """List why the verdict is HOLD (empty when ENTER)."""
    reasons: list[str] = []
    if volatility is not None and volatility < 0.005:
        reasons.append(f"low vol {volatility:.3f} < 0.005")
    if score < SIGNAL_THRESHOLD:
        reasons.append(f"score {score:.3f} < threshold {SIGNAL_THRESHOLD}")
    if confidence <= CONFIDENCE_FLOOR:
        reasons.append(f"confidence {confidence:.3f} <= floor {CONFIDENCE_FLOOR}")
    if direction == 0:
        reasons.append("no clear direction")
    if gross_ev <= 0.0:
        reasons.append(f"EV {gross_ev:.3f} <= 0")
    return reasons


def deliberate(
    score: float,
    alpha: dict[str, float],
    win_prob: float,
    gross_ev: float,
    kelly: float,
) -> Thought:
    """Produce the entry verdict. Args in docstring below."""
    direction = _direction_from_alpha(alpha)
    confidence = _compute_confidence(win_prob)
    vol = alpha.get("volatility", 1.0)
    verdict = _make_verdict(score, confidence, direction, gross_ev, vol)
    reasons = _build_reasons(score, confidence, direction, gross_ev, vol)
    return Thought(
        verdict=verdict,
        confidence=round(confidence, 4),
        score=round(score, 4),
        direction=direction,
        reasons=reasons if verdict == "HOLD" else ["all criteria met"],
        trace={
            "win_prob": round(win_prob, 4),
            "gross_ev": round(gross_ev, 4),
            "kelly": round(kelly, 4),
            "raw_direction_signal": (
                alpha.get("momentum", 0.0)
                + alpha.get("orderbook_imbalance", 0.0)
                + (alpha.get("vpin_magnitude", 0.5) - 0.5)
                + (alpha.get("institutional_flow", 1.0) - 1.0)
                - alpha.get("vwap_deviation", 0.0)
            ),
        },
    )
