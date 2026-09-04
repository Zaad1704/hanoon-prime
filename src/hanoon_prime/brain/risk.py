"""hanoon_prime.brain.risk — risk evaluation and position sizing.

Evaluates trade candidates through EV gate, Kelly sizing, and
portfolio heat limits. Calls edge.py for EV + Kelly, hippocampus
for base sizing, eyes for ATR.

Merge of rebuild's ev_gate.py + risk_manager.py + sizer.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..edge import compute_ev, kelly_fraction, score_to_win_prob
from ..immune import (
    ATR_STOP_MULT,
    ATR_TARGET_MULT,
    KELLY_FRACTION,
    MAX_CONCURRENT_POSITIONS,
    MAX_LOSS_PER_TRADE,
    MAX_POSITION_NOTIONAL,
)
from .config import ENTRY_EV_THRESHOLD


@dataclass
class SizingResult:
    """Final position sizing output."""

    shares: int = 0
    stop_price: float = 0.0
    target_price: float = 0.0
    ev: float = 0.0
    kelly: float = 0.0
    risk_pass: bool = False
    reason: str = ""


class RiskEngine:
    """Risk gate and position sizing."""

    def evaluate(
        self,
        score: float,
        confidence: float,
        entry_price: float,
        atr: float,
        open_positions: int,
    ) -> SizingResult:
        """Full risk evaluation. Returns sizing or rejection."""
        win_prob = score_to_win_prob(score)
        ev = compute_ev(win_prob)
        kelly = kelly_fraction(win_prob) * KELLY_FRACTION
        if ev["gross_ev"] < ENTRY_EV_THRESHOLD:
            return SizingResult(
                reason=f"EV {ev['gross_ev']:.3f} < {ENTRY_EV_THRESHOLD}"
            )
        if open_positions >= MAX_CONCURRENT_POSITIONS:
            return SizingResult(reason=f"Max {MAX_CONCURRENT_POSITIONS} positions")
        if atr <= 0 or entry_price <= 0:
            return SizingResult(reason="Invalid ATR/price")
        risk_per_share = atr * ATR_STOP_MULT
        max_by_notional = MAX_POSITION_NOTIONAL / entry_price
        max_by_loss = MAX_LOSS_PER_TRADE / risk_per_share
        max_by_kelly = MAX_POSITION_NOTIONAL * kelly / entry_price
        shares = max(1, int(min(max_by_notional, max_by_loss, max_by_kelly)))
        d = 1 if score > 0 else -1
        stop = round(entry_price - d * ATR_STOP_MULT * atr, 2)
        target = round(entry_price + d * ATR_TARGET_MULT * atr, 2)
        return SizingResult(
            shares=shares,
            stop_price=stop,
            target_price=target,
            ev=ev["gross_ev"],
            kelly=kelly,
            risk_pass=True,
            reason="ok",
        )
