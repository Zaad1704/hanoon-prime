"""hanoon_prime.brain.risk — risk evaluation and position sizing.

Evaluates trade candidates through EV gate, Kelly sizing, and
portfolio heat limits. Calls edge.py for EV + Kelly, hippocampus
for base sizing, eyes for ATR.

Merge of rebuild's ev_gate.py + risk_manager.py + sizer.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..edge import kelly_fraction, score_to_win_prob
from ..immune import (
    ATR_STOP_MULT,
    ATR_TARGET_MULT,
    KELLY_FRACTION,
    MAX_CONCURRENT_POSITIONS,
    MAX_LOSS_PER_TRADE,
    MAX_POSITION_NOTIONAL,
)
from .config import PENNY_NOTIONAL_CAPS
from .realized_ev import RealizedStats, ev_gate_should_enter


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
    """Risk gate and position sizing (structural EV + realized learning gate).

    The realized-EV gate pulls the structural win-probability and R:R toward
    what actually happened (RealizedStats); with no realized data it falls
    back to the structural math, so existing structural tests keep passing.
    """

    def __init__(self, realized: Optional[RealizedStats] = None) -> None:
        """Optionally bind a RealizedStats view for the learned EV gate."""
        self._realized = realized

    @staticmethod
    def _invalid_inputs(score: float, entry_price: float, atr: float) -> str | None:
        """Return a rejection reason if inputs are non-finite/invalid, else None.

        IB snapshots can carry NaN for last/close when a tick is stale or
        mid-update. NaN survives ``<= 0`` checks (NaN <= 0 is False) and
        would crash ``int(NaN)``. (Fix #75)
        """
        if not math.isfinite(entry_price) or entry_price <= 0:
            return "Invalid entry_price (non-finite or non-positive)"
        if not math.isfinite(atr) or atr <= 0:
            return "Invalid ATR (non-finite or non-positive)"
        if not math.isfinite(score):
            return "Invalid score (non-finite)"
        return None

    @staticmethod
    def _penny_notional_cap(entry_price: float) -> float:
        """Max dollar notional for a price-tier (penny-stock blowup guard)."""
        for upper, cap in PENNY_NOTIONAL_CAPS:
            if entry_price <= upper:
                return cap
        return PENNY_NOTIONAL_CAPS[-1][1]

    def _size(
        self, score: float, entry_price: float, atr: float, kelly: float
    ) -> tuple[int, float, float]:
        """Compute (shares, stop, target) under all notional/loss caps."""
        direction = 1 if score > 0 else -1
        risk_per_share = atr * ATR_STOP_MULT
        max_by_notional = MAX_POSITION_NOTIONAL / entry_price
        max_by_loss = MAX_LOSS_PER_TRADE / risk_per_share
        max_by_kelly = MAX_POSITION_NOTIONAL * kelly / entry_price
        max_by_penny = self._penny_notional_cap(entry_price) / entry_price
        shares = max(
            1, int(min(max_by_notional, max_by_loss, max_by_kelly, max_by_penny))
        )
        stop = round(entry_price - direction * ATR_STOP_MULT * atr, 2)
        target = round(entry_price + direction * ATR_TARGET_MULT * atr, 2)
        return shares, stop, target

    def evaluate(
        self,
        score: float,
        confidence: float,
        entry_price: float,
        atr: float,
        open_positions: int,
    ) -> SizingResult:
        """Full risk evaluation. Returns sizing or rejection."""
        bad = self._invalid_inputs(score, entry_price, atr)
        if bad is not None:
            return SizingResult(reason=bad)
        win_prob = score_to_win_prob(score)
        kelly = kelly_fraction(win_prob) * KELLY_FRACTION
        if not math.isfinite(kelly) or kelly <= 0:
            return SizingResult(reason="Invalid Kelly (non-finite or zero)")
        direction = 1 if score > 0 else -1
        gate = ev_gate_should_enter(
            score, win_prob, self._realized, direction, confidence=confidence
        )
        ev = gate["ev"]
        if not gate["should_enter"]:
            return SizingResult(ev=ev, kelly=kelly, reason=gate["reason"])
        if open_positions >= MAX_CONCURRENT_POSITIONS:
            return SizingResult(reason=f"Max {MAX_CONCURRENT_POSITIONS} positions")
        shares, stop, target = self._size(score, entry_price, atr, kelly)
        return SizingResult(
            shares,
            stop_price=stop,
            target_price=target,
            ev=ev,
            kelly=kelly,
            risk_pass=True,
            reason="ok",
        )
