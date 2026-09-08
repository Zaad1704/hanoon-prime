"""hanoon_prime.brain.risk — risk evaluation and position sizing.

Evaluates trade candidates through EV gate, Kelly sizing, and
portfolio heat limits. Calls edge.py for EV + Kelly, hippocampus
for base sizing, eyes for ATR.

Merge of rebuild's ev_gate.py + risk_manager.py + sizer.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from ..edge import kelly_fraction, score_to_win_prob
from ..immune import (
    KELLY_FRACTION,
    MAX_CONCURRENT_POSITIONS,
    MAX_LOSS_PER_TRADE,
    MAX_POSITION_NOTIONAL,
)
from .config import (
    MOMENTUM_FLAT_PENALTY,
    MOMENTUM_NEGATIVE_PENALTY_MAX,
    PENNY_NOTIONAL_CAPS,
    SPREAD_PENALTY_MAX,
    SPREAD_THRESHOLD_PCT,
    VWAP_CHASE_PCT,
    VWAP_CHASE_PENALTY_MAX,
)
from .horizons import params_for
from .realized_ev import RealizedStats, ev_gate_should_enter


def _ev_scale(gate: dict[str, Any]) -> float:
    """Advisory scale: full size on pass, 0.75 thin-positive, 0.5 negative."""
    if gate["should_enter"]:
        return 1.0
    return 0.75 if gate["ev"] > 0.0 else 0.5


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
    ev_scale: float = 1.0
    """Realized-EV confidence scale on shares (bounded [0.5, 1.0]).
    Advisory by construction — the realized-EV math never refuses an entry
    (nothing stands between the brain's pick and mechanical risk), it only
    sizes the conviction down when realized data distrusts the band."""
    ev_reason: str = ""
    """Realized-EV gate verdict string, for telemetry only."""
    quality_penalty: float = 0.0
    """Entry quality penalty applied as advisory score adjustment."""


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
        self,
        score: float,
        entry_price: float,
        atr: float,
        kelly: float,
        horizon: str = "scalp",
    ) -> tuple[int, float, float]:
        """Compute (shares, stop, target) under all notional/loss caps.

        Stop/target multipliers are per-horizon (scalp keeps the prime
        2×/6× ATR; longer horizons widen both, preserving 3:1 R:R).
        """
        direction = 1 if score > 0 else -1
        hp = params_for(horizon)
        stop_mult, target_mult = hp.atr_stop_mult, hp.atr_target_mult
        risk_per_share = atr * stop_mult
        max_by_notional = MAX_POSITION_NOTIONAL / entry_price
        max_by_loss = MAX_LOSS_PER_TRADE / risk_per_share
        max_by_kelly = MAX_POSITION_NOTIONAL * kelly / entry_price
        max_by_penny = self._penny_notional_cap(entry_price) / entry_price
        shares = max(
            1, int(min(max_by_notional, max_by_loss, max_by_kelly, max_by_penny))
        )
        stop = round(entry_price - direction * stop_mult * atr, 2)
        target = round(entry_price + direction * target_mult * atr, 2)
        return shares, stop, target

    def _preflight(
        self, score: float, confidence: float, entry_price: float, atr: float
    ) -> tuple[str | None, float, float, dict[str, Any]]:
        """Mechanical validity + Kelly + realized-EV advisory computation."""
        bad = self._invalid_inputs(score, entry_price, atr)
        if bad is not None:
            return bad, 0.0, 0.0, {}
        # Dynamic PRIOR_TOP widens the win-prob cap; cold/unbound ⇒ static (R5).
        pt = self._realized.dynamic_prior_top() if self._realized is not None else None
        win_prob = score_to_win_prob(score, prior_top=pt)
        kelly = kelly_fraction(win_prob) * KELLY_FRACTION
        if not math.isfinite(kelly) or kelly <= 0:
            return "Invalid Kelly (non-finite or zero)", 0.0, 0.0, {}
        direction = 1 if score > 0 else -1
        gate = ev_gate_should_enter(
            score, win_prob, self._realized, direction, confidence=confidence
        )
        return None, win_prob, kelly, gate

    def evaluate(
        self,
        score: float,
        confidence: float,
        entry_price: float,
        atr: float,
        open_positions: int,
        alpha: dict[str, float] | None = None,
        high: list[float] | None = None,
        low: list[float] | None = None,
        horizon: str = "scalp",
    ) -> SizingResult:
        """Mechanical limits + advisory realized-EV sizing (brain-first).

        The realized-EV gate NEVER refuses — it only scales sizing (bounded
        [0.5, 1.0]). Only mechanical limits (data validity, sub-rounding
        Kelly, position cap) return shares=0.

        Entry quality modifiers (VWAP chase, momentum, spread) apply as
        score penalties before final EV evaluation — advisory influence,
        not hard blocks (the thinker/cortex decides entry timing).
        """
        bad, win_prob, kelly, gate = self._preflight(
            score, confidence, entry_price, atr
        )
        if bad is not None:
            return SizingResult(reason=bad)
        ev, ev_scale = gate["ev"], _ev_scale(gate)
        if open_positions >= MAX_CONCURRENT_POSITIONS:
            return SizingResult(
                ev=ev, kelly=kelly, reason=f"Max {MAX_CONCURRENT_POSITIONS} positions"
            )
        # Entry quality modifiers (advisory score penalties)
        direction = 1 if score > 0 else -1
        quality_penalty = self._compute_entry_quality_penalty(
            alpha=alpha,
            direction=direction,
            high=high,
            low=low,
            vwap_dev=alpha.get("vwap_deviation", 0.0) if alpha else 0.0,
            momentum=alpha.get("momentum", 0.0) if alpha else 0.0,
        )
        adjusted_score = score - quality_penalty
        # Recompute EV with adjusted score (advisory adjustment)
        adjusted_win_prob = score_to_win_prob(
            adjusted_score, prior_top=gate.get("p_struct")
        )
        adjusted_kelly = kelly_fraction(adjusted_win_prob) * KELLY_FRACTION
        adjusted_ev = gate["p"] * gate["r"] - (1.0 - gate["p"])
        adjusted_ev_scale = _ev_scale(
            {**gate, "ev": adjusted_ev, "win_prob": adjusted_win_prob}
        )
        return self._compose_result(
            adjusted_score,
            entry_price,
            atr,
            adjusted_kelly,
            horizon,
            adjusted_ev,
            adjusted_ev_scale,
            gate["reason"],
            quality_penalty=quality_penalty,
        )

    def _compose_result(
        self,
        score: float,
        entry_price: float,
        atr: float,
        kelly: float,
        horizon: str,
        ev: float,
        ev_scale: float,
        ev_reason: str,
        quality_penalty: float = 0.0,
    ) -> SizingResult:
        """Mechanical size + sub-rounding guard + advisory EV scale."""
        shares, stop, target = self._size(score, entry_price, atr, kelly, horizon)
        # Sub-rounding guard (mechanical, not a gate): when Kelly sizing
        # buys less than one share the edge is below rounding noise.
        max_by_kelly = MAX_POSITION_NOTIONAL * kelly / entry_price
        if max_by_kelly < 1.0:
            reason = f"Sub-rounding edge: EV {ev:.3f} too low for 1 share"
            return SizingResult(
                ev=ev,
                kelly=kelly,
                reason=reason,
                quality_penalty=quality_penalty,
            )
        if ev_scale < 1.0:
            shares = max(1, int(shares * ev_scale))
        return SizingResult(
            shares,
            stop_price=stop,
            target_price=target,
            ev=ev,
            kelly=kelly,
            risk_pass=True,
            reason="ok",
            ev_scale=ev_scale,
            ev_reason=ev_reason,
            quality_penalty=quality_penalty,
        )

    def _compute_entry_quality_penalty(
        self,
        alpha: dict[str, float] | None,
        direction: int,
        high: list[float] | None,
        low: list[float] | None,
        vwap_dev: float = 0.0,
        momentum: float = 0.0,
    ) -> float:
        """Entry quality modifiers (advisory score penalties).

        These are SOFT penalties applied to the score before EV evaluation.
        They're advisory, not hard blocks — the thinker/cortex decides.

        1. VWAP CHASING: penalize if price is far above VWAP for longs
        2. MOMENTUM: penalize negative/zero momentum
        3. SPREAD: penalize wide spreads (slippage risk)

        Returns total quality penalty (positive value to subtract from score).
        """
        penalty = 0.0
        notes = []

        # MODIFIER 1: VWAP chasing (longs only)
        if direction > 0 and vwap_dev > VWAP_CHASE_PCT:
            pen = min(VWAP_CHASE_PENALTY_MAX, vwap_dev * 2.0)
            penalty += pen
            notes.append(f"vwap_chase({vwap_dev*100:.1f}%=-{pen:.3f})")

        # MODIFIER 2: Momentum quality
        if direction > 0:
            if momentum < 0.0:
                pen = min(MOMENTUM_NEGATIVE_PENALTY_MAX, abs(momentum) * 0.5)
                penalty += pen
                notes.append(f"neg_momentum({momentum:.3f}=-{pen:.3f})")
            elif momentum < 0.02:
                penalty += MOMENTUM_FLAT_PENALTY
                notes.append(
                    f"flat_momentum({momentum:.3f}=-{MOMENTUM_FLAT_PENALTY:.3f})"
                )

        # MODIFIER 3: Spread penalty
        if high and low and len(high) > 0 and len(low) > 0:
            last_high = float(high[-1])
            last_low = float(low[-1])
            if last_low > 0:
                spread_pct = (last_high - last_low) / last_low
                if spread_pct > SPREAD_THRESHOLD_PCT:
                    pen = min(
                        SPREAD_PENALTY_MAX, (spread_pct - SPREAD_THRESHOLD_PCT) * 2.0
                    )
                    penalty += pen
                    notes.append(f"wide_spread({spread_pct*100:.2f}%=-{pen:.3f})")

        # Log quality notes for telemetry
        if notes:
            from foundations.log import debug

            debug(
                f"Entry quality penalty: {notes} total={penalty:.3f}",
                context="brain_risk",
            )

        return penalty
