"""hanoon_prime.brain — JULI, the minimal trading brain.

Orchestrator ONLY: alpha → score → EV → think → verdict.
Also manages: sizing, exits, learning, journal, safety nets.
"""
from __future__ import annotations

from .alpha import compute_alpha
from .constants import (
    DAILY_LOSS_LIMIT,
    CONSECUTIVE_LOSSES_PAUSE,
    KELLY_FRACTION,
    MAX_CONCURRENT_POSITIONS,
    MAX_LOSS_PER_TRADE,
    MAX_POSITION_NOTIONAL,
    TARGET_R_R,
)
from .edge import compute_ev, kelly_fraction, score_to_win_prob
from .scoring import compute_score
from .thinker import Thought, deliberate

__all__ = ["Brain", "Thought"]


class Brain:
    """JULI — the minimal trading brain."""

    def __init__(self) -> None:
        self._open_positions: dict[str, dict] = {}
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        # Weights auto-calibrated by permutation test (see _calibrate.py).
        # institutional_flow gets highest weight (strongest pooled edge).
        from .constants import INDICATOR_WEIGHTS
        self._indicator_weights: dict[str, float] = dict(INDICATOR_WEIGHTS)
        self._learning_active: bool = False

    def deliberate_entry(self, close, volume, buy_volume, bid_sizes, ask_sizes) -> Thought:
        """Full entry pipeline: IB data → alpha → score → EV → verdict."""
        alpha = compute_alpha(
            close=close, volume=volume,
            buy_volume=buy_volume, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )
        score = self._score(alpha)
        win_prob = score_to_win_prob(score)
        ev = compute_ev(win_prob, reward_risk=TARGET_R_R)
        kelly = kelly_fraction(win_prob)
        return deliberate(score, alpha, win_prob, ev["gross_ev"], kelly)

    def _score(self, alpha: dict[str, float]) -> float:
        """Weighted average of normalized indicators."""
        if self._learning_active:
            return compute_score(alpha, self._indicator_weights)
        return compute_score(alpha)

    def size_position(self, win_prob: float, entry_price: float) -> float:
        """25% Kelly, capped by MAX_POSITION_NOTIONAL and MAX_LOSS_PER_TRADE."""
        kelly = kelly_fraction(win_prob) * KELLY_FRACTION
        if kelly <= 0:
            return 0.0
        max_notional = MAX_POSITION_NOTIONAL * kelly
        risk_per_share = entry_price * 0.03
        shares_by_notional = max_notional / entry_price
        shares_by_risk = MAX_LOSS_PER_TRADE / risk_per_share
        shares = min(shares_by_notional, shares_by_risk)
        if shares <= 0:
            return 0.0
        return round(shares, 4)

    def deliberate_exit(self, ticker, entry_price, high, low, close) -> str:
        """Trailing stop + fixed target. Sole EXIT verdict source."""
        pos = self._open_positions.get(ticker)
        if pos is None:
            return "HOLD"

        current = close[-1]
        direction = pos.get("direction", 1)
        peak = pos.get("peak_price", entry_price)

        if direction > 0:
            stop = peak * 0.979
            target = entry_price * 1.05
            if current <= stop or current >= target:
                return "EXIT"
        else:
            stop = peak * 1.021
            target = entry_price * 0.95
            if current >= stop or current <= target:
                return "EXIT"

        return "HOLD"

    def check_safety_nets(self) -> None:
        """Hard-stop checks. Raise RuntimeError to halt. No config bypass."""
        if self._daily_pnl < -DAILY_LOSS_LIMIT:
            raise RuntimeError(f"SAFETY NET: Daily loss ${self._daily_pnl:.2f} < -${DAILY_LOSS_LIMIT}")
        if self._consecutive_losses >= CONSECUTIVE_LOSSES_PAUSE:
            raise RuntimeError(f"SAFETY NET: {self._consecutive_losses} consecutive losses, pause")
        if len(self._open_positions) > MAX_CONCURRENT_POSITIONS:
            raise RuntimeError(f"SAFETY NET: {len(self._open_positions)} positions > {MAX_CONCURRENT_POSITIONS}")

    def enable_learning(self) -> None:
        """Activate the single online weight gradient."""
        self._learning_active = True

    def record_trade(self, ticker, won, pnl_pct, alpha_snapshot) -> None:
        """Record trade outcome. If learning, apply one gradient step."""
        from .constants import LEARNING_RATE, WEIGHT_FLOOR

        self._consecutive_losses = 0 if won else self._consecutive_losses + 1
        self._daily_pnl += pnl_pct

        if not self._learning_active:
            return

        direction = self._direction_from_alpha(alpha_snapshot)
        if direction == 0:
            return

        for name in self._indicator_weights:
            change = LEARNING_RATE * (-(1.2) if not won else 0.5)
            self._indicator_weights[name] = max(
                WEIGHT_FLOOR, min(0.50, self._indicator_weights[name] + change)
            )

        total = sum(self._indicator_weights.values())
        if total > 0:
            self._indicator_weights = {k: v / total for k, v in self._indicator_weights.items()}

    @staticmethod
    def _direction_from_alpha(alpha: dict[str, float]) -> int:
        net = sum([
            alpha.get("momentum", 0.0),
            alpha.get("orderbook_imbalance", 0.0),
            alpha.get("institutional_flow", 0.0) * 2.0,
            alpha.get("vwap_deviation", 0.0),
        ])
        if net > 0.05:
            return 1
        if net < -0.05:
            return -1
        return 0
