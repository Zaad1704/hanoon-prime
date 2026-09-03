"""hanoon_prime.hippocampus — learning system + safety nets.

IB is the SINGLE source of truth for positions, P&L, fills.
_daily_pnl is set from IB's reqPnL, not accumulated locally.

Learning rule (punishment-dominant):
  - LOSS: 2× penalty ∝ z_i × direction
  - WIN:  0.5× reward ∝ z_i × direction
  - Between trades: w_i *= 0.999 (geometric decay)
  - Clamped to [WEIGHT_MIN, WEIGHT_MAX] = [-2, +2]
"""
from __future__ import annotations

from typing import Any

from .cerebellum import INDICATOR_NAMES, compute_alpha
from .cortex import Cortex, Thought
from .edge import kelly_fraction
from .immune import (
    ATR_STOP_MULT,
    CONSECUTIVE_LOSSES_PAUSE,
    DAILY_LOSS_LIMIT,
    KELLY_FRACTION,
    LEARNING_RATE,
    MAX_CONCURRENT_POSITIONS,
    MAX_LOSS_PER_TRADE,
    MAX_POSITION_NOTIONAL,
    PAUSE_DURATION_MIN,
    PENALTY_SCALE,
    REWARD_SCALE,
    WEIGHT_DECAY,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from .types import Position


class Hippocampus:
    """JULI's brain: verdict orchestration, sizing, safety nets, learning."""

    def __init__(
        self,
        cortex: Cortex | None = None,
        weights: dict[str, float] | None = None,
        safety_enabled: bool = True,
    ) -> None:
        self._cortex: Cortex = cortex or Cortex(weights=weights)
        self._weights: dict[str, float] = dict(weights or self._cortex._weights)
        self._learning_active: bool = False
        self.safety_enabled: bool = safety_enabled
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._open_positions: dict[str, Position] = {}
        self._pause_bars: int = 0

    def deliberate_entry(
        self,
        close: Any,
        volume: Any,
        buy_volume: Any | None = None,
        bid_sizes: Any | None = None,
        ask_sizes: Any | None = None,
    ) -> Thought:
        """Full entry pipeline: cerebellum → cortex → Thought."""
        alpha = compute_alpha(
            close=close,
            volume=volume,
            buy_volume=buy_volume,
            bid_sizes=bid_sizes,
            ask_sizes=ask_sizes,
        )
        return self._cortex.evaluate(alpha)

    def size_position(self, win_prob: float, entry_price: float, atr: float) -> float:
        """Fractional Kelly (25%) capped by notional and loss limits."""
        if win_prob <= 0.0 or entry_price <= 0.0 or atr <= 0.0:
            return 0.0
        kelly = kelly_fraction(win_prob) * KELLY_FRACTION
        if kelly <= 0.0:
            return 0.0
        risk_per_share = atr * ATR_STOP_MULT
        if risk_per_share <= 0.0:
            return 0.0
        max_by_notional = MAX_POSITION_NOTIONAL / entry_price
        max_by_loss = MAX_LOSS_PER_TRADE / risk_per_share
        max_by_kelly = MAX_POSITION_NOTIONAL * kelly / entry_price
        shares = min(max_by_notional, max_by_loss, max_by_kelly)
        return float(max(1.0, round(shares)))

    def check_safety_nets(self) -> None:
        """Raise RuntimeError if any safety net is violated (R6)."""
        if not self.safety_enabled:
            return
        if self._daily_pnl < -DAILY_LOSS_LIMIT:
            raise RuntimeError(
                f"SAFETY NET: Daily loss ${self._daily_pnl:.2f} < -${DAILY_LOSS_LIMIT}"
            )
        if self._consecutive_losses >= CONSECUTIVE_LOSSES_PAUSE:
            raise RuntimeError(
                f"SAFETY NET: {self._consecutive_losses} consecutive losses"
            )
        if len(self._open_positions) > MAX_CONCURRENT_POSITIONS:
            raise RuntimeError(
                f"SAFETY NET: {len(self._open_positions)} open > {MAX_CONCURRENT_POSITIONS}"
            )

    def check_entry_allowed(self) -> bool:
        """Return True if new entries are allowed."""
        if self._pause_bars > 0:
            self._pause_bars -= 1
            return False
        try:
            self.check_safety_nets()
            return True
        except RuntimeError:
            self._pause_bars = PAUSE_DURATION_MIN
            self._consecutive_losses = 0
            return False

    def enable_learning(self) -> None:
        """Activate online weight gradient."""
        self._learning_active = True

    def record_trade(
        self,
        ticker: str,
        won: bool,
        pnl_pct: float,
        direction: int = 1,
        z_scores: dict[str, float] | None = None,
    ) -> None:
        """Record trade outcome from IB's fill data."""
        self._consecutive_losses = 0 if won else self._consecutive_losses + 1
        if not self._learning_active:
            return
        self._adapt_weights(z_scores or {}, won, direction)

    def _adapt_weights(
        self,
        z_scores: dict[str, float],
        won: bool,
        direction: int,
    ) -> None:
        """Asymmetric punishment. R8: ONLY weight-adaptation in codebase."""
        if direction == 0:
            return
        factor = REWARD_SCALE if won else -PENALTY_SCALE
        for name in INDICATOR_NAMES:
            z = z_scores.get(name, 0.0)
            delta = LEARNING_RATE * factor * z * direction
            self._weights[name] = max(
                WEIGHT_MIN, min(WEIGHT_MAX, self._weights[name] + delta)
            )
        for name in INDICATOR_NAMES:
            self._weights[name] *= WEIGHT_DECAY

    @property
    def learning_active(self) -> bool:
        """Whether the adaptive weight gradient is active."""
        return self._learning_active

    @property
    def indicator_weights(self) -> dict[str, float]:
        """Current indicator weights (copy)."""
        return dict(self._weights)

    @property
    def cortex(self) -> Cortex:
        """The Cortex instance used for verdict generation."""
        return self._cortex

    @property
    def daily_pnl(self) -> float:
        """Running P&L for the current trading day."""
        return self._daily_pnl

    @property
    def consecutive_losses(self) -> int:
        """Current streak of consecutive losing trades."""
        return self._consecutive_losses

    @property
    def open_count(self) -> int:
        """Number of currently open positions."""
        return len(self._open_positions)
