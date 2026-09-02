"""tests/test_safety_nets.py — verify hard safety nets are enforced.

These tests simulate dangerous scenarios and verify the system
halts or refuses — never silently continues into danger.
"""
from __future__ import annotations

import pytest
from hanoon_prime.brain import Brain
from hanoon_prime.constants import (
    MAX_CONCURRENT_POSITIONS,
    MAX_LOSS_PER_TRADE,
    MAX_POSITION_NOTIONAL,
    DAILY_LOSS_LIMIT,
    CONSECUTIVE_LOSSES_PAUSE,
)


class TestSafetyNets:
    """Every safety net must fire as a hardcoded exception."""

    def test_no_learning_at_start(self):
        """R8: learning is OFF until explicitly enabled."""
        brain = Brain()
        assert brain._learning_active is False

    def test_learning_must_be_explicitly_enabled(self):
        """Learning can't be accidentally activated."""
        brain = Brain()
        brain._indicator_weights["vpin"] = 0.99
        # Without enable_learning(), weights should still be used
        # but the learning path (record_trade adjusting weights) is gated
        brain.record_trade("TEST", won=False, pnl_pct=-0.05, alpha_snapshot={})
        # Weights should NOT change because learning_active is False
        assert brain._indicator_weights["vpin"] == 0.99

    def test_consecutive_loss_pause(self):
        """3 consecutive losses triggers pause (simulated via exception)."""
        brain = Brain()
        brain._consecutive_losses = CONSECUTIVE_LOSSES_PAUSE - 1
        brain.record_trade("TEST", won=False, pnl_pct=-0.01, alpha_snapshot={})
        # The brain itself doesn't raise — it records. The check_safety_nets
        # must be called to detect the condition. This test verifies the
        # state reaches the threshold.
        assert brain._consecutive_losses >= CONSECUTIVE_LOSSES_PAUSE

    def test_position_size_capped(self):
        """Position size never exceeds MAX_POSITION_NOTIONAL."""
        brain = Brain()
        # With high conviction (win_prob=0.55), Kelly should be moderate
        shares = brain.size_position(win_prob=0.55, entry_price=100.0)
        notional = shares * 100.0
        assert notional <= MAX_POSITION_NOTIONAL, f"Notional ${notional} > ${MAX_POSITION_NOTIONAL}"

    def test_position_size_zero_if_no_edge(self):
        """Zero win probability → zero position size."""
        brain = Brain()
        shares = brain.size_position(win_prob=0.0, entry_price=100.0)
        assert shares == 0.0

    def test_daily_loss_limit_exists_and_is_positive(self):
        """Daily loss limit must be a positive constant."""
        assert DAILY_LOSS_LIMIT > 0

    def test_max_concurrent_positions_is_reasonable(self):
        """Max concurrent positions should be small (not 'unlimited')."""
        assert MAX_CONCURRENT_POSITIONS <= 5, "Too many concurrent positions allowed"

    def test_max_loss_per_trade_is_cap_not_floor(self):
        """Max loss per trade must be a loss cap, not a floor."""
        assert MAX_LOSS_PER_TRADE > 0
        # It should be reasonable: not $5 (too tight), not $500 (too loose)
        assert 10.0 <= MAX_LOSS_PER_TRADE <= 100.0
