"""tests/test_safety_nets.py — verify hard safety nets are enforced.

These tests simulate dangerous scenarios and verify the system
halts or refuses — never silently continues into danger.
"""
from __future__ import annotations

import pytest

from hanoon_prime.hippocampus import Hippocampus
from hanoon_prime.immune import (
    CONSECUTIVE_LOSSES_PAUSE,
    DAILY_LOSS_LIMIT,
    MAX_CONCURRENT_POSITIONS,
    MAX_LOSS_PER_TRADE,
    MAX_POSITION_NOTIONAL,
)


class TestSafetyNets:
    """Every safety net must fire as a hardcoded exception."""

    def test_no_learning_at_start(self):
        """R8: learning is OFF until explicitly enabled."""
        brain = Hippocampus()
        assert brain.learning_active is False

    def test_learning_must_be_explicitly_enabled(self):
        """Learning can't be accidentally activated."""
        brain = Hippocampus()
        brain._weights["vpin"] = 0.99
        # Without enable_learning(), weights should still be used
        # but the learning path (record_trade adjusting weights) is gated
        brain.record_trade("TEST", won=False, pnl_pct=-0.05, direction=1, z_scores={})
        # Weights should NOT change because learning_active is False
        assert brain._weights["vpin"] == 0.99

    def test_consecutive_loss_pause(self):
        """3 consecutive losses triggers pause (simulated via exception)."""
        brain = Hippocampus()
        brain._consecutive_losses = CONSECUTIVE_LOSSES_PAUSE - 1
        brain.record_trade("TEST", won=False, pnl_pct=-0.01, direction=1, z_scores={})
        assert brain.consecutive_losses >= CONSECUTIVE_LOSSES_PAUSE

    def test_position_size_capped(self):
        """Position size never exceeds MAX_POSITION_NOTIONAL."""
        brain = Hippocampus()
        # With high conviction (win_prob=0.55), Kelly should be moderate
        shares = brain.size_position(win_prob=0.55, entry_price=100.0, atr=2.0)
        notional = shares * 100.0
        assert (
            notional <= MAX_POSITION_NOTIONAL
        ), f"Notional ${notional} > ${MAX_POSITION_NOTIONAL}"

    def test_position_size_zero_if_no_edge(self):
        """Zero win probability → zero position size."""
        brain = Hippocampus()
        shares = brain.size_position(win_prob=0.0, entry_price=100.0, atr=2.0)
        assert shares == 0.0

    def test_daily_loss_limit_exists_and_is_positive(self):
        """Daily loss limit must be a positive constant."""
        assert DAILY_LOSS_LIMIT > 0

    def test_max_concurrent_positions_is_reasonable(self):
        """Max concurrent positions should be small (not 'unlimited')."""
        assert MAX_CONCURRENT_POSITIONS <= 5

    def test_max_loss_per_trade_is_cap_not_floor(self):
        """Max loss per trade must be a loss cap, not a floor."""
        assert MAX_LOSS_PER_TRADE > 0
        assert 10.0 <= MAX_LOSS_PER_TRADE <= 100.0

    def test_safety_nets_trigger_on_violation(self):
        """check_safety_nets must raise when daily loss exceeds limit."""
        brain = Hippocampus()
        brain._daily_pnl = -DAILY_LOSS_LIMIT * 2
        with pytest.raises(RuntimeError, match="SAFETY NET"):
            brain.check_safety_nets()

    def test_consecutive_loss_pause_triggers(self):
        """check_safety_nets must raise when consecutive losses hit limit."""
        brain = Hippocampus()
        brain._consecutive_losses = CONSECUTIVE_LOSSES_PAUSE
        with pytest.raises(RuntimeError, match="SAFETY NET"):
            brain.check_safety_nets()

    def test_position_count_limit_triggers(self):
        """check_safety_nets must raise when too many positions open."""
        brain = Hippocampus()
        brain._open_positions = {
            f"TICK{i}": {"dummy": 1} for i in range(MAX_CONCURRENT_POSITIONS + 1)
        }
        with pytest.raises(RuntimeError, match="SAFETY NET"):
            brain.check_safety_nets()

    def test_atr_sized_position_respects_risk(self):
        """Position sized by ATR risk never exceeds MAX_LOSS_PER_TRADE."""
        brain = Hippocampus()
        # High conviction, low ATR → large position but risk still capped
        shares = brain.size_position(win_prob=0.55, entry_price=100.0, atr=1.0)
        risk = shares * 1.0 * 1.5  # 1.5×ATR stop
        assert risk <= MAX_LOSS_PER_TRADE


class TestSafetyNetToggle:
    """Runtime safety net toggle — webapp can enable/disable."""

    def test_default_is_enabled(self):
        """Hippocampus defaults to safety_enabled=True."""
        brain = Hippocampus()
        assert brain.safety_enabled is True

    def test_disabled_skips_daily_loss_check(self):
        """When disabled, daily loss violation must NOT raise."""
        brain = Hippocampus(safety_enabled=False)
        brain._daily_pnl = -DAILY_LOSS_LIMIT * 2
        brain.check_safety_nets()

    def test_disabled_skips_consecutive_loss_check(self):
        """When disabled, consecutive loss violation must NOT raise."""
        brain = Hippocampus(safety_enabled=False)
        brain._consecutive_losses = CONSECUTIVE_LOSSES_PAUSE
        brain.check_safety_nets()

    def test_disabled_skips_position_count_check(self):
        """When disabled, position count violation must NOT raise."""
        brain = Hippocampus(safety_enabled=False)
        brain._open_positions = {
            f"T{i}": 1 for i in range(MAX_CONCURRENT_POSITIONS + 1)
        }
        brain.check_safety_nets()

    def test_toggle_at_runtime(self):
        """Can toggle safety net on/off at runtime."""
        brain = Hippocampus(safety_enabled=False)
        brain._daily_pnl = -DAILY_LOSS_LIMIT * 2
        brain.check_safety_nets()  # no-op when disabled
        brain.safety_enabled = True
        with pytest.raises(RuntimeError, match="SAFETY NET"):
            brain.check_safety_nets()
        brain.safety_enabled = False
        brain.check_safety_nets()  # no-op again
