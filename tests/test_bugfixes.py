"""tests/test_bugfixes.py — Smoke tests for critical bug fixes.

Bug #1: _exec_decision must respect brain sizing result (threshold bypass).
Bug #2: Safety net must be enabled by default (MAX_CONCURRENT_POSITIONS).
Bug #3: Off-market entries must be blocked (US/Eastern market hours).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hanoon_prime.brain.risk import RiskEngine, SizingResult
from hanoon_prime.hippocampus import Hippocampus
from hanoon_prime.immune import MAX_CONCURRENT_POSITIONS
from hanoon_prime.monitor.sleep_manager import SleepManager, SleepState

# ── Bug #1: Entry threshold bypass ────────────────────────────────────


class TestBug1ThresholdBypass:
    """_exec_decision must skip entries when brain says sizing=0."""

    def _make_mixin(self):
        """Build a minimal BotCycleMixin-like object with mocked deps."""
        from hanoon_prime.ib_cycle import BotCycleMixin

        mixin = BotCycleMixin.__new__(BotCycleMixin)
        mixin.streamer = MagicMock()
        mixin.hippocampus = MagicMock()
        mixin.hippocampus.check_entry_allowed.return_value = True
        mixin.hippocampus._open_positions = {}
        mixin.executor = MagicMock()
        mixin.juli = MagicMock()
        mixin._sleep_mgr = MagicMock()
        mixin._sleep_mgr.get_state.return_value = SleepState(active=True)
        return mixin

    def test_skips_when_sizing_shares_zero(self):
        """Decision with SizingResult(shares=0) must NOT place bracket."""
        mixin = self._make_mixin()
        tk = MagicMock()
        tk.hasBidAsk = True
        tk.bid, tk.ask = 100.0, 101.0
        mixin.streamer.ticker_subs = {"TSLA": tk}

        dec = {
            "ticker": "TSLA",
            "direction": 1,
            "sizing": SizingResult(shares=0),
            "thought": SimpleNamespace(direction=1, score=0.02),
        }
        mixin._exec_decision(dec)
        mixin.executor.place_bracket.assert_not_called()

    def test_skips_when_sizing_none(self):
        """Decision with no sizing must NOT place bracket."""
        mixin = self._make_mixin()
        tk = MagicMock()
        tk.hasBidAsk = True
        tk.bid, tk.ask = 100.0, 101.0
        mixin.streamer.ticker_subs = {"TSLA": tk}

        dec = {
            "ticker": "TSLA",
            "direction": 1,
            "sizing": None,
            "thought": SimpleNamespace(direction=1, score=0.02),
        }
        mixin._exec_decision(dec)
        mixin.executor.place_bracket.assert_not_called()

    def test_places_bracket_when_sizing_valid(self):
        """Decision with valid sizing MUST place bracket."""
        mixin = self._make_mixin()
        tk = MagicMock()
        tk.hasBidAsk = True
        tk.bid, tk.ask = 100.0, 101.0
        mixin.streamer.ticker_subs = {"TSLA": tk}

        dec = {
            "ticker": "TSLA",
            "direction": 1,
            "sizing": SizingResult(shares=3, risk_pass=True),
            "thought": SimpleNamespace(direction=1, score=0.65),
        }
        mixin._exec_decision(dec)
        mixin.executor.place_bracket.assert_called_once()

    def test_risk_engine_rejects_below_threshold_score(self):
        """RiskEngine rejects tiny scores via EV gate (no 1-share trades)."""
        engine = RiskEngine()
        result = engine.evaluate(
            score=0.02,
            confidence=0.5,
            entry_price=100.0,
            atr=2.0,
            open_positions=0,
        )
        assert result.shares == 0
        assert result.risk_pass is False

    def test_risk_engine_rejects_at_max_positions(self):
        """RiskEngine rejects when at MAX_CONCURRENT_POSITIONS."""
        engine = RiskEngine()
        result = engine.evaluate(
            score=0.6,
            confidence=0.7,
            entry_price=100.0,
            atr=2.0,
            open_positions=MAX_CONCURRENT_POSITIONS,
        )
        assert result.shares == 0
        assert result.risk_pass is False


# ── Bug #2: Safety net enabled ────────────────────────────────────────


class TestBug2SafetyNetEnabled:
    """IBStreamingBot must start with safety_enabled=True."""

    def test_hippocampus_defaults_to_safety_enabled(self):
        """Hippocampus() defaults to safety_enabled=True."""
        brain = Hippocampus()
        assert brain.safety_enabled is True

    def test_safety_net_blocks_excess_positions(self):
        """check_safety_nets raises when positions exceed limit."""
        brain = Hippocampus(safety_enabled=True)
        brain._open_positions = {
            f"T{i}": MagicMock() for i in range(MAX_CONCURRENT_POSITIONS + 1)
        }
        with pytest.raises(RuntimeError, match="SAFETY NET"):
            brain.check_safety_nets()

    def test_safety_net_allows_within_limit(self):
        """check_safety_nets does NOT raise when within limit."""
        brain = Hippocampus(safety_enabled=True)
        brain._open_positions = {
            f"T{i}": MagicMock() for i in range(MAX_CONCURRENT_POSITIONS)
        }
        brain.check_safety_nets()  # should not raise

    def test_ib_adapter_has_hippocampus(self):
        """IBStreamingBot must create a Hippocampus instance."""
        import ast
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "hanoon_prime"
            / "ib_adapter.py"
        )
        content = src.read_text()
        assert "Hippocampus(safety_enabled=" in content


# ── Bug #3: Off-market guard ──────────────────────────────────────────


class TestBug3OffMarketGuard:
    """Entries must be blocked during off-market hours (US/Eastern)."""

    def test_sleep_manager_uses_zoneinfo(self):
        """SleepManager must use zoneinfo (not hardcoded UTC offset)."""
        import ast
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "hanoon_prime"
            / "monitor"
            / "sleep_manager.py"
        )
        content = src.read_text()
        assert "ZoneInfo" in content
        assert "America/New_York" in content

    def test_weekend_is_inactive(self):
        """SleepManager returns active=False on weekends."""
        mgr = SleepManager()
        # Mock a Saturday in US/Eastern
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
        # 2026-09-05 is a Saturday
        saturday = datetime(2026, 9, 5, 12, 0, tzinfo=et)
        with patch("hanoon_prime.monitor.sleep_manager.datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            state = mgr.get_state(ib_connected=True)
        assert state.active is False
        assert state.session == "weekend"

    def test_off_hours_is_inactive(self):
        """SleepManager returns active=False during overnight hours."""
        mgr = SleepManager()
        from datetime import datetime
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
        # 2026-09-08 is a Monday, 2:00 AM ET — off hours
        monday_2am = datetime(2026, 9, 8, 2, 0, tzinfo=et)
        with patch("hanoon_prime.monitor.sleep_manager.datetime") as mock_dt:
            mock_dt.now.return_value = monday_2am
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            state = mgr.get_state(ib_connected=True)
        assert state.active is False
        assert state.session == "overnight"

    def test_rth_is_active(self):
        """SleepManager returns active=True during RTH (10:00 AM ET weekday)."""
        mgr = SleepManager()
        from datetime import datetime
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
        # 2026-09-08 is a Monday, 10:00 AM ET — RTH
        monday_10am = datetime(2026, 9, 8, 10, 0, tzinfo=et)
        with patch("hanoon_prime.monitor.sleep_manager.datetime") as mock_dt:
            mock_dt.now.return_value = monday_10am
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            state = mgr.get_state(ib_connected=True)
        assert state.active is True
        assert state.session == "RTH"

    def test_ib_cycle_imports_sleep_manager(self):
        """ib_cycle.py must import SleepManager for market hours check."""
        import ast
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "hanoon_prime"
            / "ib_cycle.py"
        )
        content = src.read_text()
        assert "from .monitor.sleep_manager import SleepManager" in content
        assert "_SLEEP_MGR" in content

    def test_finish_cycle_skips_entries_when_market_closed(self):
        """_finish_cycle must skip entry execution when market is closed."""
        from hanoon_prime.ib_cycle import BotCycleMixin

        mixin = BotCycleMixin.__new__(BotCycleMixin)
        mixin.streamer = MagicMock()
        mixin.hippocampus = MagicMock()
        mixin.hippocampus._open_positions = {}
        mixin.hippocampus._daily_pnl = 0.0
        mixin.executor = MagicMock()
        mixin.executor.get_newly_closed_trades.return_value = []
        mixin.juli = MagicMock()
        mixin._closing = set()
        mixin._last_beat = 0.0
        mixin.ib = MagicMock()
        mixin.ib.pendingTickers.return_value = []
        mixin.journal = MagicMock()

        dec = {
            "ticker": "TSLA",
            "direction": 1,
            "sizing": SizingResult(shares=3, risk_pass=True),
            "thought": SimpleNamespace(direction=1, score=0.65),
        }
        # market_open=False should skip all entries
        mixin._finish_cycle([], [dec], 1.0, 0.0, None, market_open=False)
        mixin.executor.place_bracket.assert_not_called()
