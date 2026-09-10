"""tests/test_bugfixes.py — Smoke tests for critical bug fixes.

Bug #1: _exec_decision must respect brain sizing result (threshold bypass).
Bug #2: Safety net must be enabled by default (MAX_CONCURRENT_POSITIONS).
Bug #3: Off-market entries must be blocked (US/Eastern market hours).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hanoon_prime.brain.policy.verdict import ENTER, Verdict
from hanoon_prime.brain.risk import RiskEngine, SizingResult
from hanoon_prime.hippocampus import Hippocampus
from hanoon_prime.ib_cycle import CycleMeta
from hanoon_prime.immune import MAX_CONCURRENT_POSITIONS
from hanoon_prime.monitor.sleep_manager import SleepManager

# ── Bug #1: Entry threshold bypass ────────────────────────────────────


class TestBug1ThresholdBypass:
    """_execute_verdict must honor the Verdict's sizing (threshold bypass)."""

    def _make_mixin(self):
        """Build a minimal BotCycleMixin-like object with mocked deps."""
        from hanoon_prime.ib_cycle import BotCycleMixin

        mixin = BotCycleMixin.__new__(BotCycleMixin)
        mixin.streamer = MagicMock()
        mixin.hippocampus = MagicMock()
        mixin.hippocampus._open_positions = {}
        mixin.executor = MagicMock()
        mixin.juli = MagicMock()
        mixin.monitor = MagicMock()
        return mixin

    def _enter_verdict(self, sizing):
        from hanoon_prime.brain.policy.verdict import ENTER, Verdict

        return Verdict(
            ticker="TSLA",
            action=ENTER,
            sizing=sizing,
            horizon="scalp",
            thought=SimpleNamespace(direction=1, score=0.65),
        )

    def test_skips_when_sizing_shares_zero(self):
        """Verdict with SizingResult(shares=0) must NOT place bracket."""
        mixin = self._make_mixin()
        tk = MagicMock()
        tk.hasBidAsk = True
        tk.bid, tk.ask = 100.0, 101.0
        mixin.streamer.ticker_subs = {"TSLA": tk}

        mixin._execute_verdict(self._enter_verdict(SizingResult(shares=0)))
        mixin.executor.place_bracket.assert_not_called()

    def test_skips_when_sizing_none(self):
        """Verdict with no sizing must NOT place bracket."""
        mixin = self._make_mixin()
        tk = MagicMock()
        tk.hasBidAsk = True
        tk.bid, tk.ask = 100.0, 101.0
        mixin.streamer.ticker_subs = {"TSLA": tk}

        mixin._execute_verdict(self._enter_verdict(None))
        mixin.executor.place_bracket.assert_not_called()
        mixin.juli.brain.register_position.assert_not_called()

    def test_places_bracket_and_notes_cooldown_when_sizing_valid(self):
        """Verdict with valid sizing MUST place bracket and stamp cooldown."""
        mixin = self._make_mixin()
        tk = MagicMock()
        tk.hasBidAsk = True
        tk.bid, tk.ask = 100.0, 101.0
        mixin.streamer.ticker_subs = {"TSLA": tk}

        v = self._enter_verdict(SizingResult(shares=3, risk_pass=True))
        mixin._execute_verdict(v)
        mixin.executor.place_bracket.assert_called_once()
        mixin.juli.brain.register_position.assert_not_called()  # deferred to fill
        mixin.juli.brain.note_entry.assert_called_once_with("TSLA")

    def test_fill_confirmed_registers_and_watches(self):
        """Only IB fill confirmation registers the exits-tracker + watchers."""
        mixin = self._make_mixin()
        mixin.executor._horizons = {"TSLA": "scalp"}
        mixin.executor._brackets = {}
        mixin._confirm_fill("TSLA", 100.5)
        mixin.juli.brain.register_position.assert_called_once_with(
            "TSLA", 100.5, horizon="scalp"
        )
        mixin.streamer.attach_exit_watcher.assert_called_once()

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
        from datetime import datetime
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
        # 2026-09-08 is a Monday, 10:00 AM ET — RTH (unified lowercase id)
        monday_10am = datetime(2026, 9, 8, 10, 0, tzinfo=et)
        with patch("hanoon_prime.monitor.sleep_manager.datetime") as mock_dt:
            mock_dt.now.return_value = monday_10am
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            state = mgr.get_state(ib_connected=True)
        assert state.active is True
        assert state.session == "rth"

    def test_ib_cycle_imports_sleep_manager(self):
        """ib_cycle.py must import SleepManager for market hours check."""
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
        mixin.monitor = MagicMock()
        mixin._closing = set()
        mixin._last_beat = 0.0
        mixin.ib = MagicMock()
        mixin.ib.pendingTickers.return_value = []
        mixin.journal = MagicMock()

        dec = Verdict(
            ticker="TSLA",
            action=ENTER,
            sizing=SizingResult(shares=3, risk_pass=True),
            thought=SimpleNamespace(direction=1, score=0.65),
        )
        # market_open=False should skip all entries
        mixin._finish_cycle(
            [], [dec], None, CycleMeta(poll=1.0, started=0.0, market_open=False)
        )
        mixin.executor.place_bracket.assert_not_called()

    def _finish_cycle_mixin(self):
        """Full mixin needed by the entry-gate tests (open-market path)."""
        from hanoon_prime.ib_cycle import BotCycleMixin

        mixin = BotCycleMixin.__new__(BotCycleMixin)
        tk = MagicMock()
        tk.hasBidAsk = True
        tk.bid, tk.ask = 100.0, 101.0
        mixin.streamer = MagicMock()
        mixin.streamer.ticker_subs = {"TSLA": tk}
        mixin.hippocampus = MagicMock()
        mixin.hippocampus._open_positions = {}
        mixin.hippocampus._daily_pnl = 0.0
        mixin.executor = MagicMock()
        mixin.executor.get_newly_closed_trades.return_value = []
        mixin.executor._horizons = {}
        mixin.juli = MagicMock()
        mixin.monitor = MagicMock()
        mixin.monitor.bar_feed_fresh.return_value = True
        mixin._closing = set()
        mixin._last_beat = 0.0
        mixin._exit_reasons = {}
        mixin.ib = MagicMock()
        mixin.ib.pendingTickers.return_value = []
        mixin.journal = MagicMock()
        return mixin

    def _enter_verdict(self):
        return Verdict(
            ticker="TSLA",
            action=ENTER,
            sizing=SizingResult(shares=3, risk_pass=True),
            thought=SimpleNamespace(direction=1, score=0.65),
        )

    def test_finish_cycle_suppresses_entries_on_stale_feed(self):
        """A dead bar feed must block ENTER verdicts (no stale-data trades)."""
        mixin = self._finish_cycle_mixin()
        mixin.monitor.bar_feed_fresh.return_value = False
        mixin._finish_cycle(
            [],
            [self._enter_verdict()],
            None,
            CycleMeta(poll=1.0, started=0.0, market_open=True),
        )
        mixin.executor.place_bracket.assert_not_called()

    def test_finish_cycle_enters_on_fresh_feed(self):
        """A live feed keeps ENTER verdicts executable (no over-blocking)."""
        mixin = self._finish_cycle_mixin()
        mixin._finish_cycle(
            [],
            [self._enter_verdict()],
            None,
            CycleMeta(poll=1.0, started=0.0, market_open=True),
        )
        mixin.executor.place_bracket.assert_called_once()

    def test_finish_cycle_still_executes_exits_on_stale_feed(self):
        """Protective exits must survive a stale feed (hard stops only)."""
        mixin = self._finish_cycle_mixin()
        mixin.monitor.bar_feed_fresh.return_value = False
        mixin._finish_cycle(
            [{"ticker": "NVD", "reason": "hard_stop_breach", "type": "hard_stop"}],
            [],
            None,
            CycleMeta(poll=1.0, started=0.0, market_open=True),
        )
        mixin.executor.close_position.assert_called_once_with("NVD", mixin.streamer)

    def test_snapshot_shaped_feed_no_array_truthiness(self):
        """Regression: FIX-2026-09-07-05.

        entry_bars/compute_alpha must survive REAL snapshots.

        ib_cycle._snapshot stores numpy arrays under ``*_arr`` keys. The
        ``snap.get("close_arr") or prices`` idiom raised ValueError
        (ambiguous truth) on every live entry evaluation — invisible in
        unit tests that passed list-shaped snapshots, and fatal at the
        open: zero decisions on every ticker.
        """
        import numpy as np

        from hanoon_prime.juli_feed import compute_alpha_from_snap, entry_bars

        n = 40
        snap = {
            "close_arr": np.linspace(100.0, 101.0, n),
            "high_arr": np.linspace(101.0, 102.0, n),
            "low_arr": np.linspace(99.0, 100.0, n),
            "volume_arr": np.full(n, 1000.0),
            "buy_volume_arr": np.full(n, 500.0),
            "bid_sizes_arr": np.full(n, 10.0),
            "ask_sizes_arr": np.full(n, 12.0),
            "prices": list(np.linspace(100.0, 101.0, n)),
        }
        prices = snap["prices"]
        bars = entry_bars(snap, prices, "trending_bullish")
        assert len(bars["close"]) == n
        assert bars["regime"] == "trending_bullish"
        alpha = compute_alpha_from_snap(snap)
        assert alpha, "alpha must compute from a real-shaped snapshot"
        # The _SRC mapping bug starved volume/depth indicators silently.
        volume_ish = [k for k in alpha if "vol" in k or "flow" in k or "depth" in k]
        assert volume_ish, f"volume/depth features missing from alpha: {list(alpha)}"


# ── Brain-aware order reconcile: orphaned _closing freeze ─────────────


class TestReconcileClosing:
    """_reconcile_closing must release _closing symbols whose close died.

    FIX-2026-09-08-01: a symbol lands in ``_closing`` when a flatten is
    placed; if IB cancels it with zero fills, the position stays open but
    every acting path skips _closing symbols forever (orphan freeze).
    """

    def _mixin(self):
        from hanoon_prime.ib_cycle import BotCycleMixin

        mixin = BotCycleMixin.__new__(BotCycleMixin)
        mixin.ib = MagicMock()
        mixin._closing = {"NVD", "WHLR"}
        mixin.hippocampus = MagicMock()
        mixin.hippocampus._open_positions = {}
        mixin.log = MagicMock()
        return mixin

    def test_keeps_live_close_order(self):
        """Position open + active close trade in flight → flag survives."""
        mixin = self._mixin()
        mixin.ib.positions.return_value = [
            SimpleNamespace(contract=SimpleNamespace(symbol="NVD"), position=5),
            SimpleNamespace(contract=SimpleNamespace(symbol="WHLR"), position=-3),
        ]
        mixin.ib.openTrades.return_value = [
            SimpleNamespace(
                contract=SimpleNamespace(symbol="NVD"), isDone=lambda: False
            ),
            SimpleNamespace(
                contract=SimpleNamespace(symbol="WHLR"), isDone=lambda: False
            ),
        ]
        mixin._reconcile_closing()
        assert mixin._closing == {"NVD", "WHLR"}

    def test_releases_flat_position(self):
        """No IB position at all + no in-memory position → stale flag dropped."""
        mixin = self._mixin()
        mixin.ib.positions.return_value = [
            SimpleNamespace(contract=SimpleNamespace(symbol="NVD"), position=0)
        ]
        mixin.ib.openTrades.return_value = []
        mixin._reconcile_closing()
        assert mixin._closing == set()

    def test_keeps_flat_flag_when_memory_position_stale(self):
        """IB flat but _open_positions stale → flag MUST survive.

        FIX-2026-09-08-02: sync reads the position as open at cycle start,
        the exit fill then lands, and reconcile reads IB as flat. Dropping
        the closing flag there let the exit evaluator re-fire and place a
        duplicate order that flipped the position (NVD +27 → short via
        double SELL). Release only once in-memory state agrees.
        """
        mixin = self._mixin()
        mixin._closing = {"NVD"}
        mixin.hippocampus._open_positions = {"NVD": object()}
        mixin.ib.positions.return_value = []
        mixin.ib.openTrades.return_value = []
        mixin._reconcile_closing()
        assert mixin._closing == {"NVD"}

    def test_releases_orphan_open_position_without_live_order(self):
        """Position open but close order died (no active trade) → released."""
        mixin = self._mixin()
        mixin._closing = {"NVD"}
        mixin.hippocampus._open_positions = {"NVD": object()}
        mixin.ib.positions.return_value = [
            SimpleNamespace(contract=SimpleNamespace(symbol="NVD"), position=5)
        ]
        mixin.ib.openTrades.return_value = []
        mixin._reconcile_closing()
        assert mixin._closing == set()


class TestSweepStaleOrders:
    """_sweep_stale_orders must never touch flatten / protection orders."""

    def _mixin(self):
        from hanoon_prime.ib_cycle import BotCycleMixin

        mixin = BotCycleMixin.__new__(BotCycleMixin)
        mixin.ib = MagicMock()
        mixin.ib.positions.return_value = []
        mixin.ib.openTrades.return_value = []
        mixin.ib.cancelOrder = MagicMock()
        mixin._closing = set()
        mixin.hippocampus = MagicMock()
        mixin.hippocampus._open_positions = {}
        mixin._order_placed_ts = {}
        return mixin

    def _trade(self, sym, oid, tif="DAY", parent_id=0, status="PreSubmitted"):
        order = SimpleNamespace(
            orderId=oid,
            parentId=parent_id,
            tif=tif,
            action="BUY",
            orderType="MKT",
            totalQuantity=10,
        )
        return SimpleNamespace(
            contract=SimpleNamespace(symbol=sym),
            order=order,
            orderStatus=SimpleNamespace(status=status),
            isDone=lambda: False,
        )

    def test_cancels_stale_entry_parent_without_position(self):
        """Stale DAY pending parent + no open position → swept."""
        mixin = self._mixin()
        trade = self._trade("AAPL", oid=1001)
        mixin.ib.openTrades.return_value = [trade]
        mixin._order_placed_ts = {1001: 0.0}  # placed long ago
        mixin._sweep_stale_orders()
        mixin.ib.cancelOrder.assert_called_once_with(trade.order)

    def test_never_cancels_flatten_order(self):
        """Symbol in _closing → flatten MKT must survive."""
        mixin = self._mixin()
        mixin._closing = {"WHLR"}
        mixin.ib.positions.return_value = [
            SimpleNamespace(contract=SimpleNamespace(symbol="WHLR"), position=-3)
        ]
        trade = self._trade("WHLR", oid=1002)
        mixin.ib.openTrades.return_value = [trade]
        mixin._order_placed_ts = {1002: 0.0}
        mixin._sweep_stale_orders()
        mixin.ib.cancelOrder.assert_not_called()

    def test_never_cancels_gtc_protection(self):
        """GTC STP/LMT protection must survive the sweep."""
        mixin = self._mixin()
        protect = SimpleNamespace(
            contract=SimpleNamespace(symbol="NVD"),
            order=SimpleNamespace(orderId=1003, parentId=0, tif="GTC"),
            orderStatus=SimpleNamespace(status="PreSubmitted"),
            isDone=lambda: False,
        )
        mixin.ib.openTrades.return_value = [protect]
        mixin._order_placed_ts = {1003: 0.0}
        mixin._sweep_stale_orders()
        mixin.ib.cancelOrder.assert_not_called()

    def test_never_cancels_child_order(self):
        """Bracket children (parentId set) are never swept."""
        mixin = self._mixin()
        trade = self._trade("AAPL", oid=1004, parent_id=9000)
        mixin.ib.openTrades.return_value = [trade]
        mixin._order_placed_ts = {1004: 0.0}
        mixin._sweep_stale_orders()
        mixin.ib.cancelOrder.assert_not_called()

    def test_never_cancels_stale_parent_with_open_position(self):
        """Open position needs its entry parent tracked → not swept."""
        mixin = self._mixin()
        mixin.hippocampus._open_positions = {"NVD": object()}
        trade = self._trade("NVD", oid=1005)
        mixin.ib.openTrades.return_value = [trade]
        mixin._order_placed_ts = {1005: 0.0}
        mixin._sweep_stale_orders()
        mixin.ib.cancelOrder.assert_not_called()


class TestAdoptedPositionExitRegistration:
    """_evaluate_exits must register adopted orphans and pass live direction."""

    def _make(self):
        from hanoon_prime.juli import JuliBrain

        juli = JuliBrain.__new__(JuliBrain)
        brain = MagicMock()
        brain.exits.is_registered.return_value = False
        brain.check_exit.return_value = SimpleNamespace(
            should_exit=True, reason="ladder TIER3", exit_type="TIER3"
        )
        juli.brain = brain
        return juli

    def test_registers_adopted_position(self):
        """Orphan adopted from IB gets registered so TIER3 fires."""
        juli = self._make()
        snap = {"last": 100.0}
        exits = juli._evaluate_exits(
            {"NVD"},
            lambda t: snap,
            set(),
            {"NVD": {"direction": 1, "entry_price": 98.0, "stop_price": 95.0}},
        )
        juli.brain.register_position.assert_called_once_with("NVD", 98.0)
        assert exits and exits[0]["type"] == "TIER3"

    def test_passes_real_direction_and_stop(self):
        """check_exit receives live short direction and stop price."""
        juli = self._make()
        exits = juli._evaluate_exits(
            {"WHLR"},
            lambda t: {"last": 3.12},
            set(),
            {"WHLR": {"direction": -1, "entry_price": 3.4, "stop_price": 3.6}},
        )
        _, kw = juli.brain.check_exit.call_args
        assert kw["direction"] == -1
        assert kw["stop_price"] == 3.6
        assert exits

    def test_skips_closing_symbols(self):
        """Symbols still in _closing are never re-evaluated for exits."""
        juli = self._make()
        exits = juli._evaluate_exits({"NVD"}, lambda t: {"last": 100.0}, {"NVD"}, {})
        juli.brain.check_exit.assert_not_called()
        assert exits == []
