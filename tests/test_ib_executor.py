"""tests/test_ib_executor.py — tests for IB-as-source-of-truth behavior.

Tests verify:
1. read_ib_positions reads ALL tracked tickers (not just _brackets)
2. _record_exit does NOT write journal entries (carbon copy principle)
3. cancel_all calls ib.cancelAllOrders()
4. sync_from_ib reads brackets from IB, cancels orphans, trails stops
5. _record_exit reads IB trade P&L directly (no local journal write)
6. _ping_ib verifies connection before sync
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hanoon_prime._guard import cancel_sell_legs, open_order_for, short_positions
from hanoon_prime._ib_sync import get_ib_pnl, read_ib_positions
from hanoon_prime.brain.policy.trading_policy import TRADING_CONFIG
from hanoon_prime.ib_executor import IBExecutor
from hanoon_prime.types import Position


def make_executor(tracked=None):
    """Create IBExecutor with mocked dependencies."""
    fake_ib = MagicMock()
    brain = MagicMock()
    journal = MagicMock()
    kwargs = {"tracked_tickers": set(tracked)} if tracked else {}
    return IBExecutor(fake_ib, brain, journal, **kwargs)


def make_pos(ticker="TSLA", entry_price=100.0, shares=10, direction=1, **kw):
    """Create a Position dataclass with sensible defaults."""
    defaults = dict(
        entry_idx=0,
        stop_price=95.0,
        target_price=110.0,
        peak_price=100.0,
        score=0.5,
        atr=2.0,
    )
    defaults.update(kw)
    return Position(
        ticker=ticker,
        entry_price=entry_price,
        shares=shares,
        direction=direction,
        **defaults,
    )


def _pos(symbol, position, avg_cost):
    """Helper: mock IB Position object."""
    m = MagicMock()
    m.contract.symbol = symbol
    m.position = position
    m.avgCost = avg_cost
    m.marketPrice = avg_cost * 1.01
    return m


def _make_trade(symbol="TSLA", parentId=0, children=None, contract_sym=None):
    """Helper: build a mock trade with parent order and children."""
    parent_order = MagicMock()
    parent_order.parentId = parentId
    parent_order.children = children or []
    trade = MagicMock()
    trade.order = parent_order
    trade.isDone.return_value = False
    trade.contract = MagicMock()
    trade.contract.symbol = contract_sym or symbol
    return trade


# ---------------------------------------------------------------------------
# TestReadIbPositions — tests the standalone read_ib_positions function
# ---------------------------------------------------------------------------


class TestReadIbPositions:
    """read_ib_positions reads ALL tracked tickers from IB (not just brackets)."""

    def test_reads_all_tracked_tickers(self):
        fake_ib = MagicMock()
        fake_ib.positions.return_value = [
            _pos("AAPL", 10, 100.0),
            _pos("MSFT", -5, 200.0),
            _pos("NVDA", 3, 150.0),
        ]
        positions = read_ib_positions(fake_ib, {"AAPL", "MSFT", "NVDA"}, {})
        assert set(positions.keys()) == {"AAPL", "MSFT", "NVDA"}

    def test_ignores_untracked_positions(self):
        fake_ib = MagicMock()
        fake_ib.positions.return_value = [
            _pos("AAPL", 10, 100.0),
            _pos("GOOGL", 5, 200.0),
        ]
        positions = read_ib_positions(fake_ib, {"AAPL"}, {})
        assert set(positions.keys()) == {"AAPL"}

    def test_direction_and_shares(self):
        fake_ib = MagicMock()
        fake_ib.positions.return_value = [
            _pos("AAPL", 10, 100.0),
            _pos("MSFT", -5, 200.0),
        ]
        positions = read_ib_positions(fake_ib, {"AAPL", "MSFT"}, {})
        assert positions["AAPL"].direction == 1
        assert positions["MSFT"].direction == -1
        assert positions["AAPL"].shares == 10
        assert positions["MSFT"].shares == 5

    def test_uses_bracket_levels_when_available(self):
        fake_ib = MagicMock()
        fake_ib.positions.return_value = [_pos("AAPL", 10, 100.0)]
        positions = read_ib_positions(fake_ib, {"AAPL"}, {"AAPL": (95.0, 110.0)})
        assert positions["AAPL"].stop_price == 95.0
        assert positions["AAPL"].target_price == 110.0

    def test_zero_brackets_when_no_bracket_info(self):
        fake_ib = MagicMock()
        fake_ib.positions.return_value = [_pos("AAPL", 10, 100.0)]
        positions = read_ib_positions(fake_ib, {"AAPL"}, {})
        assert positions["AAPL"].stop_price == 0.0
        assert positions["AAPL"].target_price == 0.0

    def test_handles_ib_error_gracefully(self):
        fake_ib = MagicMock()
        fake_ib.positions.side_effect = RuntimeError("IB down")
        assert read_ib_positions(fake_ib, {"AAPL"}, {}) == {}


# ---------------------------------------------------------------------------
# TestRecordExit — tests _record_exit on IBExecutor
# ---------------------------------------------------------------------------


class TestRecordExit:
    """_record_exit must NOT write journal entries — carbon copy principle."""

    def test_uses_ib_trade_pnl(self):
        """When IB has trade P&L, it's used (converted to pct)."""
        exc = make_executor(tracked={"TSLA"})
        exc.brain._open_positions = {
            "TSLA": make_pos(direction=1, shares=10, entry_price=100.0)
        }
        ib_trade = MagicMock()
        ib_trade.contract.symbol = "TSLA"
        ib_trade.isDone.return_value = True
        ib_trade.pnl = 500.0
        ib_trade.order.auxPrice = 100.0
        ib_trade.order.lmtPrice = None
        exc.ib.trades.return_value = [ib_trade]
        streamer = MagicMock()
        streamer.get_last_price.return_value = 110.0
        exc._record_exit("TSLA", streamer)
        exc.brain.record_trade.assert_called_once()
        kwargs = exc.brain.record_trade.call_args.kwargs
        assert kwargs["pnl_pct"] == pytest.approx(500.0)

    def test_returns_zero_pnl_when_no_ib_trades(self):
        """When IB has no completed trades, pnl defaults to 0.0."""
        exc = make_executor(tracked={"TSLA"})
        exc.brain._open_positions = {
            "TSLA": make_pos(direction=1, shares=10, entry_price=100.0)
        }
        exc.ib.trades.return_value = []
        streamer = MagicMock()
        streamer.get_last_price.return_value = 110.0
        exc._record_exit("TSLA", streamer)
        exc.brain.record_trade.assert_called_once()
        kwargs = exc.brain.record_trade.call_args.kwargs
        assert kwargs["pnl_pct"] == pytest.approx(0.0)

    def test_returns_zero_pnl_on_ib_error(self):
        """When IB errors, pnl defaults to 0.0 — IB is source of truth."""
        exc = make_executor(tracked={"TSLA"})
        exc.brain._open_positions = {
            "TSLA": make_pos(direction=1, shares=10, entry_price=100.0)
        }
        exc.ib.trades.side_effect = RuntimeError("IB down")
        streamer = MagicMock()
        streamer.get_last_price.return_value = 105.0
        exc._record_exit("TSLA", streamer)
        exc.brain.record_trade.assert_called_once()
        kwargs = exc.brain.record_trade.call_args.kwargs
        assert kwargs["pnl_pct"] == pytest.approx(0.0)

    def test_no_exit_journal_entry(self):
        """_record_exit must NOT write journal entries — exits visible via snapshot."""
        exc = make_executor(tracked={"TSLA"})
        exc.brain._open_positions = {
            "TSLA": make_pos(direction=1, shares=10, entry_price=100.0)
        }
        streamer = MagicMock()
        streamer.get_last_price.return_value = 110.0
        exc.ib.trades.return_value = []
        exc._record_exit("TSLA", streamer)
        for call in exc.journal.append.call_args_list:
            args = call[0]
            if args and isinstance(args[0], dict):
                assert args[0].get("event") != "exit"


# ---------------------------------------------------------------------------
# TestNotifyOpenFills — fill-confirmed entry notifications
# ---------------------------------------------------------------------------


class TestNotifyOpenFills:
    """_notify_open_fills fires entry notifications only on IB fill confirm."""

    def test_fires_when_position_appears(self):
        exc = make_executor(tracked={"TSLA"})
        exc._pending_parent.add("TSLA")
        exc._brackets["TSLA"] = (95.0, 110.0)
        pos = make_pos(direction=1, shares=10, entry_price=100.0)
        with patch("hanoon_prime.ib_executor.trade_opened") as fn:
            exc._notify_open_fills({"TSLA": pos})
        fn.assert_called_once()
        _, side, qty, price, levels = fn.call_args[0]
        assert side == "BUY"
        assert qty == 10
        assert price == 100.0
        assert levels.stop == 95.0
        assert levels.target == 110.0
        assert "TSLA" not in exc._pending_parent

    def test_ignores_still_pending(self):
        exc = make_executor(tracked={"TSLA"})
        exc._pending_parent.add("TSLA")
        exc._notify_open_fills({})
        assert "TSLA" in exc._pending_parent

    def test_fill_hook_fires_once_position_fills(self):
        exc = make_executor(tracked={"TSLA"})
        exc._pending_parent.add("TSLA")
        seen: list[tuple[str, float]] = []
        exc.on_fill_confirmed = lambda sym, price: seen.append((sym, price))
        pos = make_pos(direction=1, shares=10, entry_price=100.0)
        with patch("hanoon_prime.ib_executor.trade_opened"):
            exc._notify_open_fills({"TSLA": pos})
        assert seen == [("TSLA", 100.0)]


# ---------------------------------------------------------------------------
# TestRecordExitNotify — all closes notify, including reconciled
# ---------------------------------------------------------------------------


class TestRecordExitNotify:
    """_record_exit notifies every close, including synthetic/reconciled."""

    def test_synthetic_close_notifies(self):
        exc = make_executor(tracked={"TSLA"})
        exc.brain._open_positions = {
            "TSLA": make_pos(direction=1, shares=10, entry_price=100.0)
        }
        exc._synthetic.add("TSLA")
        exc.ib.trades.return_value = []
        streamer = MagicMock()
        streamer.get_last_price.return_value = 110.0
        with patch("hanoon_prime.ib_executor.trade_closed") as fn:
            exc._record_exit("TSLA", streamer)
        fn.assert_called_once()
        assert fn.call_args.kwargs["reason"] == "reconciled"
        exc.brain.record_trade.assert_not_called()

    def test_real_close_notifies_plain(self):
        exc = make_executor(tracked={"TSLA"})
        exc.brain._open_positions = {
            "TSLA": make_pos(direction=1, shares=10, entry_price=100.0)
        }
        exc.ib.trades.return_value = []
        streamer = MagicMock()
        streamer.get_last_price.return_value = 110.0
        with patch("hanoon_prime.ib_executor.trade_closed") as fn:
            exc._record_exit("TSLA", streamer)
        fn.assert_called_once()
        assert fn.call_args.kwargs.get("reason", "") == ""
        exc.brain.record_trade.assert_called_once()


# ---------------------------------------------------------------------------
# TestCancelAll
# ---------------------------------------------------------------------------


class TestCancelAll:
    """cancel_all must cancel orders in IB — IB is the source of truth."""

    def test_cancel_all_calls_ib(self):
        exc = make_executor()
        exc.cancel_all()
        exc.ib.cancelAllOrders.assert_called_once()
        assert len(exc._brackets) == 0

    def test_cancel_all_survives_error(self):
        exc = make_executor()
        exc.ib.cancelAllOrders.side_effect = RuntimeError("IB down")
        exc.cancel_all()
        assert len(exc._brackets) == 0


# ---------------------------------------------------------------------------
# TestPingIb
# ---------------------------------------------------------------------------


class TestPingIb:
    """_ping_ib verifies IB connection is alive before syncing."""

    def test_returns_true_when_connected(self):
        exc = make_executor()
        exc.ib.isConnected.return_value = True
        assert exc._ping_ib() is True

    def test_returns_false_when_disconnected(self):
        exc = make_executor()
        exc.ib.isConnected.return_value = False
        assert exc._ping_ib() is False

    def test_returns_false_on_exception(self):
        exc = make_executor()
        exc.ib.isConnected.side_effect = RuntimeError("socket error")
        assert exc._ping_ib() is False

    def test_sync_skips_when_disconnected(self):
        exc = make_executor(tracked={"TSLA"})
        exc.ib.isConnected.return_value = False
        exc.brain._open_positions = {"TSLA": make_pos()}
        exc.sync_from_ib(MagicMock())
        # Should not crash — just returns early


# ---------------------------------------------------------------------------
# TestSyncFromIb — tests the sync_from_ib pipeline
# ---------------------------------------------------------------------------


class TestSyncFromIb:
    """sync_from_ib reads brackets, cancels orphans, trails stops."""

    def test_sync_reads_bracket_levels(self):
        exc = make_executor(tracked={"TSLA"})
        child_sl = MagicMock()
        child_sl.auxPrice = 95.0
        child_sl.lmtPrice = 0.0
        child_tp = MagicMock()
        child_tp.auxPrice = 0.0
        child_tp.lmtPrice = 110.0
        trade = _make_trade("TSLA", 0, [child_sl, child_tp])
        exc.ib.trades.return_value = [trade]
        exc.ib.positions.return_value = [_pos("TSLA", 10, 100.0)]
        streamer = MagicMock()
        streamer.get_last_price.return_value = 105.0
        streamer.buffer_atr.return_value = 2.0
        streamer.contracts = {"TSLA": MagicMock()}
        exc.sync_from_ib(streamer)
        assert exc._brackets["TSLA"][0] == 95.0
        assert exc._brackets["TSLA"][1] == 110.0

    def test_sync_skips_untracked_tickers(self):
        exc = make_executor(tracked={"TSLA"})
        trade = _make_trade("NVDA", 0, [])
        exc.ib.trades.return_value = [trade]
        exc.ib.positions.return_value = []
        streamer = MagicMock()
        streamer.get_last_price.return_value = 100.0
        streamer.buffer_atr.return_value = 1.0
        streamer.contracts = {}
        exc.sync_from_ib(streamer)
        assert "NVDA" not in exc._brackets

    def test_sync_survives_error(self):
        exc = make_executor(tracked={"TSLA"})
        exc.ib.trades.side_effect = RuntimeError("IB down")
        streamer = MagicMock()
        exc.sync_from_ib(streamer)  # should not raise


# ---------------------------------------------------------------------------
# TestGetIbPnl — tests the standalone get_ib_pnl function
# ---------------------------------------------------------------------------


class TestGetIbPnl:
    """get_ib_pnl reads P&L from IB fills — source of truth."""

    def test_uses_ib_trade_pnl(self):
        fake_ib = MagicMock()
        trade = MagicMock()
        trade.contract.symbol = "TSLA"
        trade.isDone.return_value = True
        trade.pnl = 500.0
        fake_ib.trades.return_value = [trade]
        pos = make_pos(direction=1, shares=10, entry_price=100.0)
        pnl = get_ib_pnl(fake_ib, "TSLA", pos)
        assert pnl == 500.0

    def test_falls_back_to_fill_price(self):
        fake_ib = MagicMock()
        fake_ib.trades.return_value = []
        pos = make_pos(direction=1, shares=10, entry_price=100.0)
        pnl = get_ib_pnl(fake_ib, "TSLA", pos)
        assert pnl == 0.0

    def test_handles_ib_error(self):
        fake_ib = MagicMock()
        fake_ib.trades.side_effect = RuntimeError("IB down")
        pos = make_pos(direction=1, shares=10, entry_price=100.0)
        pnl = get_ib_pnl(fake_ib, "TSLA", pos)
        assert pnl == 0.0


# ---------------------------------------------------------------------------
# TestOutsideRth — extended-hours trading (pre-market / post-market)
# ---------------------------------------------------------------------------
# R6: ALLOW_EXTENDED_HOURS enables outsideRth=True on ALL order types so
# that JULI can enter, exit, and protect positions during pre-market
# (4:00-9:30 AM ET) and post-market (4:00-8:00 PM ET) sessions.


class TestOutsideRth:
    """All order placement paths must set outsideRth=True for extended hours."""

    def test_place_bracket_sets_outsideRth(self):
        """place_bracket must set outsideRth=True on all bracket order legs."""
        exc = make_executor(tracked={"TSLA"})
        exc.brain.size_position.return_value = 100
        exc.brain.score_to_win_prob = MagicMock(return_value=0.7)
        thought = MagicMock()
        thought.score = 0.7
        thought.direction = 1
        streamer = MagicMock()
        streamer.buffer_atr.return_value = 2.0
        streamer.contracts = {"TSLA": MagicMock()}
        placed_orders = []
        exc.ib.bracketOrder.return_value = [MagicMock(), MagicMock(), MagicMock()]
        exc.ib.placeOrder.side_effect = lambda c, o: placed_orders.append(o)
        exc.place_bracket("TSLA", thought, 150.0, streamer)
        assert len(placed_orders) >= 1
        for order in placed_orders:
            assert order.outsideRth is True
            assert order.tif == "DAY"

    def test_close_position_sets_outsideRth(self):
        """close_position must create a MarketOrder with outsideRth=True."""
        from hanoon_prime.immune import ALLOW_EXTENDED_HOURS

        assert ALLOW_EXTENDED_HOURS is True
        exc = make_executor(tracked={"TSLA"})
        exc.brain._open_positions = {
            "TSLA": make_pos(direction=1, shares=10, entry_price=100.0)
        }
        streamer = MagicMock()
        streamer.contracts = {"TSLA": MagicMock()}
        # close_position routes orders through the ib_compat `_ib` shim
        # (not a deferred `from ib_insync import …` lookup), so patch _ib
        # directly — same idiom as TestPlaceOca patching _protect._ib.
        mock_ib_mod = MagicMock()
        with patch("hanoon_prime.ib_executor._ib", mock_ib_mod):
            exc.close_position("TSLA", streamer)
        mock_ib_mod.MarketOrder.assert_called_once()
        kwargs = mock_ib_mod.MarketOrder.call_args.kwargs
        assert kwargs["outsideRth"] is True
        assert kwargs["tif"] == "DAY"


# ---------------------------------------------------------------------------
# TestPlaceOca — tests _place_oca sets outsideRth on OCA legs
# ---------------------------------------------------------------------------


class TestPlaceOca:
    """_place_oca must set outsideRth=True on both STP and LMT legs."""

    def test_place_oca_sets_outsideRth(self):
        """_place_oca must pass outsideRth=True to Order constructor."""
        from hanoon_prime.immune import ALLOW_EXTENDED_HOURS

        assert ALLOW_EXTENDED_HOURS is True
        fake_ib = MagicMock()
        contract = MagicMock()
        placed = []
        fake_ib.placeOrder.side_effect = lambda c, o: placed.append(o)

        # Make mock Order() create objects whose .outsideRth reflects the kwarg
        def _make_order(**kw):
            m = MagicMock()
            m.outsideRth = kw.get("outsideRth")
            return m

        mock_order_cls = MagicMock(side_effect=_make_order)
        with patch("hanoon_prime._protect._ib", MagicMock(Order=mock_order_cls)):
            from hanoon_prime._protect import _place_oca
            from hanoon_prime.types import BracketOrder

            _place_oca(
                fake_ib,
                contract,
                BracketOrder("SELL", 100, 95.0, 110.0, "JULI_TSLA"),
            )
        assert len(placed) == 2
        for order in placed:
            assert order.outsideRth is True


# ---------------------------------------------------------------------------
# Short policy: SELL-open hard-block + IB account context on close notify
# ---------------------------------------------------------------------------


class TestShortPolicy:
    """direction_mode=long_only must block SELL opens at the executor level."""

    def test_sell_thought_blocked_when_long_only(self):
        from types import SimpleNamespace

        from hanoon_prime.brain.policy.trading_policy import TRADING_CONFIG

        mode = TRADING_CONFIG.direction_mode
        try:
            TRADING_CONFIG.direction_mode = "long_only"
            exc = make_executor()
            streamer = MagicMock()
            streamer.buffer_atr.return_value = 2.0
            streamer.contracts = {"TSLA": MagicMock()}
            exc.brain.size_position.return_value = 10
            thought = SimpleNamespace(direction=-1, score=0.5, confidence=0.5)
            exc.place_bracket("TSLA", thought, 150.0, streamer)
            exc.ib.bracketOrder.assert_not_called()
            exc.ib.placeOrder.assert_not_called()
        finally:
            TRADING_CONFIG.direction_mode = mode

    def test_sell_thought_allowed_when_both(self):
        from types import SimpleNamespace

        from hanoon_prime.brain.policy.trading_policy import TRADING_CONFIG

        mode = TRADING_CONFIG.direction_mode
        try:
            TRADING_CONFIG.direction_mode = "both"
            exc = make_executor()
            streamer = MagicMock()
            streamer.buffer_atr.return_value = 2.0
            streamer.contracts = {"TSLA": MagicMock()}
            exc.brain.size_position.return_value = 10
            exc.ib.bracketOrder.return_value = [MagicMock(), MagicMock(), MagicMock()]
            thought = SimpleNamespace(direction=-1, score=0.5, confidence=0.5)
            exc.place_bracket("TSLA", thought, 150.0, streamer)
            exc.ib.bracketOrder.assert_called_once()
        finally:
            TRADING_CONFIG.direction_mode = mode


class TestCloseSummary:
    """_close_summary renders IB account + realized win rate context."""

    def test_close_summary_includes_ib_context(self):
        exc = make_executor()
        exc._account_feed = {"equity": 242783.0, "daily_pnl": -554.0}
        exc._winrate_provider = lambda: (0.665, 200)
        summary = exc._close_summary()
        assert "Account $242,783" in summary
        assert "IB day -554.00" in summary
        assert "JULI WR 66.5% (n=200)" in summary

    def test_close_summary_empty_without_context(self):
        exc = make_executor()
        assert exc._close_summary() == ""


# ---------------------------------------------------------------------------
# TestShortPositions — netting guard helper: negative tracked positions
# ---------------------------------------------------------------------------


class TestShortPositions:
    """short_positions exposes only negative net positions for tracked tickers."""

    def test_returns_negative_tracked_only(self):
        fake_ib = MagicMock()
        fake_ib.positions.return_value = [
            _pos("OLB", -5, 0.3),
            _pos("SOXS", -2, 42.0),
            _pos("NOK", 100, 1.0),
        ]
        shorts = short_positions(fake_ib, {"OLB", "SOXS", "NOK"})
        assert shorts == {"OLB": 5.0, "SOXS": 2.0}

    def test_ignores_untracked_and_longs(self):
        fake_ib = MagicMock()
        fake_ib.positions.return_value = [
            _pos("OLB", -5, 0.3),
            _pos("NVDA", -10, 100.0),
            _pos("TSLA", 10, 100.0),
        ]
        shorts = short_positions(fake_ib, {"OLB", "TSLA"})
        assert shorts == {"OLB": 5.0}
        assert "NVDA" not in shorts

    def test_survives_error(self):
        fake_ib = MagicMock()
        fake_ib.positions.side_effect = RuntimeError("IB down")
        assert short_positions(fake_ib, {"OLB"}) == {}


# ---------------------------------------------------------------------------
# TestCancelSellLegs — cancel stale SELL legs without touching BUY orders
# ---------------------------------------------------------------------------


def _open_trade(symbol, action, done=False, active=True):
    """Helper: mock open Trade with order action + status."""
    m = MagicMock()
    order = MagicMock()
    order.action = action
    order.ocaGroup = f"JULI_{symbol}"
    m.order = order
    m.contract.symbol = symbol
    m.isDone.return_value = done
    return m


class TestCancelSellLegs:
    """cancel_sell_legs removes live SELL legs only."""

    def test_cancels_sell_legs_only(self):
        fake_ib = MagicMock()
        done_sell = _open_trade("OLB", "SELL", done=True)
        fake_ib.openTrades.return_value = [
            _open_trade("OLB", "SELL"),
            _open_trade("OLB", "BUY"),
            _open_trade("TSLA", "SELL"),
            done_sell,
        ]
        n = cancel_sell_legs(fake_ib, "OLB")
        assert n == 1
        cancelled_actions = [
            c.args[0].action for c in fake_ib.cancelOrder.call_args_list
        ]
        assert cancelled_actions == ["SELL"]
        assert done_sell.order not in [
            c.args[0] for c in fake_ib.cancelOrder.call_args_list
        ]

    def test_returns_zero_when_none(self):
        fake_ib = MagicMock()
        fake_ib.openTrades.return_value = [_open_trade("OLB", "BUY")]
        assert cancel_sell_legs(fake_ib, "OLB") == 0

    def test_survives_error(self):
        fake_ib = MagicMock()
        fake_ib.openTrades.side_effect = RuntimeError("IB down")
        assert cancel_sell_legs(fake_ib, "OLB") == 0


# ---------------------------------------------------------------------------
# TestNettingGuard — executor flattens accidental shorts in long_only
# ---------------------------------------------------------------------------


class TestNettingGuard:
    """_guard_netting buys back accidental shorts while long_only."""

    def test_flattens_short_when_long_only(self, monkeypatch):
        monkeypatch.setattr(TRADING_CONFIG, "direction_mode", "long_only")
        exc = make_executor(tracked={"OLB", "SOXS"})
        exc.ib.positions.return_value = [_pos("OLB", -5, 0.3)]
        exc.ib.openTrades.return_value = []
        streamer = MagicMock()
        streamer.contracts = {"OLB": MagicMock(), "SOXS": MagicMock()}
        exc._guard_netting(streamer, set())
        assert exc.ib.placeOrder.call_count == 1
        args, kwargs = exc.ib.placeOrder.call_args
        order = args[1]
        assert order.action == "BUY"
        assert int(order.totalQuantity) == 5
        assert order.tif == "DAY"
        assert order.outsideRth is True
        assert exc._flattening == {"OLB": 5}

    def test_skips_when_both_directions(self, monkeypatch):
        monkeypatch.setattr(TRADING_CONFIG, "direction_mode", "both")
        exc = make_executor(tracked={"OLB"})
        exc.ib.positions.return_value = [_pos("OLB", -5, 0.3)]
        streamer = MagicMock()
        streamer.contracts = {"OLB": MagicMock()}
        exc._guard_netting(streamer, set())
        exc.ib.placeOrder.assert_not_called()

    def test_skips_when_buy_already_in_flight(self, monkeypatch):
        monkeypatch.setattr(TRADING_CONFIG, "direction_mode", "long_only")
        exc = make_executor(tracked={"OLB"})
        exc.ib.positions.return_value = [_pos("OLB", -5, 0.3)]
        exc.ib.openTrades.return_value = [_open_trade("OLB", "BUY")]
        streamer = MagicMock()
        streamer.contracts = {"OLB": MagicMock()}
        exc._guard_netting(streamer, set())
        exc.ib.placeOrder.assert_not_called()

    def test_no_double_order_for_same_qty(self, monkeypatch):
        monkeypatch.setattr(TRADING_CONFIG, "direction_mode", "long_only")
        exc = make_executor(tracked={"OLB"})
        exc.ib.positions.return_value = [_pos("OLB", -5, 0.3)]
        exc.ib.openTrades.return_value = []
        exc._flattening = {"OLB": 5}
        streamer = MagicMock()
        streamer.contracts = {"OLB": MagicMock()}
        exc._guard_netting(streamer, set())
        exc.ib.placeOrder.assert_not_called()

    def test_prunes_flattening_when_short_closed(self, monkeypatch):
        monkeypatch.setattr(TRADING_CONFIG, "direction_mode", "long_only")
        exc = make_executor(tracked={"OLB"})
        exc.ib.positions.return_value = [_pos("OLB", 50, 0.3)]
        exc._flattening = {"OLB": 5}
        streamer = MagicMock()
        streamer.contracts = {"OLB": MagicMock()}
        exc._guard_netting(streamer, set())
        assert exc._flattening == {}

    def test_skips_sym_in_closing(self, monkeypatch):
        monkeypatch.setattr(TRADING_CONFIG, "direction_mode", "long_only")
        exc = make_executor(tracked={"OLB"})
        exc.ib.positions.return_value = [_pos("OLB", -5, 0.3)]
        exc.ib.openTrades.return_value = []
        streamer = MagicMock()
        streamer.contracts = {"OLB": MagicMock()}
        exc._guard_netting(streamer, {"OLB"})
        exc.ib.placeOrder.assert_not_called()

    def test_skips_when_contract_missing(self, monkeypatch):
        monkeypatch.setattr(TRADING_CONFIG, "direction_mode", "long_only")
        exc = make_executor(tracked={"OLB"})
        exc.ib.positions.return_value = [_pos("OLB", -5, 0.3)]
        exc.ib.openTrades.return_value = []
        streamer = MagicMock()
        streamer.contracts = {}
        exc._guard_netting(streamer, set())
        exc.ib.placeOrder.assert_not_called()

    def test_cancels_sell_legs_before_flatten(self, monkeypatch):
        monkeypatch.setattr(TRADING_CONFIG, "direction_mode", "long_only")
        exc = make_executor(tracked={"OLB"})
        exc.ib.positions.return_value = [_pos("OLB", -5, 0.3)]
        exc.ib.openTrades.return_value = [_open_trade("OLB", "SELL")]
        exc.ib.placeOrder.side_effect = lambda *a: None
        streamer = MagicMock()
        streamer.contracts = {"OLB": MagicMock()}
        with patch("hanoon_prime.ib_executor.cancel_sell_legs") as cancel:
            exc._guard_netting(streamer, set())
        cancel.assert_called_once_with(exc.ib, "OLB")
        assert exc.ib.placeOrder.call_count == 1
