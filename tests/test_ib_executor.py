"""tests/test_ib_executor.py — tests for IB-as-source-of-truth behavior.

Tests verify:
1. _read_ib_positions reads ALL tracked tickers (not just _brackets)
2. _do_exit does NOT write journal entries (carbon copy principle)
3. cancel_all calls ib.cancelAllOrders()
4. _sync_and_cancel_orders reads brackets from IB, cancels orphans, trails stops
5. _record_exit reads IB trade P&L directly (no local journal write)
6. _ping_ib verifies connection before sync
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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
    trade.isDone.return_value = False  # active order, eligible for cancel
    trade.contract = MagicMock()
    trade.contract.symbol = contract_sym or symbol
    return trade


class TestReadIbPositions:
    """_read_ib_positions reads ALL tracked tickers from IB (not just brackets)."""

    def test_reads_all_tracked_tickers(self):
        exc = make_executor(tracked={"AAPL", "MSFT", "NVDA"})
        exc.ib.positions.return_value = [
            _pos("AAPL", 10, 100.0),
            _pos("MSFT", -5, 200.0),
            _pos("NVDA", 3, 150.0),
        ]
        positions = exc._read_ib_positions()
        assert set(positions.keys()) == {"AAPL", "MSFT", "NVDA"}

    def test_ignores_untracked_positions(self):
        exc = make_executor(tracked={"AAPL"})
        exc.ib.positions.return_value = [
            _pos("AAPL", 10, 100.0),
            _pos("GOOGL", 5, 200.0),
        ]
        positions = exc._read_ib_positions()
        assert set(positions.keys()) == {"AAPL"}

    def test_direction_and_shares(self):
        exc = make_executor(tracked={"AAPL", "MSFT"})
        exc.ib.positions.return_value = [
            _pos("AAPL", 10, 100.0),
            _pos("MSFT", -5, 200.0),
        ]
        positions = exc._read_ib_positions()
        assert positions["AAPL"].direction == 1
        assert positions["MSFT"].direction == -1
        assert positions["AAPL"].shares == 10
        assert positions["MSFT"].shares == 5

    def test_uses_bracket_levels_when_available(self):
        exc = make_executor(tracked={"AAPL"})
        exc._brackets = {"AAPL": (95.0, 110.0)}
        exc.ib.positions.return_value = [_pos("AAPL", 10, 100.0)]
        positions = exc._read_ib_positions()
        assert positions["AAPL"].stop_price == 95.0
        assert positions["AAPL"].target_price == 110.0

    def test_zero_brackets_when_no_bracket_info(self):
        exc = make_executor(tracked={"AAPL"})
        exc.ib.positions.return_value = [_pos("AAPL", 10, 100.0)]
        positions = exc._read_ib_positions()
        assert positions["AAPL"].stop_price == 0.0
        assert positions["AAPL"].target_price == 0.0

    def test_handles_ib_error_gracefully(self):
        exc = make_executor(tracked={"AAPL"})
        exc.ib.positions.side_effect = RuntimeError("IB down")
        assert exc._read_ib_positions() == {}


class TestDoExit:
    """_do_exit must NOT write journal entries — carbon copy principle."""

    @patch("hanoon_prime.ib_executor.ib")
    def test_no_exit_journal_entry(self, mock_ib_mod):
        exc = make_executor(tracked={"TSLA"})
        mock_ib_mod.MarketOrder.return_value = MagicMock()
        exc.brain._open_positions = {"TSLA": make_pos(direction=-1)}
        streamer = MagicMock()
        streamer.contracts = {"TSLA": MagicMock()}
        exc._do_exit("TSLA", 95.0, "stop_loss", streamer)
        for call in exc.journal.append.call_args_list:
            args = call[0]
            if args and isinstance(args[0], dict):
                assert args[0].get("event") != "exit"

    @patch("hanoon_prime.ib_executor.ib")
    def test_places_market_order(self, mock_ib_mod):
        exc = make_executor(tracked={"TSLA"})
        mock_ib_mod.MarketOrder.return_value = MagicMock()
        exc.brain._open_positions = {"TSLA": make_pos(direction=1)}
        streamer = MagicMock()
        streamer.contracts = {"TSLA": MagicMock()}
        exc._do_exit("TSLA", 110.0, "target_hit", streamer)
        exc.ib.placeOrder.assert_called_once()
        mock_ib_mod.MarketOrder.assert_called_with("SELL", 10, tif="DAY")

    @patch("hanoon_prime.ib_executor.ib")
    def test_buys_not_sells_for_short_position(self, mock_ib_mod):
        exc = make_executor(tracked={"TSLA"})
        mock_ib_mod.MarketOrder.return_value = MagicMock()
        exc.brain._open_positions = {"TSLA": make_pos(direction=-1, shares=5)}
        streamer = MagicMock()
        streamer.contracts = {"TSLA": MagicMock()}
        exc._do_exit("TSLA", 95.0, "stop_loss", streamer)
        exc.ib.placeOrder.assert_called_once()
        mock_ib_mod.MarketOrder.assert_called_with("BUY", 5, tif="DAY")


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
        exc._read_ib_positions = MagicMock()
        exc._journal_snapshot = MagicMock()
        exc.sync_from_ib(MagicMock())
        exc._read_ib_positions.assert_not_called()
        exc._journal_snapshot.assert_not_called()


class TestSyncAndCancelOrders:
    """_sync_and_cancel_orders syncs brackets, cancels orphans, trails stops."""

    def test_sync_reads_bracket_levels(self):
        exc = make_executor(tracked={"TSLA"})
        child_sl = MagicMock()
        child_sl.auxPrice = 95.0
        child_sl.lmtPrice = 0.0
        child_tp = MagicMock()
        child_tp.auxPrice = 0.0
        child_tp.lmtPrice = 110.0
        parent_order = MagicMock()
        parent_order.parentId = 0
        parent_order.children = [child_sl, child_tp]
        trade = _make_trade("TSLA", 0, [child_sl, child_tp])
        exc.ib.trades.return_value = [trade]
        streamer = MagicMock()
        exc._sync_and_cancel_orders({"TSLA": make_pos()}, streamer)
        assert exc._brackets["TSLA"][0] == 95.0
        assert exc._brackets["TSLA"][1] == 110.0

    def test_sync_skips_untracked_tickers(self):
        exc = make_executor(tracked={"TSLA"})
        trade = _make_trade("NVDA", 0, [])
        exc.ib.trades.return_value = [trade]
        exc._sync_and_cancel_orders({}, MagicMock())
        assert len(exc._brackets) == 0

    def test_sync_skips_child_orders(self):
        exc = make_executor(tracked={"TSLA"})
        parent_order = MagicMock()
        parent_order.parentId = 0
        parent_order.children = []
        child_order = MagicMock()
        child_order.parentId = 123
        child_order.children = []
        parent_trade = _make_trade("TSLA", 0, [])
        exc.ib.trades.return_value = [parent_trade]
        exc._sync_and_cancel_orders({"TSLA": make_pos()}, MagicMock())
        assert len(exc._brackets) == 0

    def test_sync_survives_error(self):
        exc = make_executor(tracked={"TSLA"})
        exc.ib.trades.side_effect = RuntimeError("IB down")
        exc._sync_and_cancel_orders({}, MagicMock())  # should not raise

    def test_cancel_orphaned_orders(self):
        """Orders with no matching IB position are cancelled."""
        exc = make_executor(tracked={"TSLA"})
        child_sl = MagicMock()
        child_sl.auxPrice = 95.0
        child_sl.lmtPrice = 0.0
        child_tp = MagicMock()
        child_tp.auxPrice = 0.0
        child_tp.lmtPrice = 110.0
        trade = _make_trade("TSLA", 0, [child_sl, child_tp])
        exc.ib.trades.return_value = [trade]
        exc._sync_and_cancel_orders({}, MagicMock())  # No TSLA in positions
        exc.ib.cancelOrder.assert_called_once_with(trade.order)

    def test_no_cancel_for_active_positions(self):
        """Orders with matching IB position are NOT cancelled."""
        exc = make_executor(tracked={"TSLA"})
        child_sl = MagicMock()
        child_sl.auxPrice = 95.0
        child_sl.lmtPrice = 0.0
        child_tp = MagicMock()
        child_tp.auxPrice = 0.0
        child_tp.lmtPrice = 110.0
        parent_order = MagicMock()
        parent_order.parentId = 0
        parent_order.children = [child_sl, child_tp]
        trade = _make_trade("TSLA", 0, [child_sl, child_tp])
        exc.ib.trades.return_value = [trade]
        exc._sync_and_cancel_orders({"TSLA": make_pos()}, MagicMock())
        exc.ib.cancelOrder.assert_not_called()


class TestTrailStop:
    """_trail_stop modifies IB stop-loss orders for in-favor positions."""

    def test_trails_long_position(self):
        """Long position: if price > stop + ATR, move stop up."""
        exc = make_executor(tracked={"TSLA"})
        pos = make_pos(ticker="TSLA", direction=1, entry_price=100.0, atr=5.0)
        stop_child = MagicMock()
        stop_child.auxPrice = 95.0
        trade = MagicMock()
        trade.order = MagicMock()
        trade.order.children = [stop_child]
        streamer = MagicMock()
        streamer.get_last_price.return_value = 110.0  # 15 above stop, > ATR(5.0)
        streamer.buffer_atr.return_value = 5.0
        exc._trail_stop("TSLA", pos, streamer, 95.0, trade)
        assert stop_child.auxPrice == round(110.0 - 1 * 2.0 * 5.0, 2)
        assert stop_child.auxPrice == 100.0

    def test_does_not_trail_when_not_in_favor(self):
        """Long position: if price barely moved, don't trail."""
        exc = make_executor(tracked={"TSLA"})
        pos = make_pos(ticker="TSLA", direction=1, entry_price=100.0, atr=5.0)
        stop_child = MagicMock()
        stop_child.auxPrice = 96.0
        trade = MagicMock()
        trade.order = MagicMock()
        trade.order.children = [stop_child]
        streamer = MagicMock()
        streamer.get_last_price.return_value = 97.0  # only 1 above stop, < ATR(5.0)
        streamer.buffer_atr.return_value = 5.0
        exc._trail_stop("TSLA", pos, streamer, 96.0, trade)
        assert stop_child.auxPrice == 96.0  # unchanged

    def test_skips_when_no_price(self):
        exc = make_executor(tracked={"TSLA"})
        pos = make_pos(direction=1)
        streamer = MagicMock()
        streamer.get_last_price.return_value = None
        exc._trail_stop("TSLA", pos, streamer, 95.0, MagicMock())
        streamer.buffer_atr.assert_not_called()

    def test_trails_short_position(self):
        """Short position: if price < stop - ATR, move stop down."""
        exc = make_executor(tracked={"TSLA"})
        pos = make_pos(ticker="TSLA", direction=-1, entry_price=100.0, atr=5.0)
        stop_child = MagicMock()
        stop_child.auxPrice = 105.0
        trade = MagicMock()
        trade.order = MagicMock()
        trade.order.children = [stop_child]
        streamer = MagicMock()
        streamer.get_last_price.return_value = 90.0  # 15 below stop, > ATR(5.0)
        streamer.buffer_atr.return_value = 5.0
        exc._trail_stop("TSLA", pos, streamer, 105.0, trade)
        assert stop_child.auxPrice == round(90.0 - (-1) * 2.0 * 5.0, 2)
        assert stop_child.auxPrice == 100.0


class TestRecordExitPnl:
    """_record_exit reads P&L from IB trade data directly — no journal write."""

    def test_uses_ib_trade_pnl(self):
        """When IB has trade P&L, it's used (converted to pct)."""
        exc = make_executor(tracked={"TSLA"})
        exc.brain._open_positions = {
            "TSLA": make_pos(direction=1, shares=10, entry_price=100.0)
        }
        streamer = MagicMock()
        streamer.get_last_price.return_value = 110.0
        trade = MagicMock()
        trade.contract = MagicMock()
        trade.contract.symbol = "TSLA"
        trade.isDone.return_value = True
        trade.pnl = 500.0  # $500 = 500/(100*10) = 50%
        exc.ib.trades.return_value = [trade]
        exc._record_exit("TSLA", streamer)
        exc.brain.record_trade.assert_called_once()
        kwargs = exc.brain.record_trade.call_args.kwargs
        assert kwargs["pnl_pct"] == pytest.approx(0.5)

    def test_falls_back_to_local_on_no_trades(self):
        exc = make_executor(tracked={"TSLA"})
        exc.brain._open_positions = {
            "TSLA": make_pos(direction=1, shares=10, entry_price=100.0)
        }
        streamer = MagicMock()
        streamer.get_last_price.return_value = 110.0
        exc.ib.trades.return_value = []
        exc._record_exit("TSLA", streamer)
        exc.brain.record_trade.assert_called_once()
        kwargs = exc.brain.record_trade.call_args.kwargs
        assert kwargs["pnl_pct"] == pytest.approx(0.1)  # (110-100)/100 * 1

    def test_falls_back_to_local_on_ib_error(self):
        exc = make_executor(tracked={"TSLA"})
        exc.brain._open_positions = {
            "TSLA": make_pos(direction=1, shares=10, entry_price=100.0)
        }
        streamer = MagicMock()
        streamer.get_last_price.return_value = 105.0
        exc.ib.trades.side_effect = RuntimeError("IB down")
        exc._record_exit("TSLA", streamer)
        exc.brain.record_trade.assert_called_once()
        kwargs = exc.brain.record_trade.call_args.kwargs
        assert kwargs["pnl_pct"] == pytest.approx(0.05)  # (105-100)/100 * 1

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
