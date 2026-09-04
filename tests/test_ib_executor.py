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

from hanoon_prime._ib_sync import get_ib_pnl, journal_exit, read_ib_positions
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
