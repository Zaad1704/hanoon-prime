"""tests/test_pnl_display.py — Telegram P&L must be DOLLARS, not a return fraction.

The learning loop consumes ``pnl_pct`` as a return fraction (pinned by
test_pnl_units.py). But the Telegram close notification is human-facing: a
fraction displayed as ``$`` reads the wrong magnitude and mis-classifies
WINS/LOSSES/BREAKEVENS (e.g. a real -$0.37 FRSX loss showed as ``$-0.0099``
"BREAKEVEN"). These tests pin the display path to dollars.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hanoon_prime.ib_executor import IBExecutor
from hanoon_prime.types import Position, fraction_to_dollars


def _make_executor(tracked=None) -> IBExecutor:
    return IBExecutor(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        **({"tracked_tickers": set(tracked)} if tracked else {}),
    )


def _make_pos(entry_price: float, shares: float, direction: int) -> Position:
    return Position(
        ticker="TSLA",
        entry_idx=0,
        entry_price=entry_price,
        shares=shares,
        direction=direction,
        stop_price=entry_price * 0.98,
        target_price=entry_price * 1.05,
        peak_price=entry_price,
        score=0.5,
        atr=0.05,
    )


def _trade_with_pnl(symbol: str, pnl_dollars: float) -> MagicMock:
    """Mock an IB trade carrying a DOLLAR realized P&L."""
    trade = MagicMock()
    trade.isDone.return_value = True
    trade.pnl = pnl_dollars
    trade.contract.symbol = symbol
    trade.order.auxPrice = 0.0
    trade.order.lmtPrice = 0.0
    return trade


def test_fraction_to_dollars_long() -> None:
    """+0.6% on 184 sh @ $3.045 is $3.36, not 0.006."""
    pos = _make_pos(3.045, 184.0, 1)
    assert fraction_to_dollars(0.006, pos) == pytest.approx(3.36168, abs=1e-9)


def test_fraction_to_dollars_short() -> None:
    """Short loss fraction maps to dollars with correct sign."""
    pos = _make_pos(18.6, 2.0, -1)
    assert fraction_to_dollars(-0.0123, pos) == pytest.approx(-0.45756, abs=1e-9)


def test_fraction_to_dollars_guards_zero_notional() -> None:
    """Zero entry/shares must not divide — return the fraction unchanged."""
    pos = _make_pos(0.0, 0.0, 1)
    assert fraction_to_dollars(0.25, pos) == 0.25


class TestRecordExitTelegramAmount:
    """_record_exit must hand Telegram the DOLLAR P&L, not the fraction."""

    def _record(self, trade: MagicMock) -> tuple[float, str]:
        exc = _make_executor({"TSLA"})
        exc.brain._open_positions = {"TSLA": _make_pos(3.045, 184, 1)}
        exc.ib.trades.return_value = [trade]
        streamer = MagicMock()
        streamer.get_last_price.return_value = 3.05
        with patch("hanoon_prime.ib_executor.trade_closed") as fn:
            exc._record_exit("TSLA", streamer)
        fn.assert_called_once()
        pnl_arg = fn.call_args[0][2]
        side_arg = fn.call_args[0][1]
        return pnl_arg, side_arg

    def test_real_close_passes_dollars(self) -> None:
        """IB reports $1.98 realized; Telegram must show $1.98, not 0.0035."""
        pnl_arg, _ = self._record(_trade_with_pnl("TSLA", 1.98))
        assert pnl_arg == pytest.approx(1.98, rel=1e-6)

    def test_shorts_report_sign_correct(self) -> None:
        """A short that loses $0.66 notifies a NEGATIVE dollar amount."""
        trade = _trade_with_pnl("TSLA", -0.6648)
        exc = _make_executor({"TSLA"})
        exc.brain._open_positions = {"TSLA": _make_pos(3.0135, 148, -1)}
        exc.ib.trades.return_value = [trade]
        streamer = MagicMock()
        streamer.get_last_price.return_value = 3.02
        with patch("hanoon_prime.ib_executor.trade_closed") as fn:
            exc._record_exit("TSLA", streamer)
        pnl_arg = fn.call_args[0][2]
        assert pnl_arg == pytest.approx(-0.6648, rel=1e-6)
        assert pnl_arg < 0
