"""tests/test_pnl_units.py — P&L units contract for the learning loop.

The learning loop consumes ``pnl_pct`` as a *return fraction* (e.g. 0.006
== +0.6%). Evidence in-tree:

* ``brain/realized_ev.py`` stores ``abs(pnl_pct)`` into RR samples that are
  fractions (see ``runtime/juli_realized.json`` -> ``rr_samples``).
* ``monitor/exit_scoring.py`` thresholds on ``pnl_pct`` at 0.03/0.05.
* ``brain/learning_config.py`` maps ``pnl_pct`` -> bandit reward via
  ``0.5 + pnl * 5.0`` (a fraction).
* ``_ib_sync._calc_pnl_from_fill`` itself returns a fraction.

But ``_ib_sync.get_ib_pnl`` has two paths with *different units*:
``_pnl_from_trade`` returns IB's raw ``t.pnl`` in **dollars** while the
fallback returns a **fraction**. That dollars-as-percent leak corrupts the
realized-EV gate, the horizon bandit, the RR stats and the exit tuner.

These tests pin ``get_ib_pnl`` to a single unit: the return fraction.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hanoon_prime._ib_sync import get_ib_pnl
from hanoon_prime.types import Position


def make_pos(
    ticker: str = "GRAB",
    entry_price: float = 3.045,
    shares: float = 184.0,
    direction: int = 1,
) -> Position:
    """A realistic live scalp position (matches journal_live.jsonl GRAB)."""
    return Position(
        ticker=ticker,
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


def make_trade(symbol: str, pnl_dollars: float, done: bool = True) -> MagicMock:
    """Mock IB trade carrying a DOLLAR realized P&L (IB's native unit)."""
    trade = MagicMock()
    trade.isDone.return_value = done
    trade.pnl = pnl_dollars
    trade.contract.symbol = symbol
    trade.order.auxPrice = 0.0
    trade.order.lmtPrice = 0.0
    return trade


def _ib_with(*trades: MagicMock) -> MagicMock:
    ib = MagicMock()
    ib.trades.return_value = list(trades)
    return ib


def test_get_ib_pnl_returns_return_fraction_not_dollars() -> None:
    """A $1.98 win on 184 sh @ $3.045 must read as ~0.35%, not as 1.98."""
    pos = make_pos(entry_price=3.045, shares=184.0, direction=1)
    ib = _ib_with(make_trade("GRAB", pnl_dollars=1.98))

    pnl = get_ib_pnl(ib, "GRAB", pos)

    expected = 1.98 / (3.045 * 184.0)
    assert pnl == pytest.approx(expected, rel=1e-3), (
        f"get_ib_pnl returned {pnl!r}; expected the return fraction "
        f"{expected!r}. A dollar amount leaked through as pnl_pct."
    )
    assert abs(pnl) < 0.05, (
        f"|pnl|={abs(pnl):.3f} is implausible for a scalp return fraction; "
        "raw dollar P&L is being used as a percentage."
    )


def test_get_ib_pnl_sign_preserved_for_loss() -> None:
    """A $0.66 loss stays negative after unit normalization."""
    pos = make_pos(entry_price=3.0135, shares=148.0, direction=1)
    ib = _ib_with(make_trade("GRAB", pnl_dollars=-0.6648))

    pnl = get_ib_pnl(ib, "GRAB", pos)

    assert pnl < 0
    assert pnl == pytest.approx(-0.6648 / (3.0135 * 148.0), rel=1e-3)


def test_get_ib_pnl_consistent_between_trade_and_fill_paths() -> None:
    """Both IB paths must agree on units for the same economic outcome.

    Trade path carries dollars; fill path derives the fraction from
    prices. With a fill price implying +0.35%, both should return
    (approximately) the same fraction.
    """
    entry = 3.045
    shares = 184.0
    pos = make_pos(entry_price=entry, shares=shares, direction=1)

    dollars = 1.98
    fraction = dollars / (entry * shares)

    trade_path = get_ib_pnl(_ib_with(make_trade("GRAB", dollars)), "GRAB", pos)

    # fill-price fallback: t.pnl falsy -> derive from fill price.
    fill_trade = make_trade("GRAB", 0.0)
    fill_price = entry * (1.0 + fraction)
    fill_trade.order.auxPrice = fill_price
    fill_path = get_ib_pnl(_ib_with(fill_trade), "GRAB", pos)

    assert trade_path == pytest.approx(fill_path, rel=1e-2)


def test_get_ib_pnl_direction_aware_for_short() -> None:
    """Short wins when price falls; dollars negative -> fraction positive.

    IB's ``t.pnl`` is already signed in account currency, so the
    normalized fraction must keep that sign for shorts too.
    """
    pos = make_pos(entry_price=10.0, shares=100.0, direction=-1)
    ib = _ib_with(make_trade("XYZ", pnl_dollars=25.0))  # +$25 short profit

    pnl = get_ib_pnl(ib, "XYZ", pos)

    assert pnl == pytest.approx(25.0 / (10.0 * 100.0), rel=1e-3)
    assert pnl > 0
