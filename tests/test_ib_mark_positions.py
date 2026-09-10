"""mark_positions — the live positions surface used by /positions."""
from types import SimpleNamespace

from hanoon_prime._ib_marks import mark_positions


def _contract(symbol: str) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol)


def _position(symbol: str, qty: float, avg_cost: float) -> SimpleNamespace:
    return SimpleNamespace(contract=_contract(symbol), position=qty, avgCost=avg_cost)


def _portfolio_item(symbol: str, price: float, pnl: float) -> SimpleNamespace:
    return SimpleNamespace(
        contract=_contract(symbol),
        marketPrice=price,
        unrealizedPNL=pnl,
        marketValue=0.0,
        averageCost=0.0,
        realizedPNL=0.0,
        account="DU1234567",
    )


def _ticker(symbol: str, last: float) -> SimpleNamespace:
    return SimpleNamespace(
        contract=_contract(symbol), last=last, close=0.0, bid=0.0, ask=0.0
    )


def test_marks_from_portfolio_when_present() -> None:
    ib = SimpleNamespace(
        positions=lambda: [_position("TQQQ", 100, 70.0)],
        portfolio=lambda: [_portfolio_item("TQQQ", 71.13, 113.0)],
        tickers=[_ticker("TQQQ", 71.13)],
    )
    payload = mark_positions(ib)
    assert payload["count"] == 1
    row = payload["positions"][0]
    assert row["market_price"] == 71.13
    assert row["unrealized_pnl"] == 113.0
    assert row["direction"] == "LONG"
    assert row["entry_price"] == 70.0
    assert payload["total_pnl"] == 113.0


def test_portfolio_silent_falls_back_to_live_ticker() -> None:
    ib = SimpleNamespace(
        positions=lambda: [_position("SOXL", -50, 30.0)],
        portfolio=lambda: [],
        tickers=[_ticker("SOXL", 31.5)],
    )
    payload = mark_positions(ib)
    row = payload["positions"][0]
    assert row["market_price"] == 31.5
    assert row["direction"] == "SHORT"
    assert row["unrealized_pnl"] == -75.0
    assert row["pnl_pct"] == 5.0


def test_nan_mark_never_survives() -> None:
    ib = SimpleNamespace(
        positions=lambda: [_position("AIM", 100, 10.0)],
        portfolio=lambda: [_portfolio_item("AIM", float("nan"), 0.0)],
        tickers=[_ticker("AIM", 10.5)],
    )
    payload = mark_positions(ib)
    assert payload["positions"][0]["market_price"] == 10.5
    assert payload["positions"][0]["unrealized_pnl"] == 50.0


def test_zero_quantity_stubs_skipped() -> None:
    ib = SimpleNamespace(
        positions=lambda: [_position("A", 0, 10.0), _position("B", 5, 10.0)],
        portfolio=lambda: [_portfolio_item("A", 11.0, 1.0)],
        tickers=[],
    )
    payload = mark_positions(ib)
    assert payload["count"] == 1
    assert payload["positions"][0]["ticker"] == "B"


def test_disconnected_returns_empty() -> None:
    def boom() -> None:
        raise RuntimeError("IB down")

    ib = SimpleNamespace(positions=boom, portfolio=lambda: [], tickers=[])
    assert mark_positions(ib) == {
        "positions": [],
        "total_pnl": 0.0,
        "count": 0,
    }
