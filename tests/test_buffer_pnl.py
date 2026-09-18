"""Regression tests for TradeBuffer average-cost PnL accounting.

Guards the fix for the short-PnL bug: previously avg entry was only set
for buys, so every short persisted entry 0.0 and was booked as a loss
(pnl = -qty x price). Now: Long = (exit - entry) x qty,
Short = (entry - exit) x qty, fees subtracted, partial closes realized.
"""

from __future__ import annotations

from hanoon_prime.reflection.buffer import BUY, SELL, Fill, TradeBuffer


def _buf(tmp_path) -> TradeBuffer:
    return TradeBuffer(filepath=tmp_path / "b.json")


def test_short_round_trip_profit(tmp_path) -> None:
    b = _buf(tmp_path)
    assert b.on_fill(Fill("BITO", SELL, 10, 100.0, 1.0)) is None
    t = b.on_fill(Fill("BITO", BUY, 10, 90.0, 2.0))
    assert t is not None
    assert t.win is True
    assert t.pnl == 100.0
    assert t.avg_entry == 100.0
    assert t.avg_exit == 90.0
    assert t.qty == 10.0


def test_short_round_trip_loss(tmp_path) -> None:
    b = _buf(tmp_path)
    b.on_fill(Fill("BITO", SELL, 10, 100.0, 1.0))
    t = b.on_fill(Fill("BITO", BUY, 10, 110.0, 2.0))
    assert t is not None
    assert t.win is False
    assert t.pnl == -100.0


def test_fees_subtracted_both_sides(tmp_path) -> None:
    b = _buf(tmp_path)
    b.on_fill(Fill("SPY", BUY, 10, 100.0, 1.0, commission=1.0))
    t = b.on_fill(Fill("SPY", SELL, 10, 110.0, 2.0, commission=2.0))
    assert t is not None
    assert t.pnl == 97.0
    assert t.fees == 3.0


def test_partial_close_realizes_then_nets(tmp_path) -> None:
    b = _buf(tmp_path)
    b.on_fill(Fill("GGB", BUY, 10, 100.0, 1.0))
    assert b.on_fill(Fill("GGB", SELL, 4, 110.0, 2.0)) is None
    t = b.on_fill(Fill("GGB", SELL, 6, 120.0, 3.0))
    assert t is not None
    assert t.pnl == 160.0
    assert t.avg_exit == 116.0
    assert b.snapshot()["open"] == 0


def test_scale_in_weighted_entry(tmp_path) -> None:
    b = _buf(tmp_path)
    b.on_fill(Fill("MSTZ", BUY, 5, 100.0, 1.0))
    b.on_fill(Fill("MSTZ", BUY, 5, 120.0, 2.0))
    t = b.on_fill(Fill("MSTZ", SELL, 10, 130.0, 3.0))
    assert t is not None
    assert t.avg_entry == 110.0
    assert t.pnl == 200.0


def test_scale_in_after_partial_uses_remaining_avg(tmp_path) -> None:
    b = _buf(tmp_path)
    b.on_fill(Fill("DFDV", SELL, 10, 100.0, 1.0))
    b.on_fill(Fill("DFDV", BUY, 4, 90.0, 2.0))  # realized +40, open 6
    b.on_fill(Fill("DFDV", SELL, 6, 120.0, 3.0))  # book back to avg 110
    t = b.on_fill(Fill("DFDV", BUY, 12, 110.0, 4.0))
    assert t is not None
    # +40 realized on the first cover; the re-opened leg (avg 110) covers
    # flat, so only the partial profit remains. fees 0.
    assert t.pnl == 40.0


def test_flip_closes_and_reopens(tmp_path) -> None:
    b = _buf(tmp_path)
    b.on_fill(Fill("MKDW", BUY, 5, 100.0, 1.0))
    assert b.on_fill(Fill("MKDW", SELL, 10, 110.0, 2.0)) is None  # flip short 5
    t = b.on_fill(Fill("MKDW", BUY, 5, 100.0, 3.0))
    assert t is not None
    assert t.pnl == 100.0
    assert b.snapshot()["open"] == 0
