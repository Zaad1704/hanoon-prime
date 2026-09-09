"""test_coverage_services — pure-logic tests for review, data budget, and
position protection modules that were sitting at 0% coverage.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hanoon_prime._protect import (
    _atr_levels,
    _place_oca,
    _validate_protection,
    protect_position,
)
from hanoon_prime.data.budget import MAX_TBT, DataBudget
from hanoon_prime.ib_order_sweep import (
    _get_oca_orders,
    _is_valid_protection,
    _sweep_stale_orders,
    sweep_zombies,
)
from hanoon_prime.reflection.buffer import BUY, SELL, Fill, TradeBuffer
from hanoon_prime.reflection.review import (
    ReviewReport,
    ReviewSession,
    TradePostmortem,
    TradeReview,
)
from hanoon_prime.types import BracketOrder


def _fake_trade(
    order_type: str, action: str = "SELL", qty: int = 100, oca: str = "JULI_TSLA"
) -> SimpleNamespace:
    return SimpleNamespace(
        order=SimpleNamespace(
            orderType=order_type, action=action, totalQuantity=qty, ocaGroup=oca
        ),
        orderStatus=SimpleNamespace(status="Submitted"),
    )


def _spot(
    sym: str,
    order_type: str,
    qty: float,
    action: str = "SELL",
    status: str = "PreSubmitted",
    oca: str = "",
    aux: float | None = None,
    lmt: float | None = None,
) -> SimpleNamespace:
    """A richer fake trade: contract symbol, price legs, and arbitrary status."""
    fields: dict[str, object] = {
        "orderType": order_type,
        "action": action,
        "totalQuantity": qty,
        "ocaGroup": oca,
    }
    if aux is not None:
        fields["auxPrice"] = aux
    if lmt is not None:
        fields["lmtPrice"] = lmt
    return SimpleNamespace(
        order=SimpleNamespace(**fields),
        orderStatus=SimpleNamespace(status=status),
        contract=SimpleNamespace(symbol=sym),
    )


def _positions(*held: tuple[str, float]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(contract=SimpleNamespace(symbol=sym), position=pos)
        for sym, pos in held
    ]


# ── reflection/review ───────────────────────────────────────────────────
class TestReviewSession:
    def test_empty_buffer(self):
        rep = ReviewSession(None).run_daily_review()
        assert isinstance(rep, ReviewReport)
        assert rep.action_items == []

    def test_no_trades(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        rep = ReviewSession(buf).run_daily_review()
        assert rep.summary.get("total", 0) == 0

    def test_with_mixed_trades(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        buf.on_fill(Fill("TSLA", BUY, 10, 100.0, 1.0))
        buf.on_fill(Fill("TSLA", SELL, 10, 110.0, 2.0))  # win
        buf.on_fill(Fill("TSLA", BUY, 10, 100.0, 3.0))
        buf.on_fill(Fill("TSLA", SELL, 10, 90.0, 4.0))  # loss
        rep = ReviewSession(buf).run_daily_review()
        assert rep.summary["total"] == 2
        assert "wr" in rep.summary

    def test_low_wr_generates_item(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        # five losing round trips for the same ticker
        for i in range(5):
            t0 = 1000.0 + i
            buf.on_fill(Fill("TSLA", BUY, 1, 100.0, t0))
            buf.on_fill(Fill("TSLA", SELL, 1, 90.0, t0 + 1.0))
        rep = ReviewSession(buf).run_daily_review()
        assert any("Portfolio WR" in item for item in rep.action_items)

    def test_weekly_matches_daily(self, tmp_path):
        buf = TradeBuffer(filepath=tmp_path / "b.json")
        daily = ReviewSession(buf).run_daily_review()
        weekly = ReviewSession(buf).run_weekly_review()
        assert weekly.summary == daily.summary


class TestTradeReview:
    def test_win_postmortem(self):
        trade = SimpleNamespace(ticker="TSLA", pnl=20.0, win=True)
        pm = TradeReview().review(trade, alpha=None)
        assert isinstance(pm, TradePostmortem)
        assert pm.ticker == "TSLA"
        assert pm.won is True
        assert pm.lessons[0].startswith("Won")

    def test_loss_with_strong_alpha(self):
        trade = SimpleNamespace(ticker="TSLA", pnl=-15.0, win=False)
        alpha = {"momentum": 0.8, "rsi": 0.5}
        pm = TradeReview().review(trade, alpha=alpha)
        assert pm.won is False
        assert any("Strong momentum" in lesson for lesson in pm.lessons)

    def test_loss_without_alpha(self):
        trade = SimpleNamespace(ticker="TSLA", pnl=-5.0, win=False)
        pm = TradeReview().review(trade)
        assert len(pm.lessons) == 1


# ── data budget ─────────────────────────────────────────────────────────
class TestDataBudget:
    def test_positions_get_tbt(self):
        db = DataBudget()
        to_sub, to_unsub = db.allocate(positions={"A", "B"}, candidates=[])
        assert to_sub == {"A": "TBT", "B": "TBT"}
        assert to_unsub == set()

    def test_candidates_fill_tbt_then_l1(self):
        db = DataBudget()
        candidates = [f"S{i}" for i in range(20)]
        to_sub, _ = db.allocate(positions=set(), candidates=candidates)
        assert len(db.get_tbt_tickers()) == MAX_TBT
        assert len(to_sub) == 20

    def test_diff_detects_changes(self):
        db = DataBudget()
        db.allocate(positions={"A", "B"}, candidates=[])
        # drop B, add C
        to_sub, to_unsub = db.allocate(positions={"A", "C"}, candidates=[])
        assert to_sub == {"C": "TBT"}
        assert to_unsub == {"B"}

    def test_remove_and_count(self):
        db = DataBudget()
        db.allocate(positions={"A", "B"}, candidates=[])
        db.remove("A")
        assert "A" not in db.get_all_tracked()
        counts = db.count_tiers()
        assert counts["TBT"] == 1

    def test_tbt_cap_respected(self):
        db = DataBudget()
        db.allocate(positions={"A", "B"}, candidates=[])
        # candidates should not be added to TBT if full
        assert len(db.get_tbt_tickers()) <= MAX_TBT


# ── _protect ────────────────────────────────────────────────────────────
class TestProtect:
    def test_is_valid_protection(self):
        assert _is_valid_protection([]) is False
        valid = [_fake_trade("STP"), _fake_trade("LMT")]
        assert _is_valid_protection(valid) is True
        bad_types = [_fake_trade("STP"), _fake_trade("STP")]
        assert _is_valid_protection(bad_types) is False

    def test_is_valid_protection_rejects_degenerate_prices(self):
        # target == stop (OLB-style) is not a real take-profit pair.
        degenerate = [
            _spot("OLB", "STP", 1003, aux=0.27, oca="JULI_OLB"),
            _spot("OLB", "LMT", 1003, lmt=0.27, oca="JULI_OLB"),
        ]
        assert _is_valid_protection(degenerate) is False
        sane = [
            _spot("OLB", "STP", 1003, aux=0.27, oca="JULI_OLB"),
            _spot("OLB", "LMT", 1003, lmt=0.29, oca="JULI_OLB"),
        ]
        assert _is_valid_protection(sane) is True

    def test_validate_protection_price_sanity(self):
        good = [
            _spot("TSLA", "STP", 100, action="SELL", aux=99.0, oca="JULI_TSLA"),
            _spot("TSLA", "LMT", 100, action="SELL", lmt=110.0, oca="JULI_TSLA"),
        ]
        assert _validate_protection(good, 100, "SELL") is True
        degenerate = [
            _spot("TSLA", "STP", 100, action="SELL", aux=110.0, oca="JULI_TSLA"),
            _spot("TSLA", "LMT", 100, action="SELL", lmt=110.0, oca="JULI_TSLA"),
        ]
        assert _validate_protection(degenerate, 100, "SELL") is False

    def test_atr_levels_never_degenerate(self):
        stop, target = _atr_levels(0.27, 0.0, 1)
        assert stop < target
        assert target > 0.27
        assert target - stop >= 0.009
        short_stop, short_target = _atr_levels(0.27, 0.0, -1)
        assert short_target < short_stop
        assert short_stop > 0.27
        assert short_stop - short_target >= 0.009

    def test_validate_protection(self):
        good = [
            _fake_trade("STP", action="SELL", qty=100),
            _fake_trade("LMT", action="SELL", qty=100),
        ]
        assert _validate_protection(good, 100, "SELL") is True
        bad_qty = [
            _fake_trade("STP", action="SELL", qty=50),
            _fake_trade("LMT", action="SELL", qty=100),
        ]
        assert _validate_protection(bad_qty, 100, "SELL") is False
        assert _validate_protection([_fake_trade("STP")], 1, "SELL") is False

    def test_sweep_zombies_empty(self):
        ib = MagicMock()
        ib.openTrades.return_value = []
        assert sweep_zombies(ib) is None

    def test_sweep_zombies_fixes_broken(self):
        ib = MagicMock()
        broken = _fake_trade("STP", oca="JULI_X")  # only one order -> invalid
        ib.openTrades.return_value = [broken]
        ib.positions.return_value = []
        sweep_zombies(ib)
        ib.cancelOrder.assert_called_once_with(broken.order)

    def test_sweep_zombies_opentrades_exception(self):
        ib = MagicMock()
        ib.openTrades.side_effect = RuntimeError("conn")
        sweep_zombies(ib)
        ib.cancelOrder.assert_not_called()

    def test_sweep_cancels_stale_mkt_mismatching_position(self):
        ib = MagicMock()
        stale = _spot("OLB", "MKT", 446)  # 446 queued vs 1003 held -> stale
        good = _spot("NOK", "MKT", 39)  # full-size queued exit -> keep
        # LABT has a valid JULI OCA pair; its queued MARKET leg must survive.
        ib.openTrades.return_value = [
            stale,
            good,
            _spot("LABT", "STP", 148, oca="JULI_LABT"),
            _spot("LABT", "LMT", 148, oca="JULI_LABT"),
        ]
        ib.positions.return_value = _positions(("OLB", 1003.0), ("NOK", 39.0))
        ib.cancelOrder = MagicMock()
        sweep_zombies(ib)
        assert ib.cancelOrder.call_count == 1
        assert ib.cancelOrder.call_args[0][0] is stale.order

    def test_sweep_cancels_orphan_buy_on_flat_symbol(self):
        ib = MagicMock()
        orphan = _spot("PDSB", "LMT", 1227, action="BUY", status="Submitted")
        kept = _spot("SOXS", "LMT", 12, action="BUY", status="Submitted")
        ib.openTrades.return_value = [orphan, kept]
        ib.positions.return_value = _positions(("SOXS", 8.0))
        sweep_zombies(ib)
        assert ib.cancelOrder.call_count == 1
        assert ib.cancelOrder.call_args[0][0] is orphan.order

    def test_sweep_keeps_known_pending_parent(self):
        ib = MagicMock()
        pending = _spot("ACCL", "LMT", 134, action="BUY", status="Submitted")
        ib.openTrades.return_value = [pending]
        ib.positions.return_value = []
        sweep_zombies(ib, known_pending={"ACCL"})
        ib.cancelOrder.assert_not_called()

    def test_sweep_stale_orders_skips_without_positions(self):
        ib = MagicMock()
        stale = _spot("OLB", "MKT", 446)
        ib.openTrades.return_value = [stale]
        ib.positions.side_effect = RuntimeError("conn")
        _sweep_stale_orders(ib, [stale])
        ib.cancelOrder.assert_not_called()

    def test_get_oca_orders(self):
        ib = MagicMock()
        t1 = _fake_trade("STP", oca="JULI_TSLA")
        t2 = _fake_trade("LMT", oca="JULI_TSLA")
        t3 = _fake_trade("STP", oca="JULI_OTHER")
        ib.openTrades.return_value = [t1, t2, t3]
        result = _get_oca_orders(ib, "TSLA")
        assert result == [t1, t2]

    def test_place_oca_uses_stub_when_ib_missing(self, monkeypatch):
        # _ib None (ib_insync absent, or stubbed here) -> Order() raises
        # AttributeError. Patched explicitly so the test is deterministic
        # whether or not the optional ib extra is installed.
        monkeypatch.setattr("hanoon_prime._protect._ib", None)
        contract = SimpleNamespace()
        with pytest.raises(AttributeError):
            _place_oca(
                MagicMock(),
                contract,
                BracketOrder("SELL", 100, 95.0, 110.0, "JULI_TSLA"),
            )

    def test_protect_adopts_position(self, monkeypatch):
        # Inject a fake ib_insync-compatible module so _place_oca can build orders.
        fake_ib = SimpleNamespace(
            Order=lambda **kw: SimpleNamespace(**kw),
            Stock=lambda sym, sec, cur: f"{sym}/{sec}",
        )
        monkeypatch.setattr("hanoon_prime._protect._ib", fake_ib)

        ib_client = MagicMock()
        pos = SimpleNamespace(contract=SimpleNamespace(symbol="TSLA"), position=100)
        ib_client.positions.return_value = [pos]
        ib_client.openTrades.return_value = []
        ib_client.qualifyContracts.return_value = [SimpleNamespace()]

        streamer = SimpleNamespace(
            get_last_price=lambda s: 100.0,
            buffer_atr=lambda s: 2.0,
        )
        brackets: dict[str, tuple[float, float]] = {}
        protect_position(ib_client, {"TSLA"}, brackets, set(), streamer)
        assert "TSLA" in brackets
        assert brackets["TSLA"][0] < 100.0 < brackets["TSLA"][1]
        ib_client.placeOrder.assert_called()

    def test_protect_skips_no_data(self, monkeypatch):
        monkeypatch.setattr(
            "hanoon_prime._protect._ib",
            SimpleNamespace(Order=lambda **k: None, Stock=lambda *a: "c"),
        )
        ib_client = MagicMock()
        ib_client.positions.return_value = [
            SimpleNamespace(contract=SimpleNamespace(symbol="TSLA"), position=100)
        ]
        ib_client.openTrades.return_value = []
        streamer = SimpleNamespace(
            get_last_price=lambda s: None, buffer_atr=lambda s: 2.0
        )
        protect_position(ib_client, {"TSLA"}, {}, set(), streamer)
        # no price -> cannot place protection
        ib_client.placeOrder.assert_not_called()
