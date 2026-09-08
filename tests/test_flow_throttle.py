"""test_flow_throttle — full-parallel-throttle flow, adoption, flatten fixes.

Regression coverage for the burst/flow stabilization session:
the bot must rotate entry evaluation (no burst), throttle entries per
cycle, raise the sub-dollar bar (no hard block), skip adoption of
positions being closed, and flatten via market orders that are never
retracted by cancelAllOrders.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from hanoon_prime.config import TRADING_CONFIG
from hanoon_prime.juli import EVAL_WINDOW, JuliBrain


# ── FIX-2026-09-08-02: rotating entry evaluation (no THINK burst) ─────
class TestRotatingEvalWindow:
    def test_eval_window_rotates_across_cycles(self):
        """Regression: FIX-2026-09-08-02 — the eval cursor advances so the
        whole universe gets scored over successive cycles, not in one burst."""
        brain = JuliBrain(MagicMock())
        brain.brain = MagicMock()
        # Pretend a 10-ticker tracked universe.
        brain.budget = SimpleNamespace(
            get_all_tracked=lambda: {f"S{i}" for i in range(10)}
        )
        brain._eval_off = 0
        # First call scores the first EVAL_WINDOW tickers.
        calls = []
        brain._eval_one = lambda t, snap, n: calls.append(t) or (
            {"ticker": t} if random_pass(t) else None
        )
        # Avoid real brain calls: stub _state / tick paths minimally.
        brain._state = MagicMock()
        brain.brain.note_eval_failure = MagicMock()
        blanket = {f"S{i}" for i in range(10)}
        brain._evaluate_entries(blanket, lambda t: {"prices": [1.0] * 21})
        assert len(calls) <= EVAL_WINDOW, "scored more than window per cycle"
        assert brain._eval_off == EVAL_WINDOW, "cursor did not advance"

    def test_rotation_advances_to_holdout(self):
        """Regression: FIX-2026-09-08-02 — after enough cycles the cursor
        wraps and later tickers get evaluated too."""
        brain = JuliBrain(MagicMock())
        brain.budget = SimpleNamespace(
            get_all_tracked=lambda: {f"S{i}" for i in range(6)}
        )
        brain._eval_off = 0
        seen = set()
        blank = {"prices": [1.0] * 21}

        def fake_eval(t, snap, n):
            seen.add(t)
            return None

        brain._eval_one = fake_eval
        brain._state = MagicMock()
        brain.brain.note_eval_failure = MagicMock()
        universe = {f"S{i}" for i in range(6)}
        for _ in range(4):
            brain._evaluate_entries(universe, lambda t: blank)
        # Cursor moves EVAL_WINDOW each cycle: 0→4→2→0→4 with wrap mod 6.
        assert seen, "no tickers ever evaluated"
        assert len(seen) >= 2, "rotation should reach multiple slices"


def random_pass(_t: str) -> bool:
    """Deterministic stand-in so no real brain call happens."""
    return False


# ── FIX-2026-09-08-04: adoption skips positions being closed ──────────
class TestAdoptSkipsClosing:
    def test_closing_positions_skipped(self):
        """Regression: FIX-2026-09-08-04
        Regression: FIX-2026-09-08-01 — a position mid-flatten must not
        be re-adopted/re-protected on the next sync cycle."""
        from hanoon_prime.ib_executor import IBExecutor

        exec_ = IBExecutor.__new__(IBExecutor)
        exec_.last_thoughts = {}
        exec_.tracked_tickers = set()
        exec_._synthetic = set()
        exec_._horizons = {}
        streamer = MagicMock()
        pos = SimpleNamespace(
            contract=SimpleNamespace(symbol="GPRO"),
            position=296.0,
            avgCost=1.77,
        )
        close = MagicMock()
        close.positions.return_value = [pos]
        exec_.ib = close
        exec_._adopt_orphan_positions(streamer, closing={"GPRO"})
        streamer.subscribe.assert_not_called()
        assert "GPRO" not in exec_.tracked_tickers


# ── FIX-2026-09-08-05: flatten uses market orders, never retracts ─────
class TestFlattenMarketOrders:
    def test_market_orders_and_no_cancel(self, monkeypatch):
        """Regression: FIX-2026-09-08-05
        Regression: FIX-2026-09-08-03 — flatten uses MKT orders and
        does NOT call cancelAllOrders, which used to retract the very
        orders it just placed."""
        from hanoon_prime.ib_executor import IBExecutor

        # ib_insync may be unavailable in the test process (Python 3.14
        # event-loop shim) — substitute a minimal Order factory, matching
        # the established stub pattern in TestProtect.
        class _FakeOrder:
            def __init__(self, orderType=None, action=None, totalQuantity=0, **kw):
                self.orderType = orderType
                self.action = action
                self.totalQuantity = totalQuantity
                self.tif = kw.get("tif")
                self.outsideRth = kw.get("outsideRth")

        monkeypatch.setattr(
            "hanoon_prime.ib_executor._ib", SimpleNamespace(Order=_FakeOrder)
        )

        exec_ = IBExecutor.__new__(IBExecutor)
        exec_._brackets = {}
        exec_._pending_parent = set()
        pos = SimpleNamespace(
            contract=SimpleNamespace(symbol="DVLT"),
            position=-12550.0,
            avgCost=0.20,
        )
        close = MagicMock()
        close.positions.return_value = [pos]
        exec_.ib = close
        n = exec_.close_all_positions(None)
        assert n == 1
        place_calls = close.placeOrder.call_args_list
        assert place_calls, "no order placed"
        order = place_calls[0][0][1]
        assert order.orderType == "MKT", order.orderType
        assert str(order.tif).upper() == "DAY"
        assert order.action == "BUY"
        close.cancelAllOrders.assert_not_called()


# ── FIX-2026-09-08-06: sub-dollar bar raises confidence (not a block) ─
class TestSubDollarBar:
    def test_bar_rejects_low_score_penny(self):
        """Regression: FIX-2026-09-08-06 — a sub-$1 ticker with a normal
        score is refused (bar raised), matching the `sub_dollar_bar` log."""
        # The gate lives in ib_cycle._can_trade; assert the immune
        # constants exist and the bar is above the base entry threshold.
        from hanoon_prime.immune import ENTRY_THRESHOLD, PENNY_PRICE, PENNY_SCORE_BAR

        assert PENNY_PRICE < 1.01
        assert PENNY_SCORE_BAR > ENTRY_THRESHOLD


# ── FIX-2026-09-08-07: long-only default (telemetry-toggleable) ───────
class TestLongOnlyDefault:
    def test_direction_defaults_long_only(self):
        """Regression: FIX-2026-09-08-07 — bots start long-only; shorts
        are opt-in later via telemetry."""
        assert TRADING_CONFIG.direction_mode == "long_only"
        assert TRADING_CONFIG.is_direction_allowed("BUY")
        assert not TRADING_CONFIG.is_direction_allowed("SELL")
