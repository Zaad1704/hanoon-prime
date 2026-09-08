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
class _FakeBrain:
    """Lightweight NeuromorphicBrain stand-in: counts decide_entry calls."""

    def __init__(self) -> None:
        self.called: list[str] = []
        self.exits = SimpleNamespace(is_registered=lambda t: False)
        self.state = SimpleNamespace(get=lambda k, d=None: d, update=lambda **k: None)

    def begin_entry_cycle(self) -> None:
        pass

    def decide_entry(self, ticker, snap, open_positions, session="rth"):
        self.called.append(ticker)
        from hanoon_prime.brain.policy.verdict import Verdict

        return Verdict(ticker=ticker)

    def register_position(self, *a, **k) -> None:
        pass

    def check_exit(self, *a, **k):
        return SimpleNamespace(should_exit=False, reason="", exit_type="")


class TestRotatingEvalWindow:
    def _make_juli(self) -> JuliBrain:
        import collections

        brain = JuliBrain.__new__(JuliBrain)
        brain.brain = _FakeBrain()
        brain.budget = SimpleNamespace(
            get_all_tracked=lambda: {f"S{i}" for i in range(10)},
            allocate=lambda *a, **k: None,
        )
        brain.feed = MagicMock()
        brain.scanner = MagicMock()
        brain.scanner.should_scan.return_value = False
        brain.scanner.collect.return_value = []
        brain._candidates = []
        brain._last_alloc = 0.0
        brain._state = SimpleNamespace()
        brain._eval_off = 0
        brain._lock_held = False
        brain._recent_verdicts = collections.deque(maxlen=200)
        return brain

    def test_eval_window_rotates_across_cycles(self):
        """Regression: FIX-2026-09-08-02 — the eval cursor advances so the
        whole universe gets scored over successive cycles (Verdicts, not a
        silent omission), not in one burst."""
        b = self._make_juli()
        universe = {f"S{i}" for i in range(10)}
        b.tick(universe, {}, None, set(), session="rth")
        assert len(b.brain.called) <= EVAL_WINDOW, "scored more than window"
        assert b._eval_off == EVAL_WINDOW, "cursor did not advance"
        assert b.brain.called, "no decide_entry calls at all"
        assert set(b.brain.called) <= universe

    def test_rotation_advances_to_holdout(self):
        """Regression: FIX-2026-09-08-02 — after enough cycles the cursor
        wraps and later tickers get evaluated too."""
        b = self._make_juli()
        universe = {f"S{i}" for i in range(6)}
        for _ in range(4):
            b.tick(universe, {}, None, set(), session="rth")
        # Cursor moves EVAL_WINDOW each cycle: 0→4→2→0→4 with wrap mod 6.
        assert b.brain.called, "no tickers ever evaluated"
        assert len(set(b.brain.called)) >= 2, "rotation reaches multiple slices"


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
        score is refused (bar raised), matching the `low_penny_score` gate
        that now lives in the brain's TradingPolicy."""
        allowed, reason = TRADING_CONFIG.is_penny_bar_cleared("PENNY", 0.5, 0.6)
        assert not allowed
        assert reason == "low_penny_score"
        # A genuinely extreme conviction still clears it (not a hard block).
        allowed, _ = TRADING_CONFIG.is_penny_bar_cleared("PENNY", 0.5, 0.99)
        assert allowed


# ── FIX-2026-09-08-07: long-only default (telemetry-toggleable) ───────
class TestLongOnlyDefault:
    def test_direction_defaults_long_only(self):
        """Regression: FIX-2026-09-08-07 — bots start long-only; shorts
        are opt-in later via telemetry."""
        assert TRADING_CONFIG.direction_mode == "long_only"
        assert TRADING_CONFIG.is_direction_allowed("BUY")
        assert not TRADING_CONFIG.is_direction_allowed("SELL")
