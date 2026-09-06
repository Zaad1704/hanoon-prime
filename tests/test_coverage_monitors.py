"""test_coverage_monitors — tests for previously-untested monitoring and
sensory-periphery modules: deliberation, amygdala, thalamus, eyes, indicators
integration, planning, the monitor daemons, reconciliation, enforcement,
and the telegram notifier.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from hanoon_prime._telegram import _429_COOLDOWN, send, trade_opened
from hanoon_prime.brain.amygdala import Amygdala
from hanoon_prime.brain.amygdala import MarketQuote as AmygQuote
from hanoon_prime.brain.cognitive.planning import PlanEngine
from hanoon_prime.brain.config import SIGNAL_THRESHOLD
from hanoon_prime.brain.deliberation import Deliberator, Modifiers
from hanoon_prime.brain.episodic import EpisodicMemory
from hanoon_prime.brain.indicators import CORE_NAMES, INDICATOR_NAMES, compute_all_alpha
from hanoon_prime.brain.memory import JuliMemory
from hanoon_prime.brain.reflection import Reflector, TradeClose
from hanoon_prime.brain.shared_state import BrainState
from hanoon_prime.brain.thalamus import MarketQuote as ThalQuote
from hanoon_prime.brain.thalamus import Thalamus
from hanoon_prime.eyes import (
    _is_header_row,
    _parse_csv_row,
    compute_buy_volume,
    compute_vwap,
    estimate_bid_ask,
    load_ohlcv,
    rolling_atr,
)
from hanoon_prime.monitor.decision_health import DecisionHealthTracker
from hanoon_prime.monitor.enforcement import (
    DeepEnforcement,
    HealthBudget,
    HealthDiagnosis,
)
from hanoon_prime.monitor.exit_scoring import ExitScorer
from hanoon_prime.monitor.portfolio_risk import PortfolioRiskManager
from hanoon_prime.monitor.reconciliation import Reconciliation
from hanoon_prime.monitor.watchdog import STALE_THRESHOLD, Watchdog
from hanoon_prime.types import BarSeries, ExitLevels

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market_data"


# ── Eyes ─────────────────────────────────────────────────────────────────
class TestEyes:
    def test_is_header_row(self):
        assert _is_header_row(["datetime", "close"]) is True
        assert _is_header_row(["ticker", "AAPL"]) is True
        assert _is_header_row(["nope"]) is True
        assert (
            _is_header_row(["2026-07-06 09:30:00", "100", "101", "99", "100", "1000"])
            is False
        )

    def test_parse_csv_row(self):
        row = ["2026-07-06 09:30:00", "100", "101", "99", "100", "1000"]
        assert _parse_csv_row(row) == (100.0, 101.0, 99.0, 100.0, 1000.0)

    def test_parse_bad_row(self):
        assert _parse_csv_row(["x"]) is None

    def test_load_ohlcv_and_indicators(self):
        path = DATA_DIR / "AAPL_1min.csv"
        if not path.exists():
            pytest.skip("no AAPL data")
        data = load_ohlcv(path)
        assert len(data["close"]) > 0
        c, h, l, v = data["close"], data["high"], data["low"], data["volume"]
        bv = compute_buy_volume(c, h, l, v)
        bid, ask = estimate_bid_ask(v, bv)
        assert len(bv) == len(c)
        assert len(bid) >= 1 and len(ask) >= 1
        assert rolling_atr(h, l, c, period=14) > 0
        assert compute_vwap(c, v) > 0


# ── Indicators integration (covers indicators_core + _tech + cerebellum) ─
class TestIndicators:
    def test_compute_all_alpha_keys(self):
        path = DATA_DIR / "AAPL_1min.csv"
        if not path.exists():
            pytest.skip("no AAPL data")
        data = load_ohlcv(path)
        bars = BarSeries(
            data["close"],
            data["high"],
            data["low"],
            data["volume"],
            compute_buy_volume(
                data["close"], data["high"], data["low"], data["volume"]
            ),
            None,
            None,
        )
        alpha = compute_all_alpha(bars)
        assert set(CORE_NAMES).issubset(alpha.keys())
        assert "volatility" in alpha
        assert len(alpha) >= len(INDICATOR_NAMES)

    def test_compute_all_alpha_constant_flat(self):
        c = np.full(60, 100.0)
        bars = BarSeries(c, c, c, c, c * 0.5, None, None)
        alpha = compute_all_alpha(bars)
        assert -1.0 <= alpha["momentum"] <= 1.0
        assert -1.0 <= alpha["vpin"] <= 1.0


# ── Deliberator ───────────────────────────────────────────────────────────
class TestDeliberator:
    def test_hold_when_beneath_threshold(self):
        d = Deliberator(threshold=0.6)
        result = d.deliberate(0.05, 0.5, Modifiers())
        assert result.verdict == "HOLD"
        assert result.direction == 0

    def test_buy_when_strong(self):
        d = Deliberator(threshold=SIGNAL_THRESHOLD)
        mods = Modifiers(episodic_mod=0.05, affective_mod=0.02)
        result = d.deliberate(0.9, 0.6, mods)
        assert result.verdict == "BUY"
        assert result.direction == 1
        assert result.score > 0

    def test_sell_on_negative(self):
        d = Deliberator(threshold=SIGNAL_THRESHOLD)
        result = d.deliberate(-0.9, 0.6, Modifiers())
        assert result.verdict == "SELL"
        assert result.direction == -1


# ── Amygdala ─────────────────────────────────────────────────────────────
class TestAmygdala:
    def test_normal_is_safe(self):
        a = Amygdala()
        q = AmygQuote(bid=100.0, ask=100.5, last=100.0, volume=1000)
        prices = [100.0] * 10
        t = a.evaluate("TSLA", q, atr=1.0, prices=prices)
        assert t.trigger_exit is False
        assert t.score > 0.5

    def test_spread_widen_triggers_exit(self):
        a = Amygdala()
        q = AmygQuote(bid=100.0, ask=105.0, last=100.0, volume=1000)
        t = a.evaluate("TSLA", q, atr=1.0)
        assert t.trigger_exit is True
        assert t.fear > 0


# ── Thalamus ────────────────────────────────────────────────────────────
class TestThalamus:
    def _mk(self, **kw):
        base = dict(
            bid=99.95, ask=100.05, last=100.0, volume=50000, daily_volume=2_000_000
        )
        base.update(kw)
        return ThalQuote(**base)

    def test_eligible(self):
        t = Thalamus().screen("TSLA", self._mk())
        assert t.eligible is True
        assert t.salience > 0

    def test_low_price_rejected(self):
        t = Thalamus().screen("TSLA", self._mk(last=3.0))
        assert t.eligible is False
        assert "price" in t.reason

    def test_low_volume_rejected(self):
        t = Thalamus().screen("TSLA", self._mk(daily_volume=1000))
        assert t.eligible is False
        assert "volume" in t.reason

    def test_rank_and_stale(self):
        th = Thalamus()
        v = th.screen("A", self._mk(last=100.0, volume=100000))
        v2 = th.screen("B", self._mk(last=100.0, volume=200000))
        ranked = th.rank([v, v2])
        assert ranked[0].salience >= ranked[-1].salience
        assert th.is_stale("C") is True  # never seen


# ── Planning (Monte Carlo) ──────────────────────────────────────────────
class TestPlanEngine:
    def test_flat_prices_zero_modifier(self):
        close = np.full(30, 100.0)
        high = close
        low = close
        assert PlanEngine().simulate(close, high, low, direction=1) == 0.0

    def test_varying_prices_bounded(self):
        close = np.linspace(100, 105, 30)
        high = close + 1
        low = close - 1
        mod = PlanEngine().simulate(close, high, low, direction=1)
        assert -0.03 <= mod <= 0.03


# ── Watchdog ────────────────────────────────────────────────────────────
class TestWatchdog:
    def test_no_panic_fresh(self):
        w = Watchdog()
        assert w.check_panic() is False
        assert w.check_stale() == []

    def test_stale_after_threshold(self):
        w = Watchdog()
        w.tick_received("TSLA")
        w._tick_times["TSLA"] = time.time() - STALE_THRESHOLD - 1.0
        assert w.check_stale() == ["TSLA"]

    def test_panic_then_clear(self):
        w = Watchdog()
        w.tick_received("TSLA")
        w._last_heartbeat = time.time() - 61.0
        assert w.check_panic() is True
        w.clear_panic()
        w._last_heartbeat = time.time()  # fresh heartbeat clears the trigger
        assert w.check_panic() is False
        assert w.snapshot()["panic_mode"] is False


# ── Reconciliation ────────────────────────────────────────────────────────
class TestReconciliation:
    def test_seeds_missing_position(self):
        rec = Reconciliation()
        exe = SimpleNamespace(last_thoughts={})
        pos = SimpleNamespace(avgCost=100.0, position=10)
        seeded = rec.reconcile({"TSLA": pos}, exe)
        assert seeded == ["TSLA"]
        assert rec.is_seeded("TSLA")
        assert exe.last_thoughts["TSLA"]["direction"] == 1

    def test_no_position_no_seed(self):
        rec = Reconciliation()
        exe = SimpleNamespace(last_thoughts={"TSLA": {"x": 1}})
        assert rec.reconcile({}, exe) == []

    def test_zero_price_not_seeded(self):
        rec = Reconciliation()
        exe = SimpleNamespace(last_thoughts={})
        pos = SimpleNamespace(avgCost=0.0, position=5)
        rec.reconcile({"TSLA": pos}, exe)
        assert rec.is_seeded("TSLA") is False
        rec.clear_seed("TSLA")


# ── Enforcement ──────────────────────────────────────────────────────────
class TestEnforcement:
    def test_run_pulse_rate_limited(self):
        eng = DeepEnforcement()
        first = eng.run_pulse({"TSLA": {"stale": False}}, {})
        assert first is not None
        second = eng.run_pulse({}, {})  # within 5s window -> rate limit
        assert second.checks == []
        assert second.score == 1.0

    def test_memory_health_warning(self):
        eng = DeepEnforcement()
        eng._last_run = 0.0  # force run
        mem = SimpleNamespace(snapshot=lambda: {"win_rate": 0.05})
        rep = eng.run_pulse({}, {}, memory=mem)
        assert rep.has_critical is False
        assert rep.score < 1.0

    def test_health_budget_and_diagnosis(self):
        hb = HealthBudget()
        hb.update("brain", 0.2)
        hb.update("eyes", 0.8)
        assert hb.get_overall() > 0.0
        assert "brain" in hb.get_critical_modules()
        issues = HealthDiagnosis().diagnose({"brain": 0.2, "eyes": 0.8})
        assert any("brain" in i for i in issues)


# ── Decision health ──────────────────────────────────────────────────────
class TestDecisionHealth:
    def test_empty_health(self):
        assert DecisionHealthTracker().get_health().overall_score == 1.0

    def test_with_entries_and_exits(self):
        dh = DecisionHealthTracker()
        dh.record_entry("TSLA", 0.7, True)
        dh.record_entry("TSLA", 0.3, False)
        dh.record_exit("TSLA", "stop", True)
        health = dh.get_health()
        assert health.n_decisions == 2
        assert len(health.issues) >= 0


# ── Portfolio risk ───────────────────────────────────────────────────────
class TestPortfolioRisk:
    def test_safe_market(self):
        mgr = PortfolioRiskManager()
        st = mgr.update(equity=100000.0, positions={"TSLA": 5000})
        assert st.blocked is False
        assert mgr.pre_trade_risk_gate() is True

    def test_max_positions_blocks(self):
        mgr = PortfolioRiskManager()
        mgr.update(100000.0, {"A": 1, "B": 2, "C": 3})
        st = mgr.update(100000.0, {"A": 1, "B": 2, "C": 3, "D": 4})
        assert st.blocked is True

    def test_drawdown_blocks(self):
        mgr = PortfolioRiskManager()
        mgr.update(100000.0, {"TSLA": 1000})
        st = mgr.update(89000.0, {"TSLA": 1000})  # -11% drawdown
        assert st.blocked is True


# ── Exit scoring ─────────────────────────────────────────────────────────
class TestExitScoring:
    def test_no_price(self):
        h = ExitScorer().score_exit(
            "TSLA", entry_price=0.0, current_price=100.0, direction=1
        )
        assert h.verdict == 0
        assert h.reason == "no_price"

    def test_stop_loss(self):
        h = ExitScorer().score_exit(
            "TSLA", entry_price=100.0, current_price=95.0, direction=1
        )
        assert h.verdict == 1
        assert h.reason == "stop_loss"

    def test_profit_lock_via_threshold(self):
        scorer = ExitScorer(threshold=0.9)
        h = scorer.score_exit(
            "TSLA", entry_price=100.0, current_price=106.0, direction=1
        )
        assert h.verdict == 1
        assert h.reason == "profit_lock"

    def test_health_collapse_via_threshold(self):
        scorer = ExitScorer(threshold=2.0)
        h = scorer.score_exit(
            "TSLA", entry_price=100.0, current_price=100.0, direction=1
        )
        assert h.verdict == 1
        assert h.reason == "health_collapse"

    def test_ok_with_bars(self):
        path = DATA_DIR / "AAPL_1min.csv"
        if not path.exists():
            pytest.skip("no AAPL data")
        data = load_ohlcv(path)
        c = data["close"]
        bars = BarSeries(
            c,
            data["high"],
            data["low"],
            data["volume"],
            compute_buy_volume(c, data["high"], data["low"], data["volume"]),
            None,
            None,
        )
        h = ExitScorer().score_exit(
            "AAPL",
            entry_price=float(c[-20]),
            current_price=float(c[-1]),
            direction=1,
            bars=bars,
        )
        assert h.verdict in (0, 1)
        assert 0.0 <= h.score <= 1.0


# ── Position monitor ─────────────────────────────────────────────────────
class TestPositionMonitor:
    def test_cycle_updates_state(self, monkeypatch):
        from hanoon_prime.monitor.position_monitor import PositionMonitor

        monkeypatch.setattr(
            "hanoon_prime.monitor.position_monitor.time.sleep", lambda *_: None
        )
        state = BrainState()
        pm = PositionMonitor(state, pulse_sec=0.01)
        pm._cycle()
        assert state.get("panic_mode") is False

    def test_start_stop(self, monkeypatch):
        from hanoon_prime.monitor.position_monitor import PositionMonitor

        monkeypatch.setattr(
            "hanoon_prime.monitor.position_monitor.time.sleep", lambda *_: None
        )
        pm = PositionMonitor(BrainState())
        pm.start()
        assert pm._running is True
        pm.stop()
        assert pm._running is False


# ── Telegram (no env / 429 / success) ───────────────────────────────────
class TestTelegram:
    def test_send_disabled_no_token(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TRADING_BOT_TELEGRAM_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("TRADING_BOT_TELEGRAM_CHAT_ID", raising=False)
        assert send("hello") is False

    def test_429_cooldown(self, monkeypatch):
        import hanoon_prime._telegram as tg

        monkeypatch.setattr("hanoon_prime._telegram.time.sleep", lambda *_: None)
        monkeypatch.setattr(tg, "_get_token", lambda: "token")
        monkeypatch.setattr(tg, "_get_chat_id", lambda: "chat")
        monkeypatch.setattr(tg, "_cooldown_until", time.time() + _429_COOLDOWN)
        tg._last_send = time.time()
        tg._bucket = {"count": 0.0, "window": time.time()}
        assert send("msg") is False

    def test_success(self, monkeypatch):
        import hanoon_prime._telegram as tg

        monkeypatch.setattr("hanoon_prime._telegram.time.sleep", lambda *_: None)
        monkeypatch.setattr(tg, "_get_token", lambda: "token")
        monkeypatch.setattr(tg, "_get_chat_id", lambda: "chat")
        monkeypatch.setattr(tg, "_cooldown_until", 0.0)
        tg._last_send = time.time()
        tg._bucket = {"count": 0.0, "window": time.time()}
        resp = MagicMock()
        monkeypatch.setattr(tg.urllib.request, "urlopen", lambda *a, **k: resp)
        assert send("ok") is True

    def test_trade_opened_no_env(self):
        levels = ExitLevels(stop=95.0, target=110.0)
        trade_opened("TSLA", "BUY", 100, 100.0, levels)  # must not raise


# ── Indicators smoke for deliberator/reflectors ──────────────────────────
class TestBrainSmoke:
    def test_deliberate_and_reflect(self, tmp_path):
        path = DATA_DIR / "AAPL_1min.csv"
        if not path.exists():
            pytest.skip("no AAPL data")
        data = load_ohlcv(path)
        c = data["close"]
        bv = compute_buy_volume(c, data["high"], data["low"], data["volume"])
        bars = BarSeries(c, data["high"], data["low"], data["volume"], bv, None, None)
        alpha = compute_all_alpha(bars)
        d = Deliberator()
        res = d.deliberate(0.6, 0.6, Modifiers())
        assert res.verdict in ("BUY", "SELL", "HOLD")
        mem = JuliMemory(path=tmp_path / "state.json")
        refl = Reflector(mem, EpisodicMemory())
        refl.on_trade_close(
            TradeClose(
                ticker="AAPL",
                won=True,
                pnl_pct=0.06,
                direction=1,
                alpha=alpha,
                predicted_score=res.score,
            )
        )
        assert mem.total_trades == 1
