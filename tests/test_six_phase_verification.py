"""tests/test_six_phase_verification — the 6-Phase live-readiness verification.

Executes the phased verification contract before live capital:

  Phase 1  Macro→micro integration: S2 state shift → bounded modifier →
           S1 blend from shared memory, zero HTTP on the fast path.
  Phase 2  System 1 sanity: negative-score gate, ATR-relative brackets,
           orphan-order cleanup, Meta-DNN defect fallback.
  Phase 3  System 2 resilience: dead :8765 → bounded fallback modifier,
           warning logged, 1,000 fast ticks keep processing.
  Phase 4  Telemetry: one snapshot frame carries fast-path metrics AND
           slow-path brain states (regime/modifiers/CoT carrier).
  Phase 5  Forward-test gauges: the instruments the 24h paper soak reads.
  Live     Halim :8765 health + inference-cycle probes (skip when down).

Phase 6 (readiness verdict) is reported, not tested.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.request
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hanoon_prime.brain.config import HALIM_MOD_BOUND
from hanoon_prime.brain.consolidation import CYCLE_INTERVAL, ConsolidationEngine
from hanoon_prime.brain.meta_label import MetaLabelModel
from hanoon_prime.brain.meta_label_dnn import MetaDNN, expand_features
from hanoon_prime.brain.orchestrator import NeuromorphicBrain
from hanoon_prime.brain.policy.verdict import ENTER, HOLD, Verdict
from hanoon_prime.brain.shared_state import DEFAULT_POLICY_STATE, BrainState
from hanoon_prime.ib_order_sweep import sweep_zombies
from hanoon_prime.juli_feed import check_tick_latency
from hanoon_prime.monitor.sleep_manager import SleepManager

# ── harness ──────────────────────────────────────────────────────────────


def _snap(price: float = 100.0) -> dict:
    return {
        "prices": [price] * 40,
        "bid": price - 0.01,
        "ask": price + 0.01,
        "mid": price,
        "last": price,
        "ts": time.time(),
        "volume": 10_000,
        "atr": 2.0,
    }


def _brain(tick_result: dict | None = None) -> NeuromorphicBrain:
    """Brain with a stubbed cortex tick (real gates/sizing/brackets below it)."""
    b = NeuromorphicBrain(enable_neuromorphic=False)
    if tick_result is not None:
        b.tick = lambda alpha, ticker, **kw: tick_result  # type: ignore[method-assign]
    b.state.update(
        policy_state={
            **DEFAULT_POLICY_STATE,
            "equity_synced": True,
            "equity": 100_000.0,
        },
        account_feed={"equity": 100_000.0, "daily_pnl": 0.0, "positions": {}},
    )
    return b


def _entered_tick(score: float = 0.9, direction: int = 1) -> dict:
    return {
        "verdict": "BUY" if direction > 0 else "SELL",
        "thought": SimpleNamespace(score=score, direction=direction, confidence=0.6),
    }


def _skip_if_eod() -> None:
    """Real-tick score assertions are void inside the EOD flatten window."""
    remaining = SleepManager().minutes_to_close()
    if 0 < remaining <= 15:
        pytest.skip("EOD flatten window — scores zeroed by design")


def _spot(sym, order_type, qty, action="SELL", status="Submitted", oca="", parent=0):
    return SimpleNamespace(
        order=SimpleNamespace(
            orderType=order_type,
            action=action,
            totalQuantity=qty,
            ocaGroup=oca,
            parentId=parent,
        ),
        orderStatus=SimpleNamespace(status=status),
        contract=SimpleNamespace(symbol=sym),
    )


def _pos(symbol: str, size: float):
    return SimpleNamespace(contract=SimpleNamespace(symbol=symbol), position=size)


def _make_executor(tracked=None):
    from hanoon_prime.ib_executor import IBExecutor

    kwargs = {"tracked_tickers": set(tracked)} if tracked else {}
    return IBExecutor(MagicMock(), MagicMock(), MagicMock(), **kwargs)


def _engine(
    halim_url: str = "http://127.0.0.1:1",
) -> tuple[ConsolidationEngine, BrainState]:
    state = BrainState()
    eng = ConsolidationEngine(state, halim_url=halim_url)
    state.set_latest_alpha({"spy_momentum": 0.05})
    return eng, state


# ════════════════════════════════════════════════════════════════════════
# Phase 1 — Macro-to-Micro Integration (S2 → shared state → S1 blend)
# ════════════════════════════════════════════════════════════════════════


class TestPhase1Integration:
    def test_s2_bounds_out_of_band_modifier_before_shared_state(self):
        """A raw HALIM modifier (observed 0.8 live historically) is clamped
        to ±HALIM_MOD_BOUND BEFORE it enters shared state."""
        for raw, want in ((0.8, 0.03), (-0.9, -0.03), (0.02, 0.02)):
            eng, state = _engine()
            eng.halim.get_modifier = lambda *a, _r=raw, **k: _r  # type: ignore[method-assign]
            eng._update_halim()
            assert state.get("halim_modifier") == pytest.approx(want)
            assert abs(state.get("halim_modifier")) <= HALIM_MOD_BOUND

    def test_s2_regime_shift_published_to_shared_state(self):
        eng, state = _engine()
        eng.halim.get_regime = lambda *a, **k: {  # type: ignore[method-assign]
            "regime": "volatile",
            "multiplier": 1.4,
            "confidence": 0.9,
            "risk_adjustment": "defensive",
            "key_drivers": ["vpin_spike"],
            "description": "breakdown forming",
        }
        eng._update_regime()
        assert state.get("regime_label") == "volatile"
        assert state.get("regime_multiplier") == pytest.approx(1.4)

    def test_local_regime_fallback_when_halim_label_unknown(self):
        """S2 falls back to the local numpy detector so the label never
        sticks at 'unknown' when HALIM is unreachable."""
        eng, state = _engine()
        state.set_latest_prices([100 + i * 0.5 for i in range(30)])
        eng._local_regime()
        assert state.get("regime_label") == "trending_bullish"
        assert state.get("regime_source") == "local_fallback"

    def test_post_trade_cot_lands_in_state(self, monkeypatch):
        """Post-trade CoT: HALIM postmortem insight is published to state
        (the brain_state_CoT carrier telemetry reads)."""
        eng, state = _engine()
        eng.halim.analyze_trade = lambda td: {  # type: ignore[method-assign]
            "insight": "High toxic flow via VPIN. Scaling conviction down."
        }
        eng._halim_postmortem("TSLA", False, -0.02, 1, {"vpin": 0.8})
        insight = state.get("halim_last_insight", {})
        assert "toxic flow" in insight.get("insight", "")

    def test_fast_path_never_performs_http(self, monkeypatch):
        """S1 reads the latest state snapshot from shared memory; any
        network call inside decide_entry fails the test."""

        def boom(*a, **k):
            raise AssertionError("fast path must not touch the network")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        with patch("hanoon_prime.brain.orchestrator.META_DNN_ENABLED", False):
            b = _brain()
            v = b.decide_entry("TST", _snap(), {}, "rth")
        assert isinstance(v, Verdict)

    def test_halim_modifier_blends_into_score_within_bound(self):
        """Final score moves with halim_modifier, ordered by sign, and the
        delta equals the bounded contribution (±0.03 → 0.06 spread)."""
        _skip_if_eod()
        alpha = {"spy_momentum": 0.02, "vpin": 0.5}
        b1, b2 = NeuromorphicBrain(enable_neuromorphic=False), NeuromorphicBrain(
            enable_neuromorphic=False
        )
        b1.state.update(halim_modifier=0.03)
        b2.state.update(halim_modifier=-0.03)
        r1 = b1.tick(alpha, "TST", entry_price=100.0, atr=2.0, open_positions=0)
        r2 = b2.tick(alpha, "TST", entry_price=100.0, atr=2.0, open_positions=0)
        assert r1["trace"]["halim"] == pytest.approx(0.03)
        assert r2["trace"]["halim"] == pytest.approx(-0.03)
        assert r1["score"] > r2["score"]
        assert (r1["score"] - r2["score"]) == pytest.approx(0.06, abs=1e-6)

    def test_unbounded_state_modifier_clamped_at_fast_path_read(self):
        """Defense in depth: even a corrupted shared state cannot push the
        fast-path contribution past ±0.03."""
        b = NeuromorphicBrain(enable_neuromorphic=False)
        b.state.update(halim_modifier=0.8)
        r = b.tick({}, "TST", entry_price=100.0, atr=2.0, open_positions=0)
        assert r["trace"]["halim"] == pytest.approx(HALIM_MOD_BOUND)

    def test_live_halim_endpoint_alive(self):
        """Phase 1 step 1: MLX Metal endpoint alive on :8765."""
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8765/health", timeout=2
            ) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"HALIM serve not running on :8765: {exc}")
        assert body.get("ok") is True


# ════════════════════════════════════════════════════════════════════════
# Phase 2 — System 1 Sanity & Hardening
# ════════════════════════════════════════════════════════════════════════


class TestPhase2System1Sanity:
    @patch("hanoon_prime.brain.orchestrator.META_DNN_ENABLED", False)
    @pytest.mark.parametrize("score", [-0.9, -0.5, -0.05])
    def test_negative_score_long_is_hold_never_enter(self, score: float):
        """A long conviction below zero has no conviction: HOLD, never ENTER."""
        b = _brain(_entered_tick(score=score, direction=1))
        v = b.decide_entry("TST", _snap(), {}, "rth")
        assert v.action == HOLD
        assert v.reason == "no_signal"
        assert v.action != ENTER

    @patch("hanoon_prime.brain.orchestrator.META_DNN_ENABLED", False)
    def test_positive_score_long_enters_with_bracket(self):
        """Control for the negative-score test: the same harness with a
        positive score DOES enter, with ATR-relative positive brackets."""
        b = _brain(_entered_tick(score=0.9, direction=1))
        v = b.decide_entry("TST", _snap(price=100.0), {}, "rth")
        assert v.action == ENTER
        stop, target = float(v.stop), float(v.target)
        assert stop > 0 and target > 0
        assert stop < 100.0 < target
        # scalp horizon: 2×ATR stop / 6×ATR target → 3:1 R:R (d* scales both)
        assert (target - 100.0) / (100.0 - stop) == pytest.approx(3.0, abs=0.05)

    def test_only_enter_verdicts_reach_the_broker(self):
        """Zero broker orders for non-ENTER verdicts (the execution gate)."""
        from hanoon_prime.ib_cycle import BotCycleMixin

        cyc = SimpleNamespace(
            journal=MagicMock(),
            monitor=MagicMock(),
            _execute_verdict=MagicMock(),
        )
        cyc.monitor.bar_feed_fresh.return_value = True
        non_enter = [
            Verdict(ticker=f"T{i}", action=HOLD, reason="no_signal", stage="pipeline")
            for i in range(3)
        ]
        BotCycleMixin._execute_entries(cyc, True, non_enter)  # type: ignore[arg-type]
        cyc._execute_verdict.assert_not_called()
        assert cyc.journal.append.call_count == 3  # verdicts still journaled

        entered = Verdict(ticker="OK", action=ENTER, reason="admitted", stage="entry")
        BotCycleMixin._execute_entries(cyc, True, [entered])  # type: ignore[arg-type]
        cyc._execute_verdict.assert_called_once()

        cyc._execute_verdict.reset_mock()
        BotCycleMixin._execute_entries(cyc, False, [entered])  # type: ignore[arg-type]
        cyc._execute_verdict.assert_not_called()  # market closed → suppressed

    def test_place_bracket_aborts_on_invalid_atr(self):
        exc = _make_executor(tracked={"TSLA"})
        streamer = MagicMock()
        streamer.buffer_atr.return_value = 0.0
        streamer.contracts = {"TSLA": MagicMock()}
        exc.place_bracket(
            "TSLA", SimpleNamespace(direction=1, score=0.7), 150.0, streamer
        )
        exc.ib.placeOrder.assert_not_called()
        assert "TSLA" not in exc._brackets

    def test_place_bracket_rejects_nan_stop_target_before_orders(self):
        """NaN stop/target must never reach IB as 'Limit Price=nan'."""
        from hanoon_prime.brain.risk import SizingResult

        exc = _make_executor(tracked={"TSLA"})
        streamer = MagicMock()
        streamer.buffer_atr.return_value = 2.0
        streamer.contracts = {"TSLA": MagicMock()}
        sizing = SizingResult(
            shares=10, stop_price=float("nan"), target_price=110.0, risk_pass=True
        )
        exc.place_bracket(
            "TSLA",
            SimpleNamespace(direction=1, score=0.7),
            150.0,
            streamer,
            sizing=sizing,
        )
        exc.ib.placeOrder.assert_not_called()
        exc.ib.bracketOrder.assert_not_called()
        assert "TSLA" not in exc._brackets
        assert "TSLA" not in exc._pending_parent

    def test_orphan_bracket_children_cancelled_when_parent_never_fills(self):
        """Partial-fill/rejected parent → its resting children are orphans
        on a flat symbol and must be cancelled immediately."""
        ib = MagicMock()
        orphan = _spot("FLATCO", "LMT", 100, action="SELL", parent=3)
        held = _spot("HELDCO", "LMT", 100, action="SELL", parent=4)
        ib.openTrades.return_value = [orphan, held]
        ib.positions.return_value = [_pos("HELDCO", 10.0)]
        sweep_zombies(ib)
        cancelled = [c.args[0] for c in ib.cancelOrder.call_args_list]
        assert cancelled == [orphan.order]

    def test_cancel_all_clears_all_in_memory_order_state(self):
        exc = _make_executor(tracked={"TSLA"})
        exc._brackets["TSLA"] = (95.0, 110.0)
        exc._pending_parent.add("TSLA")
        exc.cancel_all()
        exc.ib.cancelAllOrders.assert_called_once()
        assert exc._brackets == {}
        assert exc._pending_parent == set()

    def test_corrupt_dnn_artifact_degrades_without_crash(self, tmp_path, monkeypatch):
        """Corrupt Meta-DNN weight file → gate degrades, never raises."""
        p = tmp_path / "dnn.json"
        p.write_text("{definitely not json!!")
        dnn = MetaDNN(path=p)  # must not raise
        admit, p_win, _scale = dnn.infer(
            expand_features(0.8, 0.9, 0.5, 1, 0.02, 0.1, 0.3)
        )
        assert admit is True
        assert 0.0 <= p_win <= 1.0

        model = MetaLabelModel(path=tmp_path / "meta.json")
        monkeypatch.setattr("hanoon_prime.brain.meta_label.META_DNN_ENABLED", True)
        monkeypatch.setattr("hanoon_prime.brain.meta_label._get_dnn", lambda: dnn)
        admit2, _p2, _s2 = model.gate(
            0.8,
            0.9,
            0.5,
            "trend_up",
            "scalp",
            direction=1,
            atr_ratio=0.02,
            obi=0.1,
            vpin=0.3,
        )
        assert admit2 is True  # defect tolerated: admit at threshold, no veto-cascade

    def test_dnn_exception_falls_back_to_rule_heuristics(self, tmp_path, monkeypatch):
        """Weights unreadable at call time → shallow rule-based gate, no crash."""

        def _boom():
            raise RuntimeError("weights gone")

        model = MetaLabelModel(path=tmp_path / "meta.json")
        monkeypatch.setattr("hanoon_prime.brain.meta_label.META_DNN_ENABLED", True)
        monkeypatch.setattr("hanoon_prime.brain.meta_label._get_dnn", _boom)
        admit, p, scale = model.gate(
            0.8,
            0.9,
            0.5,
            "trend_up",
            "scalp",
            direction=1,
            atr_ratio=0.02,
            obi=0.1,
            vpin=0.3,
        )
        assert admit is True
        assert 0.0 <= p <= 1.0
        assert 0.0 <= scale <= 1.0


# ════════════════════════════════════════════════════════════════════════
# Phase 3 — System 2 Resilience & Fail-Safe
# ════════════════════════════════════════════════════════════════════════


class TestPhase3System2Resilience:
    def test_dead_halim_defaults_modifier_to_zero_without_raise(self):
        """Port 8765 blocked/stopped → modifier defaults to 0.000, no exception."""
        eng, state = _engine(halim_url="http://127.0.0.1:1")
        eng._update_halim()
        assert state.get("halim_modifier") == pytest.approx(0.0)

    def test_last_valid_modifier_survives_within_cache_ttl(self):
        """Fallback branch: a cached modifier from before the outage is
        still served within its TTL instead of snapping to zero."""
        eng, state = _engine(halim_url="http://127.0.0.1:1")
        eng.halim._cache["spy_momentum"] = {"modifier": -0.025, "ts": time.time()}
        eng._update_halim()
        assert state.get("halim_modifier") == pytest.approx(-0.025)

    def test_dead_halim_logs_warning_without_exception(self, caplog):
        """Fail-safe visibility: a System 2 outage emits a WARNING, and the
        cycle method returns normally (tick processing is never stopped)."""
        eng, state = _engine(halim_url="http://127.0.0.1:1")
        with caplog.at_level(
            logging.WARNING, logger="hanoon_prime.brain.halim_adapter"
        ):
            eng._update_halim()
        assert any("HALIM query failed" in r.message for r in caplog.records)

    def test_dead_regime_query_falls_back_to_mechanical_heuristics(self):
        eng, state = _engine(halim_url="http://127.0.0.1:1")
        regime = eng.halim.get_regime({"adx": 30.0, "momentum": -0.05}, [100.0] * 30)
        assert isinstance(regime, dict)
        assert regime["regime"] in {"bear", "bull", "range", "normal"} or regime.get(
            "description", ""
        ).startswith("Mechanical fallback")

    def test_thousand_ticks_with_system2_dead(self):
        """1,000 fast-path events while :8765 is unreachable: every tick
        processes, the last-valid modifier is preserved, and latency stays
        inside the fast-path budget (strict <1ms verified out-of-band; the
        in-suite budget accounts for coverage instrumentation)."""
        b = NeuromorphicBrain(enable_neuromorphic=False)
        b.state.update(halim_modifier=-0.025)  # last valid value pre-outage
        alpha = {"spy_momentum": 0.01}
        lat: list[float] = []
        for _ in range(1000):
            t0 = time.perf_counter_ns()
            r = b.tick(alpha, "SPY", entry_price=100.0, atr=2.0, open_positions=0)
            lat.append((time.perf_counter_ns() - t0) / 1000.0)
            assert r["trace"]["halim"] == pytest.approx(-0.025)
        lat.sort()
        budget_us = 5_000.0 if "coverage" in sys.modules else 1_000.0
        assert lat[len(lat) // 2] < budget_us, f"p50={lat[len(lat)//2]:.0f}us"
        assert lat[int(len(lat) * 0.99)] < 25_000.0  # production stall threshold
        assert b.state.get("halim_modifier") == pytest.approx(-0.025)


# ════════════════════════════════════════════════════════════════════════
# Phase 4 — Telemetry & Dashboard Payload
# ════════════════════════════════════════════════════════════════════════


class _StubIB:
    def isConnected(self):
        return True

    @property
    def tickers(self):
        t = SimpleNamespace(
            contract=SimpleNamespace(symbol="SPY"),
            last=0.0,
            close=0.0,
            bid=0.0,
            ask=0.0,
        )
        return [t]

    def positions(self):
        p = SimpleNamespace()
        p.contract = SimpleNamespace(symbol="SPY")
        p.position = 100.0
        p.avgCost = 500.0
        return [p]

    def portfolio(self):
        item = SimpleNamespace()
        item.contract = SimpleNamespace(symbol="SPY")
        item.position = 100.0
        item.marketPrice = 505.0
        item.unrealizedPNL = 500.0
        item.marketValue = 50500.0
        return [item]


class TestPhase4Telemetry:
    def test_snapshot_frame_carries_fast_and_slow_brain_state(
        self, tmp_path, monkeypatch
    ):
        from hanoon_prime.brain import meta_label as _ml
        from hanoon_prime.telemetry import SNAPSHOT_INTERVAL, TelemetryAPI

        # hermetic DNN singleton (never read production artifacts from tests)
        monkeypatch.setattr(_ml, "_DNN_INSTANCE", MetaDNN(path=tmp_path / "dnn.json"))

        brain = NeuromorphicBrain(enable_neuromorphic=False)
        brain.state.update(
            tick_latency_us=420.0,
            halim_modifier=-0.025,
            thinker_modifier=0.0,
            regime_multiplier=1.0,
            regime_label="volatile",
            halim_last_insight={
                "insight": "Toxic flow via VPIN; scale conviction down."
            },
        )
        bot = SimpleNamespace(
            ib=_StubIB(),
            journal=SimpleNamespace(count=lambda: 42),
            hippocampus=SimpleNamespace(safety_enabled=True, _daily_pnl=-10.0),
            _halted=False,
            _last_beat=1.0,
            streamer=SimpleNamespace(ticker_subs={"SPY": object()}),
            monitor=SimpleNamespace(
                snapshot=lambda: {"healthy": True, "cycle_lag_s": 0.2}
            ),
            juli=SimpleNamespace(brain=brain, _recent_verdicts=None),
        )
        jp = tmp_path / "journal.jsonl"
        jp.write_text('{"event": "position_closed", "pnl": 1.0}\n')
        api = TelemetryAPI(bot, jp)
        api.start(port=0)
        try:
            time.sleep(SNAPSHOT_INTERVAL + 0.5)
            port = api._server.server_address[1]
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/snapshot", timeout=5
            ) as r:
                snap = json.loads(r.read().decode())
        finally:
            api.stop()

        s2 = snap["system2"]
        assert s2["halim_modifier"] == pytest.approx(-0.025)
        assert s2["thinker_modifier"] == pytest.approx(0.0)
        assert s2["regime_multiplier"] == pytest.approx(1.0)
        assert s2["regime_label"] == "volatile"

        bs = snap["brain"]["brain_state"]
        assert bs["tick_latency_us"] == pytest.approx(420.0)  # fast-path metric
        assert "last_p_win" in snap["brain"]["meta_label_dnn"]  # p_win carrier

        hal = snap["halim"]
        assert hal["halim_modifier"] == pytest.approx(-0.025)
        assert "toxic flow" in hal["halim_last_insight"]["insight"].lower()

        assert snap["meta"]["built_ts"] > 0


# ════════════════════════════════════════════════════════════════════════
# Phase 5 — Forward-Test (24h paper soak) instrumentation
# ════════════════════════════════════════════════════════════════════════


class TestPhase5SoakGauges:
    def test_tick_latency_gauge_returns_microseconds(self):
        t0 = time.perf_counter_ns() - 1_000_000  # ~1ms already elapsed
        us = check_tick_latency(t0, "TST")
        assert isinstance(us, float)
        assert 500.0 < us < 25_000.0  # measured, and under the warn threshold

    def test_system2_cycle_period_is_30s(self):
        assert CYCLE_INTERVAL == 30.0

    def test_execution_quality_gauges_present(self):
        from hanoon_prime.monitor.exec_quality import ExecQuality

        summary = ExecQuality().summary()
        assert "fills" in summary
        assert "slippage_bps_mean" in summary

    def test_live_halim_inference_within_cycle_budget(self):
        """Phase 5 metric: one S2 inference must finish well inside 15s."""
        payload = json.dumps(
            {"prompt": 'Reply with exactly: {"ok": 1}', "purpose": "reasoning"}
        ).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8765/v1/complete",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"HALIM serve not running on :8765: {exc}")
        elapsed = time.time() - t0
        assert elapsed < 15.0, f"S2 inference took {elapsed:.2f}s (budget 15s)"
        assert body.get("ok") is True
