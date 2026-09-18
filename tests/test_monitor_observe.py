"""Tests for the observe-only MonitorSuite (dormant monitors, read-only)."""

from __future__ import annotations

import time
from types import SimpleNamespace

from hanoon_prime.monitor.observe import MonitorSuite


class _Buf:
    def __init__(self, trades):
        self._t = trades

    def get_trades(self):
        return list(self._t)


def _trade(ticker: str, win: bool):
    return SimpleNamespace(ticker=ticker, win=win, pnl=1.0 if win else -1.0)


def _bot(trades, ticks, thoughts=None):
    cons = SimpleNamespace(buffer=_Buf(trades))
    brain = SimpleNamespace(_consolidation=cons)
    juli = SimpleNamespace(brain=brain)
    return SimpleNamespace(
        juli=juli,
        streamer=SimpleNamespace(last_data_ts=ticks),
        executor=SimpleNamespace(last_thoughts=thoughts or {}),
        _position_marks={"positions": []},
    )


def test_snapshot_shape_and_flags() -> None:
    suite = MonitorSuite(_bot([], {}), SimpleNamespace())
    snap = suite.snapshot()
    assert snap["enabled"] is True
    assert snap["mode"] == "observe-only"
    for key in (
        "watchdog",
        "decision_health",
        "enforcement",
        "health_budget",
        "exit_scorer",
        "reconciliation",
    ):
        assert key in snap


def test_pulse_feeds_all_monitors() -> None:
    now = time.time()
    trades = [_trade("BITO", True), _trade("GGB", False)]
    suite = MonitorSuite(
        _bot(trades, {"SPY": now, "QQQ": now - 999}), SimpleNamespace()
    )
    suite.pulse()
    snap = suite.snapshot()
    assert snap["watchdog"]["tickers_tracked"] == 1  # stale QQQ not fed
    assert snap["decision_health"]["n_decisions"] == 2
    assert snap["enforcement"]["score"] == 1.0
    assert len(snap["enforcement"]["checks"]) == 3
    assert snap["exit_scorer"]["mode"] == "observe"
    assert snap["reconciliation"]["seed_count"] == 0


def test_observe_only_never_starts_or_writes() -> None:
    state = SimpleNamespace()
    suite = MonitorSuite(_bot([_trade("SPY", True)], {"SPY": time.time()}), state)
    suite.pulse()
    # PositionMonitor's action loop is never started, so panic/auto-flatten
    # is unreachable and shared state is never written.
    assert suite._pm._thread is None
    assert suite.watchdog.snapshot()["panic_mode"] is False
    assert not hasattr(state, "panic_mode")


def test_start_seeds_existing_trades() -> None:
    suite = MonitorSuite(_bot([_trade("OLD", True)], {}), SimpleNamespace())
    suite.start()
    try:
        suite.pulse()
        assert suite.snapshot()["decision_health"]["n_decisions"] == 0
    finally:
        suite.stop()
