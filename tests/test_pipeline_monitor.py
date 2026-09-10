"""tests/test_pipeline_monitor.py — continuous pipeline health daemon.

Covers: vitals recording, staleness/stall detection, incident journaling,
heal-flag handoff, recovery transitions, and telemetry snapshot shape.
The monitor must be alert-only: it never places orders.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hanoon_prime.monitor.pipeline import (  # noqa: E402
    BAR_STALE_SECS,
    BRAIN_STALL_CYCLES,
    RESUB_MIN_SECS,
    PipelineMonitor,
)


class FakeJournal:
    """Minimal journal double with a real path for size probes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.events: list[dict] = []

    def append(self, event: dict) -> None:
        self.events.append(event)


def make_bot(tmp_path: Path, decisions: int = 0) -> SimpleNamespace:
    """A bot double with the attributes the monitor reads."""
    buf = SimpleNamespace(close=[1.0, 2.0, 3.0])
    return SimpleNamespace(
        streamer=SimpleNamespace(buffers={"AAPL": buf}),
        ib=SimpleNamespace(isConnected=lambda: True),
        juli=SimpleNamespace(brain=SimpleNamespace(_decision_count=decisions)),
        journal=FakeJournal(tmp_path / "j.json"),
    )


def test_record_cycle_publishes_vitals(tmp_path):
    """Vitals reflect connection, bar sizes, decisions, journal size."""
    bot = make_bot(tmp_path)
    mon = PipelineMonitor(bot, bot.journal)
    mon.record_cycle(market_open=True)
    v = mon.snapshot()["vitals"]
    assert v["market_open"] is True
    assert v["ib_connected"] is True
    assert v["bar_sizes"] == {"AAPL": 3}
    assert v["decision_count"] == 0


def test_record_cycle_carries_session(tmp_path):
    """Vitals reflect the active session and its gate state."""
    bot = make_bot(tmp_path)
    mon = PipelineMonitor(bot, bot.journal)
    mon.record_cycle(market_open=True, session="pre_market")
    v = mon.snapshot()["vitals"]
    assert v["session"] == "pre_market"
    assert v["session_active"] is True
    mon.record_cycle(market_open=False, session="overnight")
    v = mon.snapshot()["vitals"]
    assert v["session_active"] is False


def test_ib_disconnect_incident_and_recovery(tmp_path):
    """A disconnect journals an incident; recovery clears it."""
    bot = make_bot(tmp_path)
    bot.ib.isConnected = lambda: False
    mon = PipelineMonitor(bot, bot.journal)
    mon.record_cycle(market_open=True)
    mon._check_once()
    snap = mon.snapshot()
    assert snap["healthy"] is False
    assert "ib_connected" in snap["failing"]
    assert any(e["event"] == "pipeline_incident" for e in bot.journal.events)
    bot.ib.isConnected = lambda: True
    mon.record_cycle(market_open=True)
    mon._check_once()
    assert mon.snapshot()["healthy"] is True


def test_stale_bars_flagged_only_when_market_open(tmp_path):
    """Bar staleness alerts intra-session, not over the weekend."""
    bot = make_bot(tmp_path)
    mon = PipelineMonitor(bot, bot.journal)
    mon.record_cycle(market_open=True)
    mon._last_bar_growth = time.time() - BAR_STALE_SECS - 10
    mon._check_once()
    assert "bars_fresh" in mon.snapshot()["failing"]
    mon2 = PipelineMonitor(make_bot(tmp_path), bot.journal)
    mon2.record_cycle(market_open=False)
    mon2._last_bar_growth = time.time() - BAR_STALE_SECS - 10
    mon2._check_once()
    assert mon2.snapshot()["healthy"] is True


def test_brain_stall_sets_heal_flag(tmp_path):
    """A stalled brain over the threshold requests a re-subscribe."""
    bot = make_bot(tmp_path)
    mon = PipelineMonitor(bot, bot.journal)
    for _ in range(BRAIN_STALL_CYCLES):
        mon.record_cycle(market_open=True)  # decision_count never moves
        mon._check_once()
    assert mon.pop_heal() is True
    assert mon.pop_heal() is False  # consumed exactly once


def test_stale_bars_sets_heal_when_market_open(tmp_path):
    """A stale bar feed triggers the re-subscribe heal."""
    bot = make_bot(tmp_path)
    mon = PipelineMonitor(bot, bot.journal)
    mon.record_cycle(market_open=True)
    mon._last_bar_growth = time.time() - BAR_STALE_SECS - 10
    mon._check_once()
    assert "bars_fresh" in mon.snapshot()["failing"]
    assert mon.pop_heal() is True


def test_heal_for_bars_fresh_is_throttled(tmp_path):
    """Re-subscribe heal requests are rate-limited, not looped per tick."""
    bot = make_bot(tmp_path)
    mon = PipelineMonitor(bot, bot.journal)
    mon.record_cycle(market_open=True)
    mon._last_bar_growth = time.time() - BAR_STALE_SECS - 10
    mon._check_once()
    assert mon.pop_heal() is True  # first request honoured
    mon._check_once()  # still failing, but inside the throttle window
    assert mon.pop_heal() is False  # no re-set yet
    mon._last_resub_set -= RESUB_MIN_SECS + 1
    mon._check_once()  # window elapsed and still failing
    assert mon.pop_heal() is True


def test_bars_fresh_survives_buffer_saturation(tmp_path):
    """Data arrival keeps the feed fresh even when buffer length plateaus.

    Buffers trim to LOOKBACK_BARS, so len stops growing once every buffer
    fills; bars_fresh must track data-arrival recency or it would fail
    forever on a perfectly healthy, continuously-updating feed.
    """
    bot = make_bot(tmp_path)
    bot.streamer.last_data_ts = {"AAPL": 0.0}
    mon = PipelineMonitor(bot, bot.journal)
    mon.record_cycle(market_open=True)
    assert mon.bar_feed_fresh() is True  # init grace window
    bot.streamer.buffers["AAPL"].close = [1.0] * 70  # saturated
    bot.streamer.last_data_ts = {"AAPL": time.time()}  # data still flowing
    mon.record_cycle(market_open=True)
    assert mon.bar_feed_fresh() is True  # len plateau ≠ dead feed
    bot.streamer.last_data_ts = {"AAPL": time.time() - BAR_STALE_SECS - 5}
    mon.record_cycle(market_open=True)
    assert mon.bar_feed_fresh() is False  # feed genuinely silent now


def test_incidents_capped_and_shape(tmp_path):
    """The incident list stays bounded and journals full detail."""
    bot = make_bot(tmp_path)
    bot.ib.isConnected = lambda: False
    mon = PipelineMonitor(bot, bot.journal)
    for i in range(60):
        bot.journal._path.write_text("x" * (i + 1))
        mon.record_cycle(market_open=True)
        mon._check_once()
    assert len(mon._incidents) <= 50
    inc = mon._incidents[-1]
    assert set(inc) == {"ts", "check", "detail", "market_open"}


def test_eval_failure_burst_alerts(tmp_path):
    """Accumulating entry-eval failures alert while market is open."""
    bot = make_bot(tmp_path)
    bot.juli = SimpleNamespace(
        brain=SimpleNamespace(_decision_count=0), _eval_fail_count=0
    )
    mon = PipelineMonitor(bot, bot.journal)
    mon.record_cycle(market_open=True)
    mon._check_once()
    assert mon.snapshot()["healthy"] is True
    bot.juli._eval_fail_count = 5  # burst of failures since last check
    mon.record_cycle(market_open=True)
    mon._check_once()
    snap = mon.snapshot()
    assert not snap["healthy"]
    assert "entry_evals" in snap["failing"]


def test_snapshot_shape_for_telemetry(tmp_path):
    """The /pipeline payload carries the documented keys."""
    mon = PipelineMonitor(make_bot(tmp_path), make_bot(tmp_path).journal)
    snap = mon.snapshot()
    assert set(snap) == {
        "healthy",
        "failing",
        "vitals",
        "stall_cycles",
        "incidents_recent",
    }
    assert snap["healthy"] is True
    assert json.dumps(snap)  # telemetry-serializable
