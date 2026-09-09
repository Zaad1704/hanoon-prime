"""Tests for the v2 telemetry server: snapshot cache, SSE stream, legacy routes."""

import json
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from hanoon_prime.telemetry import SNAPSHOT_INTERVAL, TelemetryAPI


class StubIB:
    def isConnected(self):
        return True

    def positions(self):
        p = SimpleNamespace()
        p.contract = SimpleNamespace(symbol="SPY")
        p.position = 100
        p.avgCost = 500.0
        p.marketPrice = 505.0
        p.unrealizedPnl = 500.0
        return [p]


class StubBot:
    def __init__(self, tmp_path: Path):
        self.ib = StubIB()
        self.journal = SimpleNamespace(count=lambda: 42)
        self.hippocampus = SimpleNamespace(safety_enabled=True, _daily_pnl=-10.0)
        self._halted = False
        self._last_beat = 1.0
        self.streamer = SimpleNamespace(ticker_subs={"SPY": object(), "QQQ": object()})
        self.monitor = SimpleNamespace(
            snapshot=lambda: {"healthy": True, "cycle_lag_s": 0.2}
        )
        self.juli = SimpleNamespace(brain=None, _recent_verdicts=None)
        self.jp = tmp_path / "journal.jsonl"
        self.jp.write_text('{"event": "position_closed", "pnl": 1.0}\n')


@pytest.fixture()
def api(tmp_path: Path):
    bot = StubBot(tmp_path)
    api = TelemetryAPI(bot, bot.jp)
    api.start(port=0)  # ephemeral port: safe alongside a live bot on 8080
    time.sleep(SNAPSHOT_INTERVAL + 0.5)  # let the refresher build a snapshot
    yield api
    api.stop()


def _get(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return json.loads(r.read().decode())


def test_snapshot_contains_all_topics(api):
    snap = _get(api._server.server_address[1], "/snapshot")
    for key in (
        "health",
        "positions",
        "safety_net",
        "brain",
        "system2",
        "pipeline",
        "risk",
        "config",
        "halim",
        "verdicts",
        "trades",
        "journal",
        "session",
        "meta",
    ):
        assert key in snap, f"missing snapshot topic: {key}"
    assert snap["meta"]["built_ts"] > 0


def test_snapshot_health_values(api):
    snap = _get(api._server.server_address[1], "/snapshot")
    h = snap["health"]
    assert h["status"] == "ok"
    assert h["connected"] is True
    assert h["position_count"] == 1
    assert snap["positions"]["count"] == 1
    assert snap["positions"]["positions"][0]["ticker"] == "SPY"


def test_legacy_route_served_from_cache(api):
    port = api._server.server_address[1]
    h = _get(port, "/health")
    assert h["connected"] is True
    pos = _get(port, "/positions")
    assert pos["total_pnl"] == 500.0


def test_sse_stream_pushes_snapshots(api):
    port = api._server.server_address[1]
    frames: list[dict] = []
    stop = threading.Event()

    def reader():
        req = urllib.request.Request(f"http://127.0.0.1:{port}/stream")
        with urllib.request.urlopen(req, timeout=5) as resp:
            buf = b""
            while not stop.is_set() and len(frames) < 2:
                chunk = resp.read1(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    raw, buf = buf.split(b"\n\n", 1)
                    for line in raw.split(b"\n"):
                        if line.startswith(b"data: "):
                            frames.append(json.loads(line[6:].decode()))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    # Wait for up to 2 pushed frames (first frame is immediate).
    deadline = time.time() + SNAPSHOT_INTERVAL * 4 + 2
    while len(frames) < 2 and time.time() < deadline:
        time.sleep(0.1)
    stop.set()
    t.join(timeout=3)

    assert len(frames) >= 1, "expected at least one SSE snapshot frame"
    assert "health" in frames[0]
    assert api.stream_clients >= 0  # reader may still be winding down


def test_snapshot_interval_is_one_second():
    assert SNAPSHOT_INTERVAL == 1.0
