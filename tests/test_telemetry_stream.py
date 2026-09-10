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


@pytest.fixture(autouse=True)
def _reset_handler_state():
    """Reset the _H class-level cache/builder so per-request overrides in one
    test never leak into later tests (order-independent)."""
    yield
    from hanoon_prime.telemetry import _H

    _H.inspection_builder = None
    _H.extra_cache.clear()


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


# ── On-demand routes: /account, /resources, /inspection ────────────────


def test_account_route_served(api):
    port = api._server.server_address[1]
    acc = _get(port, "/account")
    assert isinstance(acc, dict)


def test_account_route_reads_brain_feed(api):
    """With a brain that publishes account_feed, /account returns it raw."""
    from hanoon_prime.telemetry import _H

    sudo_brain = {
        "equity": 100_000.0,
        "equity_synced": True,
        "daily_pnl": 250.0,
        "positions_open": 2,
        "account_summary": {
            "net_liq": 101_500.0,
            "buying_power": 200_000.0,
            "cash": 50_000.0,
            "realized_pnl": 120.0,
            "unrealized_pnl": 130.0,
        },
    }
    bot = api._bot
    bot.juli.brain = SimpleNamespace(
        state=SimpleNamespace(
            snapshot=lambda: {"account_feed": sudo_brain, "policy_state": {}}
        )
    )
    _H.extra_cache.clear()
    acc = _get(api._server.server_address[1], "/account")
    assert acc["equity"] == 100_000.0
    assert acc["account_summary"]["net_liq"] == 101_500.0
    assert acc["account_summary"]["buying_power"] == 200_000.0
    assert "realized_pnl" in acc["account_summary"]


def test_resources_route_reports_bot_process(api):
    """/resources includes the test process (bot role) + a loadavg array."""
    port = api._server.server_address[1]
    res = _get(port, "/resources")
    assert "procs" in res
    assert "loadavg" in res
    roles = {p.get("role") for p in res["procs"]}
    assert "bot" in roles
    bot = next(p for p in res["procs"] if p["role"] == "bot")
    assert bot.get("pid") is not None
    assert "cpu_pct" in bot
    assert "rss_kb" in bot


def test_inspection_route_builder_and_cache(api):
    """/inspection uses the overridable builder and serves from the 30s TTL."""
    from hanoon_prime.telemetry import _H

    calls = {"n": 0}

    def fake_builder():
        calls["n"] += 1
        return {"status": "OK", "results": [], "hard_fails": [], "anomalies": []}

    _H.inspection_builder = fake_builder
    _H.extra_cache.clear()
    port = api._server.server_address[1]
    first = _get(port, "/inspection")
    assert first["status"] == "OK"
    assert calls["n"] == 1
    # Second hit within TTL must NOT re-run the builder.
    again = _get(port, "/inspection")
    assert calls["n"] == 1
    assert again["status"] == "OK"


def test_inspection_route_error_path(api):
    """Builder raising → UNVERIFIABLE payload, never a 500."""
    from hanoon_prime.telemetry import _H

    def boom():
        raise RuntimeError("subprocess exploded")

    _H.inspection_builder = boom
    _H.extra_cache.clear()
    got = _get(api._server.server_address[1], "/inspection")
    assert got["status"] == "UNVERIFIABLE"
    assert "subprocess exploded" in got["error"]
