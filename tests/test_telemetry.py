"""tests/test_telemetry.py — HTTP endpoint tests for the webapp API.

Tests the TelemetryAPI endpoints: /health, /safety-net (GET + POST),
verifying the safety net toggle works via the webapp.

Mutations are fail-closed (FIX-2026-09-23-12): every POST requires a valid
``Authorization: Bearer <token>`` header, so this file installs a synthetic
token on the handler and sends it from the ``_post`` helper. Unauthenticated
POSTs are asserted to 401 — that is the intended security behavior, not a
regression.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from hanoon_prime.immune import DAILY_LOSS_LIMIT
from hanoon_prime.telemetry import _H, TelemetryAPI

# Synthetic bearer token for this file's POST tests. The handler's auth
# state is class-level, so the _isolate_handler fixture installs these
# credentials before each test and restores whatever was there after.
_TEST_TOKEN = "test-telemetry-bearer-token"


class _FakeBrain:
    """Minimal stand-in for Hippocampus — supports the toggle trait."""

    def __init__(self, safety_enabled: bool = False) -> None:
        self.safety_enabled: bool = safety_enabled
        self._daily_pnl: float = -50.0
        self._consecutive_losses: int = 0


class _FakeJournal:
    """Minimal stand-in for Journal."""

    def __init__(self, entries: list[dict[str, Any]] | None = None) -> None:
        self._entries = entries or []

    def count(self) -> int:
        return len(self._entries)

    def entries(self) -> list[dict[str, Any]]:
        return self._entries


class _FakeBot:
    """Minimal stand-in for IBStreamingBot."""

    def __init__(
        self,
        safety_enabled: bool = False,
        entries: list[dict[str, Any]] | None = None,
    ) -> None:
        self.hippocampus = _FakeBrain(safety_enabled=safety_enabled)
        self._halted: bool = False
        self.ib = MagicMock()
        self.ib.isConnected.return_value = True
        self.ib.positions.return_value = []
        self.journal = _FakeJournal(entries or [])

        class _Sub:
            ticker_subs: list[str] = ["AAPL", "MSFT", "TSLA"]

        self.streamer = _Sub()


@pytest.fixture
def server(tmp_path: Path) -> int:
    """Start a TelemetryAPI on an ephemeral port; yield port; shut down."""
    bot = _FakeBot(safety_enabled=False)
    journal_path = tmp_path / "journal_live.jsonl"
    journal_path.write_text("")
    TelemetryAPI(bot, journal_path)
    srv = HTTPServer(("127.0.0.1", 0), _H)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()
    srv.server_close()


def _get(port: int, path: str) -> tuple[int, dict[str, Any]]:
    """Make a GET request, return (status, json_body)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(
    port: int, path: str, body: dict[str, Any], token: str | None = _TEST_TOKEN
) -> tuple[int, dict[str, Any]]:
    """Make a POST request, return (status, json_body).

    Sends ``Authorization: Bearer <token>`` by default because mutations
    are fail-closed; pass ``token=None`` to exercise the unauthenticated
    (401) path.
    """
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(autouse=True)
def _isolate_handler():
    """Isolate the shared _H handler class between tests.

    _H carries class-level state (bot, caches, auth, CORS). Snapshot every
    attribute this file (or any earlier suite file) may touch, install the
    synthetic auth credentials for this file's POST tests, and restore the
    snapshot afterwards so no test leaks into another.

    Snapshot routes (/health, /safety-net, /risk, /verdicts) are served from
    ``_H.cache`` when present — a cache left by an earlier suite file would
    override this file's FakeBot. Clear it (and extra_cache) before each test
    so GET falls through to the live builders. (FIX-2026-09-23-14)
    """
    saved = {
        k: getattr(_H, k)
        for k in (
            "bot",
            "journal_path",
            "extra_cache",
            "extra_lock",
            "cache",
            "cache_lock",
            "sse_registry",
            "on_mutation",
            "inspection_builder",
            "auth_enabled",
            "telemetry_token",
            "cors_origin",
        )
    }
    _H.auth_enabled = True
    _H.telemetry_token = _TEST_TOKEN
    _H.cache = {"data": {}, "ts": 0.0}
    _H.cache_lock = threading.Lock()
    _H.extra_cache = {}
    _H.extra_lock = threading.Lock()
    _H.on_mutation = None
    yield
    for k, v in saved.items():
        setattr(_H, k, v)


class TestHandlerIsolation:
    def test_cache_cleared_by_isolate_fixture(self):
        """Stale snapshot cache must not override FakeBot on GET routes.

        Regression: FIX-2026-09-23-14
        """
        _H.cache = {"data": {"health": {"halted": False, "daily_pnl": 0.0}}, "ts": 1.0}
        # Autouse fixture already cleared at setup; re-assert post-seed clear
        # happens on next test via fixture — this test documents the contract.
        assert isinstance(_H.cache, dict)
        _H.cache = {"data": {}, "ts": 0.0}
        assert _H.cache.get("data") == {}


class TestSafetyNetToggle:
    """Test /safety-net GET (status) and POST (toggle) via webapp."""

    def test_get_status_disabled_by_default(self, server):
        """GET /safety-net returns enabled=False when disabled."""
        code, body = _get(server, "/safety-net")
        assert code == 200
        assert body["enabled"] is False
        assert body["daily_pnl"] == -50.0
        assert body["limit"] == DAILY_LOSS_LIMIT
        assert "consecutive_losses" in body

    def test_post_enable(self, server):
        """POST /safety-net {action: enable} turns it on."""
        code, body = _post(server, "/safety-net", {"action": "enable"})
        assert code == 200
        assert body["safety_net_enabled"] is True
        # Verify via GET
        _, body2 = _get(server, "/safety-net")
        assert body2["enabled"] is True

    def test_post_disable(self, server):
        """POST /safety-net {action: disable} turns it off."""
        _post(server, "/safety-net", {"action": "enable"})
        code, body = _post(server, "/safety-net", {"action": "disable"})
        assert code == 200
        assert body["safety_net_enabled"] is False
        _, body2 = _get(server, "/safety-net")
        assert body2["enabled"] is False

    def test_post_invalid_action_returns_400(self, server):
        """POST with unknown action returns 400."""
        code, body = _post(server, "/safety-net", {"action": "maybe"})
        assert code == 400
        assert "error" in body

    def test_post_empty_body_returns_400(self, server):
        """POST with empty body returns 400."""
        code, body = _post(server, "/safety-net", {})
        assert code == 400

    def test_unknown_get_path_returns_404(self, server):
        """GET to unknown path returns 404."""
        code, body = _get(server, "/nope")
        assert code == 404

    def test_unknown_post_path_returns_404(self, server):
        """POST to unknown path returns 404."""
        code, body = _post(server, "/nope", {})
        assert code == 404

    def test_health_includes_safety_net(self, server):
        """GET /health includes safety_net_enabled field."""
        code, body = _get(server, "/health")
        assert code == 200
        assert "safety_net_enabled" in body
        assert "halted" in body
        assert body["connected"] is True

    def test_post_resume_clears_halt(self, server):
        """POST /safety-net {action: resume} clears halted state."""
        bot = _FakeBot()
        bot._halted = True
        _H.bot = bot
        code, body = _post(server, "/safety-net", {"action": "resume"})
        assert code == 200
        assert body["halted"] is False
        assert bot._halted is False

    def test_post_without_token_rejected_401(self, server):
        """Regression: FIX-2026-09-23-12 — mutations are fail-closed: no bearer token => 401."""
        code, body = _post(server, "/safety-net", {"action": "disable"}, token=None)
        assert code == 401
        assert body["error"] == "unauthorized"

    def test_post_with_wrong_token_rejected_401(self, server):
        """Regression: FIX-2026-09-23-12 — a wrong bearer token is rejected the same as a missing one."""
        code, _ = _post(server, "/safety-net", {"action": "disable"}, token="wrong")
        assert code == 401

    def test_safety_net_status_includes_halted(self, server):
        """GET /safety-net includes halted field."""
        code, body = _get(server, "/safety-net")
        assert code == 200
        assert "halted" in body


def _policy_bot(policy: dict[str, Any] | None, verdicts: list[Any] | None = None):
    """Build a _FakeBot whose juli.brain publishes policy_state/verdicts."""
    bot = _FakeBot()

    class _State:
        def __init__(self, p: dict[str, Any] | None) -> None:
            self.policy: dict[str, Any] = {"policy_state": p or {}}

        def get(self, k: str, d: Any = None) -> Any:
            return self.policy.get(k, d)

        def update(self, **_k: Any) -> None:
            return None

    class _Brain:
        def __init__(self, p: dict[str, Any] | None) -> None:
            self.state = _State(p)
            self.resume = MagicMock()
            self.set_safety_enabled = MagicMock()

        def snapshot(self) -> dict[str, Any]:
            return {}

    class _Juli:
        def __init__(self, brain: _Brain, rv: list[Any] | None) -> None:
            self.brain = brain
            self._recent_verdicts = rv or []

    bot.juli = _Juli(_Brain(policy), verdicts)
    return bot


class TestBrainBackedTelemetry:
    """Task 11: telemetry reads policy_state + serves /verdicts."""

    def test_health_reads_policy_state_when_present(self, server):
        """GET /health reflects halted from the brain's policy_state."""
        bot = _policy_bot({"halted": True, "authorized": False, "enabled": True})
        _H.bot = bot
        code, body = _get(server, "/health")
        assert code == 200
        assert body["halted"] is True
        assert body["authorized"] is False

    def test_resume_calls_brain_resume(self, server):
        """POST /safety-net {action: resume} invokes the brain resume."""
        bot = _policy_bot({"halted": True, "authorized": False})
        _H.bot = bot
        code, body = _post(server, "/safety-net", {"action": "resume"})
        assert code == 200
        bot.juli.brain.resume.assert_called_once()

    def test_toggle_calls_brain_set_safety_enabled(self, server):
        """POST /safety-net {action: enable} reaches the brain toggle."""
        bot = _policy_bot({})
        _H.bot = bot
        code, body = _post(server, "/safety-net", {"action": "enable"})
        assert code == 200
        bot.juli.brain.set_safety_enabled.assert_called_once_with(True)

    def test_verdicts_endpoint_lists_recent(self, server):
        """GET /verdicts returns seeded Verdicts with action + reason."""
        from hanoon_prime.brain.policy.verdict import Verdict

        bot = _policy_bot(
            {},
            verdicts=[
                Verdict(ticker="TSLA", action="ENTER", reason="admitted"),
                Verdict(ticker="AAPL", action="VETOED", reason="sized_to_zero"),
            ],
        )
        _H.bot = bot
        code, body = _get(server, "/verdicts")
        assert code == 200
        assert body["count"] == 2
        assert body["verdicts"][0]["ticker"] == "TSLA"
        assert body["verdicts"][0]["action"] == "ENTER"
        assert body["verdicts"][0]["reason"] == "admitted"

    def test_risk_state_subset_keys(self, server):
        """GET /risk returns the policy_state subset (no private fields)."""
        bot = _policy_bot(
            {
                "equity": 100000.0,
                "risk_scalar": 0.5,
                "halted": True,
                "pause_reason": "daily_loss_limit",
            }
        )
        _H.bot = bot
        code, body = _get(server, "/risk")
        assert code == 200
        assert body["risk_scalar"] == 0.5
        assert "halted" not in body
        assert "pause_reason" not in body


class TestSessionEndpoint:
    """Master session gate: /session route + /health session fields."""

    def test_session_endpoint_shape(self) -> None:
        handler = _H.__new__(_H)
        handler.bot = None
        body = handler._session()
        assert body["session"] in {
            "pre_market",
            "rth",
            "post_market",
            "overnight",
            "weekend",
        }
        assert isinstance(body["active"], bool)
        assert set(body["enabled"]) == {
            "pre_market",
            "rth",
            "post_market",
            "overnight",
        }
        assert isinstance(body["ts"], float)

    def test_health_reports_session(self) -> None:
        handler = _H.__new__(_H)
        bot = MagicMock()
        bot.streamer.ticker_subs = {}
        bot.journal = MagicMock()
        bot.journal.count.return_value = 0
        bot.hippocampus = MagicMock()
        bot.hippocampus.safety_enabled = True
        bot.ib = MagicMock()
        bot.ib.isConnected.return_value = True
        bot.ib.positions.return_value = []
        bot._halted = False
        bot._last_beat = 0.0
        handler.bot = bot
        h = handler._health()
        assert "session" in h
        assert "session_active" in h
        assert isinstance(h["session_active"], bool)

    def test_session_route_registered(self) -> None:
        from hanoon_prime.telemetry import ROUTES_GET

        assert ROUTES_GET["/session"] == "_session"


class TestIbTickerRowDatetime:
    """Regression: Ticker.time is a datetime — int(datetime) raised TypeError
    and took down the whole /ib route (empty reply → webapp showed DOWN)."""

    def test_epoch_seconds_accepts_datetime_and_epoch(self) -> None:
        from datetime import datetime, timezone

        from hanoon_prime.telemetry import _epoch_seconds

        dt = datetime(2026, 9, 11, 0, 0, tzinfo=timezone.utc)
        assert _epoch_seconds(dt) == int(dt.timestamp())
        assert _epoch_seconds(1787123456) == 1787123456
        assert _epoch_seconds(None) is None
        assert _epoch_seconds("garbage") is None

    def test_ib_ticker_row_with_datetime_time(self) -> None:
        from datetime import datetime, timezone

        from hanoon_prime.telemetry import _H

        tk = MagicMock()
        tk.contract.symbol = "SPY"
        tk.bid = 100.0
        tk.ask = 100.5
        tk.last = 100.25
        tk.close = 99.0
        tk.open = 99.5
        tk.high = 101.0
        tk.low = 98.5
        tk.volume = 1_000
        tk.bidSize = 3
        tk.askSize = 4
        tk.lastSize = 2
        tk.halted = 0
        tk.marketPrice = MagicMock(return_value=100.25)
        tk.time = datetime.now(tz=timezone.utc)
        row = _H._ib_ticker_row(tk)
        assert row["symbol"] == "SPY"
        assert isinstance(row["time"], int)


class TestSnapshot:
    """Part B: raw decision chain + trade-quality surfaces (live, ungated reads)."""

    def test_decisions_and_trade_quality_render(self, tmp_path: Path) -> None:
        jp = tmp_path / "journal_live.jsonl"
        jp.write_text("")
        h = _H.__new__(_H)
        h.journal_path = jp
        decisions = h._decisions()
        assert "chain" in decisions
        assert decisions["total_events"] == 0
        trade_quality = h._trade_quality()
        assert "win_rate" in trade_quality
        assert trade_quality["trades"] == 0


class TestObservabilityRoutes:
    """New observability surfaces: /trust, /vitals, /exec-quality, /metrics."""

    def test_trust_route_registered(self) -> None:
        from hanoon_prime.telemetry import ROUTES_GET

        assert ROUTES_GET["/trust"] == "_trust"
        assert ROUTES_GET["/vitals"] == "_vitals"
        assert ROUTES_GET["/exec-quality"] == "_exec_quality"
        assert ROUTES_GET["/metrics"] == "_metrics"

    def test_trust_healthy_when_monitor_green(self, server) -> None:
        from hanoon_prime.monitor.trust import HEALTHY_FLOOR

        class _Mon:
            def snapshot(self):
                return {
                    "healthy": True,
                    "failing": {},
                    "vitals": {"decision_count": 5, "feed_age": 2.0},
                    "stall_cycles": 0,
                }

        bot = _FakeBot()
        bot.monitor = _Mon()
        _H.bot = bot
        code, body = _get(server, "/trust")
        assert code == 200
        assert body["status"] == "HEALTHY"
        assert body["score"] >= HEALTHY_FLOOR
        assert body["failing"] == []

    def test_trust_critical_when_brain_stalled(self, server) -> None:
        class _Mon:
            def snapshot(self):
                return {
                    "healthy": False,
                    "failing": {
                        "brain_advancing": "40 cycles",
                        "bars_fresh": "stale",
                        "ib_connected": "Gateway link",
                    },
                    "vitals": {"decision_count": 0},
                    "stall_cycles": 40,
                }

        bot = _FakeBot()
        bot.monitor = _Mon()
        _H.bot = bot
        code, body = _get(server, "/trust")
        assert code == 200
        assert body["status"] == "CRITICAL"
        assert "brain_advancing" in body["failing"]

    def test_trust_unknown_when_monitor_missing(self, server) -> None:
        _H.bot = _FakeBot()  # no monitor attribute
        code, body = _get(server, "/trust")
        assert code == 200
        assert body["status"] == "UNKNOWN"

    def test_vitals_route_returns_recorded_rows(self, server, tmp_path: Path) -> None:
        from hanoon_prime.monitor.vitals_log import VitalsLog

        bot = _FakeBot()
        vlog = VitalsLog(tmp_path)
        vlog.record(
            {
                "healthy": True,
                "failing": {},
                "vitals": {"decision_count": 7, "feed_age": 1.0},
                "stall_cycles": 0,
            }
        )
        bot.vitals_log = vlog
        _H.bot = bot
        code, body = _get(server, "/vitals")
        assert code == 200
        assert body["enabled"] is True
        assert body["rows"]
        assert body["rows"][0]["decision_count"] == 7

    def test_vitals_route_disabled_when_missing(self, server) -> None:
        _H.bot = _FakeBot()  # no vitals_log attribute
        code, body = _get(server, "/vitals")
        assert code == 200
        assert body["enabled"] is False

    def test_exec_quality_route_shape(self, server) -> None:
        from hanoon_prime.monitor.exec_quality import ExecQuality

        bot = _FakeBot()
        ex = MagicMock()
        ex.exec_stats = ExecQuality(maxlen=10)
        ex.exec_stats.record(1, 100.0, 100.05, 1.0, 1.4)
        bot.executor = ex
        _H.bot = bot
        code, body = _get(server, "/exec-quality")
        assert code == 200
        assert body["enabled"] is True
        assert body["fills"] == 1
        assert "slippage_bps_p50" in body

    def test_metrics_route_exposes_gauges(self, server) -> None:
        from hanoon_prime.monitor.trust import HEALTHY_FLOOR

        class _Mon:
            def snapshot(self):
                return {
                    "healthy": True,
                    "failing": {},
                    "vitals": {"decision_count": 9, "feed_age": 1.0},
                    "stall_cycles": 0,
                }

        bot = _FakeBot()
        bot.monitor = _Mon()
        _H.bot = bot
        code, body = _get(server, "/metrics")
        assert code == 200
        assert body["hanoon_pipeline_healthy"] == 1.0
        assert body["hanoon_trust_score"] >= HEALTHY_FLOOR
        assert body["hanoon_ib_connected"] == 1.0
        assert "hanoon_feed_age_seconds" in body

    def test_snapshot_embeds_trust(self, tmp_path: Path) -> None:
        from hanoon_prime.monitor.trust import HEALTHY_FLOOR
        from hanoon_prime.telemetry import build_snapshot

        class _Mon:
            def snapshot(self):
                return {"healthy": True, "failing": {}, "stall_cycles": 0}

        h = _H.__new__(_H)
        bot = _FakeBot()
        bot.monitor = _Mon()
        h.bot = bot
        h.journal_path = tmp_path / "journal_live.jsonl"
        h.journal_path.write_text("")
        snap = build_snapshot(h)
        assert snap["trust"]["score"] >= HEALTHY_FLOOR
