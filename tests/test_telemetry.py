"""tests/test_telemetry.py — HTTP endpoint tests for the webapp API.

Tests the TelemetryAPI endpoints: /health, /safety-net (GET + POST),
verifying the safety net toggle works via the webapp.
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


def _post(port: int, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Make a POST request, return (status, json_body)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


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
        assert body["session"] in {"pre_market", "rth", "post_market", "overnight"}
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
