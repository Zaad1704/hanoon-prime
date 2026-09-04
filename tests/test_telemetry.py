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
        assert body["connected"] is True
