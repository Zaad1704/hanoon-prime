"""tests/test_telemetry_auth.py — bearer-gate + CORS tightening for mutations.

Security contract for the webapp API:
  * POST mutations require a bearer token (no token / wrong token => 401).
  * GET stays open (perimeter-protected tunnel), but CORS is pinned to a
    configured origin — never the '*' wildcard.
  * When TELEMETRY_AUTH_ENABLED, start() auto-generates a 0600 token at
    runtime/telemetry.token and binds it onto the handler.

Previously every route set Access-Control-Allow-Origin: * and POST mutations
(/safety-net, /config, /flatten) required no bearer — so anyone reaching
:8080 (localhost/LAN/tunnel) could close all positions or drop the safety net.
"""
from __future__ import annotations

import json
import stat
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import hanoon_prime.telemetry as tel
from hanoon_prime.telemetry import _H, TelemetryAPI

TOKEN = "test-token-1234"
CORS_ORIGIN = "https://app.example.com"


class _FakeBrain:
    def __init__(self) -> None:
        self.safety_enabled = False
        self._daily_pnl = -50.0
        self._consecutive_losses = 0


class _FakeJournal:
    """Minimal stand-in for Journal."""

    def __init__(self, entries: list[dict[str, Any]] | None = None) -> None:
        self._entries = entries or []

    def count(self) -> int:
        return len(self._entries)

    def entries(self) -> list[dict[str, Any]]:
        return self._entries


class _FakeBot:
    def __init__(self) -> None:
        self.hippocampus = _FakeBrain()
        self.ib = MagicMock()
        self.ib.isConnected.return_value = True
        self.ib.positions.return_value = []
        self.journal = _FakeJournal([])

        class _Sub:
            ticker_subs: list[str] = ["AAPL", "MSFT", "TSLA"]

        self.streamer = _Sub()


@pytest.fixture(autouse=True)
def _reset_handler_security() -> Any:
    """Restore _H auth/CORS class attrs after every test (isolation)."""
    yield
    _H.auth_enabled = False
    _H.telemetry_token = ""
    _H.cors_origin = None


def _start(bot: Any, jp: Path) -> HTTPServer:
    TelemetryAPI(bot, jp)
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _post(
    port: int, path: str, body: dict[str, Any], token: str | None = None
) -> tuple[int, dict[str, Any]]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(port: int, path: str) -> tuple[int, Any, str | None]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
            return (
                r.status,
                json.loads(r.read()),
                r.headers.get("Access-Control-Allow-Origin"),
            )
    except urllib.error.HTTPError as e:
        return (
            e.code,
            json.loads(e.read()),
            e.headers.get("Access-Control-Allow-Origin"),
        )


class TestBearerGate:
    def test_post_without_token_is_rejected(self, tmp_path) -> None:
        monkeypatch_H(auth_enabled=True, telemetry_token=TOKEN, cors_origin=CORS_ORIGIN)
        jp = tmp_path / "journal_live.jsonl"
        jp.write_text("")
        srv = _start(_FakeBot(), jp)
        port = srv.server_address[1]
        try:
            code, body = _post(port, "/safety-net", {"action": "enable"})
            assert code == 401
            assert body["error"] == "unauthorized"
        finally:
            srv.shutdown()
            srv.server_close()

    def test_post_with_wrong_token_is_rejected(self, tmp_path) -> None:
        monkeypatch_H(auth_enabled=True, telemetry_token=TOKEN, cors_origin=CORS_ORIGIN)
        jp = tmp_path / "journal_live.jsonl"
        jp.write_text("")
        srv = _start(_FakeBot(), jp)
        port = srv.server_address[1]
        try:
            code, _body = _post(
                port, "/safety-net", {"action": "enable"}, token="wrong"
            )
            assert code == 401
        finally:
            srv.shutdown()
            srv.server_close()

    def test_post_with_valid_token_succeeds(self, tmp_path) -> None:
        monkeypatch_H(auth_enabled=True, telemetry_token=TOKEN, cors_origin=CORS_ORIGIN)
        jp = tmp_path / "journal_live.jsonl"
        jp.write_text("")
        srv = _start(_FakeBot(), jp)
        port = srv.server_address[1]
        try:
            code, _body = _post(port, "/safety-net", {"action": "enable"}, token=TOKEN)
            assert code == 200
        finally:
            srv.shutdown()
            srv.server_close()

    def test_get_still_open_when_auth_on(self, tmp_path) -> None:
        monkeypatch_H(auth_enabled=True, telemetry_token=TOKEN, cors_origin=CORS_ORIGIN)
        jp = tmp_path / "journal_live.jsonl"
        jp.write_text("")
        srv = _start(_FakeBot(), jp)
        port = srv.server_address[1]
        try:
            code, _body, _aco = _get(port, "/health")
            assert code == 200
        finally:
            srv.shutdown()
            srv.server_close()

    def test_authorized_open_when_disabled(self) -> None:
        h = _H.__new__(_H)
        h.auth_enabled = False
        h.telemetry_token = ""
        assert h._authorized() is True


class TestCORS:
    def test_aco_reflects_configured_origin_not_wildcard(self, tmp_path) -> None:
        monkeypatch_H(auth_enabled=True, telemetry_token=TOKEN, cors_origin=CORS_ORIGIN)
        jp = tmp_path / "journal_live.jsonl"
        jp.write_text("")
        srv = _start(_FakeBot(), jp)
        port = srv.server_address[1]
        try:
            _code, _body, aco = _get(port, "/health")
            assert aco == CORS_ORIGIN
            assert aco != "*"
        finally:
            srv.shutdown()
            srv.server_close()

    def test_no_aco_when_origin_unconfigured(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(_H, "auth_enabled", False)
        monkeypatch.setattr(_H, "telemetry_token", "")
        monkeypatch.setattr(_H, "cors_origin", None)
        monkeypatch.setattr(_H, "_cors_origin", lambda self: None)
        jp = tmp_path / "journal_live.jsonl"
        jp.write_text("")
        srv = _start(_FakeBot(), jp)
        port = srv.server_address[1]
        try:
            code, _body, aco = _get(port, "/health")
            assert code == 200
            assert aco is None  # deny cross-origin when origin unknown
        finally:
            srv.shutdown()
            srv.server_close()


class TestTokenProvisioning:
    def test_apply_security_posture_generates_0600_token(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(tel, "TELEMETRY_AUTH_ENABLED", True)
        monkeypatch.setenv("TELEMETRY_CORS_ORIGIN", "https://web.hanoon")
        jp = tmp_path / "journal_live.jsonl"
        jp.write_text("")
        api = TelemetryAPI(_FakeBot(), jp)
        api._apply_security_posture()
        tp = api._token_path()
        assert tp.exists()
        mode = stat.S_IMODE(tp.stat().st_mode)
        assert mode == 0o600
        assert _H.auth_enabled is True
        assert _H.telemetry_token == tp.read_text().strip()
        assert len(_H.telemetry_token) >= 32
        assert _H.cors_origin == "https://web.hanoon"

    def test_open_when_auth_disabled(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(tel, "TELEMETRY_AUTH_ENABLED", False)
        jp = tmp_path / "journal_live.jsonl"
        jp.write_text("")
        api = TelemetryAPI(_FakeBot(), jp)
        api._apply_security_posture()
        assert _H.auth_enabled is False
        assert _H.telemetry_token == ""


def monkeypatch_H(**attrs: Any) -> None:
    """Set _H class attrs for the current test (restored by autouse fixture)."""
    for k, v in attrs.items():
        setattr(_H, k, v)
