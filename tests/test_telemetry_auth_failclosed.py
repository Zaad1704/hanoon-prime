"""tests/test_telemetry_auth_failclosed.py — fail-closed bearer gate.

Regression: FIX-2026-09-23-12 (Class F — safety-net integrity).

The bearer gate on POST mutations (/safety-net, /config, /flatten) used to
FAIL OPEN: `_authorized()` returned True when auth was disabled or the token
was unset, GET /auth handed the token to any CORS-allowed browser origin
(and to clients sending no Origin at all), and POST_ROUTES was a stale
subset that omitted /flatten from allow-list audits.

New contract:
  * `_authorized()` denies on ANY misconfiguration (auth off, no token,
    bad/missing Authorization header) — never allows.
  * GET /auth returns metadata only; the token NEVER leaves the host.
    It lives in runtime/telemetry.token (0600); the user copies it into
    the dashboard's bearer-token field for mutations.
  * POST_ROUTES is frozenset(POST_HANDLERS): every served mutation route
    is auditable, including /flatten.
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

import hanoon_prime.telemetry as tel
from hanoon_prime.telemetry import _H, POST_HANDLERS, POST_ROUTES, TelemetryAPI

TOKEN = "failclosed-test-token-5678"


class _FakeBrain:
    def __init__(self) -> None:
        self.safety_enabled = False


class _FakeJournal:
    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

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
        self.journal = _FakeJournal()

        class _Sub:
            ticker_subs: list[str] = ["AAPL"]

        self.streamer = _Sub()


def _set_H(**attrs: Any) -> None:
    for k, v in attrs.items():
        setattr(_H, k, v)


def _handler(**attrs: Any) -> Any:
    """Bare handler instance with instance-level auth attrs (no HTTP)."""
    h = _H.__new__(_H)
    h.auth_enabled = attrs.get("auth_enabled", False)
    h.telemetry_token = attrs.get("telemetry_token", "")
    h.headers = attrs.get("headers", {})
    return h


@pytest.fixture(autouse=True)
def _reset_handler_security():
    yield
    _H.auth_enabled = False
    _H.telemetry_token = ""
    _H.cors_origin = None
    tel._FLATTEN_REQUESTED.clear()


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


class TestAuthorizedFailClosed:
    """Regression: FIX-2026-09-23-12 — unit-level gate semantics."""

    def test_denies_when_auth_disabled_even_with_token(self) -> None:
        h = _handler(auth_enabled=False, telemetry_token=TOKEN)
        assert h._authorized() is False

    def test_denies_when_token_missing(self) -> None:
        h = _handler(auth_enabled=True, telemetry_token="")
        assert h._authorized() is False

    def test_denies_wrong_token(self) -> None:
        h = _handler(
            auth_enabled=True,
            telemetry_token=TOKEN,
            headers={"Authorization": "Bearer wrong"},
        )
        assert h._authorized() is False

    def test_denies_missing_header(self) -> None:
        h = _handler(auth_enabled=True, telemetry_token=TOKEN, headers={})
        assert h._authorized() is False

    def test_denies_malformed_scheme(self) -> None:
        h = _handler(
            auth_enabled=True,
            telemetry_token=TOKEN,
            headers={"Authorization": TOKEN},  # missing "Bearer " prefix
        )
        assert h._authorized() is False

    def test_allows_correct_bearer(self) -> None:
        h = _handler(
            auth_enabled=True,
            telemetry_token=TOKEN,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert h._authorized() is True


class TestPostMutationsFailClosedHTTP:
    """Regression: FIX-2026-09-23-12 — every mutation denies when misconfigured."""

    def _srv(self, tmp_path, **attrs: Any):
        _set_H(**attrs)
        jp = tmp_path / "journal_live.jsonl"
        jp.write_text("")
        srv = _start(_FakeBot(), jp)
        return srv, srv.server_address[1]

    def test_post_denied_when_auth_disabled(self, tmp_path) -> None:
        srv, port = self._srv(tmp_path, auth_enabled=False, telemetry_token="")
        try:
            code, body = _post(port, "/safety-net", {"action": "disable"})
            assert code == 401
            assert body["error"] == "unauthorized"
        finally:
            srv.shutdown()
            srv.server_close()

    def test_post_denied_when_token_unset_but_enabled(self, tmp_path) -> None:
        srv, port = self._srv(tmp_path, auth_enabled=True, telemetry_token="")
        try:
            code, _ = _post(port, "/config", {})
            assert code == 401
        finally:
            srv.shutdown()
            srv.server_close()

    def test_every_mutation_route_denied_without_token(self, tmp_path) -> None:
        """No POST route (incl. /flatten) is reachable without a token."""
        srv, port = self._srv(tmp_path, auth_enabled=True, telemetry_token=TOKEN)
        try:
            for path in ("/safety-net", "/config", "/flatten"):
                code, _ = _post(port, path, {})
                assert code == 401, path
        finally:
            srv.shutdown()
            srv.server_close()

    def test_flatten_gated_but_reachable_with_token(self, tmp_path) -> None:
        """/flatten was missing from POST_ROUTES yet served — now audited + gated."""
        srv, port = self._srv(tmp_path, auth_enabled=True, telemetry_token=TOKEN)
        try:
            code, _ = _post(port, "/flatten", {"order_type": "market"})
            assert code == 401  # no token
            code, _ = _post(port, "/flatten", {"order_type": "market"}, token=TOKEN)
            assert code == 200  # gate passes; handler runs
            assert tel._FLATTEN_REQUESTED.get("order_type") == "market"
        finally:
            srv.shutdown()
            srv.server_close()


class TestRouteInventorySingleSource:
    """Regression: FIX-2026-09-23-12 — POST_ROUTES == POST_HANDLERS, always."""

    def test_post_routes_covers_every_handler(self) -> None:
        assert POST_ROUTES == frozenset(POST_HANDLERS)
        assert "/flatten" in POST_ROUTES
        assert "/safety-net" in POST_ROUTES
        assert "/config" in POST_ROUTES

    def test_no_silent_extra_routes(self) -> None:
        # If someone adds a POST handler, it is automatically audited.
        assert set(POST_ROUTES) == set(POST_HANDLERS.keys())
