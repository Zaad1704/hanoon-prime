"""HALIM session gate — stdlib import-only, fail-safe asleep."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "halim") not in sys.path:
    sys.path.insert(0, str(ROOT / "halim"))

from halim import session_gate  # noqa: E402


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *a: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _patch_urlopen(monkeypatch, result: MagicMock) -> None:
    monkeypatch.setattr(session_gate.urllib.request, "urlopen", result)


class TestSessionPoller:
    def test_ok_active(self, monkeypatch) -> None:
        _patch_urlopen(
            monkeypatch,
            MagicMock(return_value=_Resp(b'{"active": true, "session": "rth"}')),
        )
        session_gate.poll_once()
        assert session_gate.asleep() is False
        assert session_gate.snapshot() == {"active": True, "session": "rth"}

    def test_ok_deactivated(self, monkeypatch) -> None:
        _patch_urlopen(
            monkeypatch,
            MagicMock(return_value=_Resp(b'{"active": false, "session": "overnight"}')),
        )
        session_gate.poll_once()
        assert session_gate.asleep() is True

    def test_timeout_falls_back_asleep(self, monkeypatch) -> None:
        urlopen = MagicMock(side_effect=TimeoutError("gate down"))
        _patch_urlopen(monkeypatch, urlopen)
        session_gate.poll_once()
        assert session_gate.asleep() is True
        assert session_gate.snapshot()["session"] == "unknown"

    def test_garbage_falls_back_asleep(self, monkeypatch) -> None:
        _patch_urlopen(monkeypatch, MagicMock(return_value=_Resp(b"not json{")))
        session_gate.poll_once()
        assert session_gate.asleep() is True

    def test_url_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("HALIM_SESSION_URL", "http://127.0.0.1:9999/gate")
        urlopen = MagicMock(return_value=_Resp(b'{"active": true, "session": "rth"}'))
        _patch_urlopen(monkeypatch, urlopen)
        session_gate.poll_once()
        last_url = urlopen.call_args.args[0]
        assert last_url == "http://127.0.0.1:9999/gate"

    def test_wake_after_sleep(self, monkeypatch) -> None:
        _patch_urlopen(
            monkeypatch,
            MagicMock(side_effect=TimeoutError("gate down")),
        )
        session_gate.poll_once()
        assert session_gate.asleep() is True
        urlopen = MagicMock(return_value=_Resp(b'{"active": true, "session": "rth"}'))
        _patch_urlopen(monkeypatch, urlopen)
        session_gate.poll_once()
        assert session_gate.asleep() is False
