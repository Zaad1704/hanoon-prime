"""Halim's half of the whole-system session gate.

The bot owns the gate (telemetry GET /session). Halim polls it and falls
back to ASLEEP on any error, so a deactivated session silences inference.
Stdlib only — keep this importable in minimal environments (it is what
the repo test suite imports, not the full serve module).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Any, Dict

DEFAULT_URL = "http://127.0.0.1:8080/session"

_lock = threading.Lock()
_state: Dict[str, Any] = {"active": True, "session": "unknown"}


def session_url() -> str:
    """Gate URL — HALIM_SESSION_URL or the default telemetry endpoint."""
    return os.getenv("HALIM_SESSION_URL", DEFAULT_URL)


def poll_interval() -> float:
    """Seconds between polls — HALIM_SESSION_POLL_SEC (default 20)."""
    return float(os.getenv("HALIM_SESSION_POLL_SEC", "20.0"))


def snapshot() -> Dict[str, Any]:
    """Thread-safe copy of the last observed gate state."""
    with _lock:
        return dict(_state)


def asleep() -> bool:
    """True when the whole system is asleep (or unknown → fail-safe)."""
    return not bool(snapshot().get("active", True))


def poll_once(url: str | None = None) -> Dict[str, Any]:
    """Fetch the gate once; any error transitions to asleep (fail-safe)."""
    global _state
    try:
        with urllib.request.urlopen(url or session_url(), timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        snap: Dict[str, Any] = {
            "active": bool(data.get("active")),
            "session": str(data.get("session", "unknown")),
        }
    except Exception:
        snap = {"active": False, "session": "unknown"}
    with _lock:
        _state = snap
    return dict(snap)


def _loop() -> None:
    while True:
        poll_once()
        time.sleep(poll_interval())


def start() -> threading.Thread:
    """Start the daemon poller; first poll fires immediately."""
    thread = threading.Thread(target=_loop, daemon=True, name="halim-session-poller")
    thread.start()
    return thread