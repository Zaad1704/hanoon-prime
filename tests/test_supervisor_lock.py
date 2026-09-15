"""test_supervisor_lock.py — bot_supervisor single-instance lock.

The lock serializes supervisor launches so two instances can never fight
over the bot/telemetry ports or overwrite each other's pidfile.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

_SPEC = importlib.util.spec_from_file_location(
    "bot_supervisor", SCRIPTS / "bot_supervisor.py"
)
assert _SPEC is not None and _SPEC.loader is not None
bot_supervisor = importlib.util.module_from_spec(_SPEC)
sys.modules["bot_supervisor"] = bot_supervisor
_SPEC.loader.exec_module(bot_supervisor)


@pytest.fixture
def sup_pidfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pidfile = tmp_path / "bot_supervisor.pid"
    monkeypatch.setattr(bot_supervisor, "SUPERVISOR_PID", pidfile)
    return pidfile


def test_acquire_lock_single_instance(sup_pidfile: Path) -> None:
    assert bot_supervisor._acquire_lock() is True
    assert int(sup_pidfile.read_text().strip()) == os.getpid()


def test_acquire_lock_refuses_second_instance(sup_pidfile: Path) -> None:
    assert bot_supervisor._acquire_lock() is True
    assert bot_supervisor._acquire_lock() is False


def test_acquire_lock_reclaims_stale_pidfile(sup_pidfile: Path) -> None:
    sup_pidfile.write_text("999999999\n")
    assert bot_supervisor._lock_is_stale() is True
    assert bot_supervisor._acquire_lock() is True
    assert int(sup_pidfile.read_text().strip()) == os.getpid()


def test_acquire_lock_keeps_live_pidfile(sup_pidfile: Path) -> None:
    sup_pidfile.write_text(f"{os.getpid()}\n")
    assert bot_supervisor._lock_is_stale() is False
    assert bot_supervisor._acquire_lock() is False
    assert int(sup_pidfile.read_text().strip()) == os.getpid()


def test_acquire_lock_handles_corrupt_pidfile(sup_pidfile: Path) -> None:
    sup_pidfile.write_text("not-a-pid\n")
    assert bot_supervisor._lock_is_stale() is True
    assert bot_supervisor._acquire_lock() is True
