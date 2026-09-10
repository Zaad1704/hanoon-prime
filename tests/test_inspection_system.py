"""Processes + identity joint checks."""
import os

from hanoon_prime.inspection import system
from hanoon_prime.inspection.checks import FAIL, OK, UNVERIFIABLE
from hanoon_prime.inspection.ctx import InspectionContext


def _ctx(tmp_path) -> InspectionContext:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.pid_dir.mkdir(parents=True, exist_ok=True)
    return ctx


def test_service_ok_when_pidfile_alive(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    ctx.pid_file("hanoon_prime").write_text(str(os.getpid()))
    r = system.bot_alive(ctx)
    assert r.status == OK
    assert r.evidence.get("pid") == os.getpid()


def test_service_fail_on_stale_pidfile(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    ctx.pid_file("hanoon_prime").write_text("999999")
    assert system.bot_alive(ctx).status == FAIL


def test_service_pgrep_fallback_when_no_pidfile(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(system, "_pgrep", lambda p: [1234])
    assert system.cloudflared_alive(ctx).status == OK


def test_service_fail_when_no_pidfile_and_down(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(system, "_pgrep", lambda p: [])
    assert system.cloudflared_alive(ctx).status == FAIL


def test_single_bot_double_is_fail(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(system, "_pgrep", lambda p: [1, 2])
    assert system.single_bot(ctx).status == FAIL


def test_single_bot_ok(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(system, "_pgrep", lambda p: [123])
    assert system.single_bot(ctx).status == OK


def test_bot_identity_fail_on_stray_cmdline(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    ctx.pid_file("hanoon_prime").write_text(str(os.getpid()))
    monkeypatch.setattr(system, "_cmdline", lambda pid: "/some/other/script.py")
    assert system.bot_from_trusted_checkout(ctx).status == FAIL


def test_bot_identity_ok(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    ctx.pid_file("hanoon_prime").write_text(str(os.getpid()))
    monkeypatch.setattr(
        system, "_cmdline", lambda pid: "/x/.venv/bin/python -m hanoon_prime.cli"
    )
    assert system.bot_from_trusted_checkout(ctx).status == OK


def test_identity_unverifiable_when_bot_down(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    assert system.bot_from_trusted_checkout(ctx).status == UNVERIFIABLE
