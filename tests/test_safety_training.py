"""tests/test_safety_training.py — Phase 1 safety-net hardening.

Regression tests for the user-directed training kill-switch bypass,
the fail-closed paper-only guard, persisted halt/latch state, the
implemented 60-minute auto-resume, and kill flattening.

Hermetic: the autouse fixture below redirects the safety state file to
tmp_path so these tests never touch the repo runtime/safety_state.json.
"""

from __future__ import annotations

import logging
import time

import pytest

from hanoon_prime import immune
from hanoon_prime.brain.policy import safety as safety_mod
from hanoon_prime.brain.policy.safety import SafetyProducer


@pytest.fixture(autouse=True)
def _hermetic_safety_state(tmp_path, monkeypatch):
    """Redirect the persisted safety state into the per-test tmp dir."""
    monkeypatch.setattr(
        safety_mod, "_SAFETY_STATE_FILE", tmp_path / "safety_state.json"
    )


def _producer(**kwargs):
    kwargs.setdefault("notify", lambda _reason: None)
    return SafetyProducer(**kwargs)


# ── FIX-2026-09-23-01: named training bypass ───────────────────────────


def test_bypass_flag_is_named_and_user_directed():
    """Regression: FIX-2026-09-23-01"""
    # The user explicitly disabled the $500 kill for paper training
    # (2026-09-23). The override must be a named literal, never silent.
    assert immune.TRAINING_KILL_BYPASS is True


def test_bypass_logs_loud_startup_warning(caplog):
    """Regression: FIX-2026-09-23-01"""
    with caplog.at_level(logging.CRITICAL, logger="hanoon_prime.brain.policy.safety"):
        SafetyProducer(notify=lambda _r: None)
    assert "TRAINING_KILL_BYPASS IS ACTIVE" in caplog.text


def test_bypass_exposed_on_producer():
    """Regression: FIX-2026-09-23-01"""
    assert _producer().kill_bypass is True


def test_kill_skipped_while_bypassed_other_halts_apply():
    """Regression: FIX-2026-09-23-01"""
    s = _producer(enabled=True)
    s.on_daily_pnl(-600.0)
    ok, reason = s.authorized()
    assert s.latched is False  # the $500 kill did NOT latch
    # -$600 also breaches the -$200 halt, so entries stay blocked anyway.
    assert (ok, reason) == (False, "daily_loss_limit")


# ── FIX-2026-09-23-02: fail-closed paper-only guard ────────────────────


def test_fail_closed_refuses_non_paper_account():
    """Regression: FIX-2026-09-23-02"""
    with pytest.raises(RuntimeError, match="TRAINING_KILL_BYPASS"):
        immune.assert_paper_only_for_training(
            account="DU1234567", port=immune.IB_PAPER_PORT
        )


def test_fail_closed_refuses_live_port():
    """Regression: FIX-2026-09-23-02"""
    with pytest.raises(RuntimeError, match="TRAINING_KILL_BYPASS"):
        immune.assert_paper_only_for_training(account="PAPER", port=immune.IB_LIVE_PORT)


def test_fail_closed_allows_paper():
    """Regression: FIX-2026-09-23-02"""
    immune.assert_paper_only_for_training(
        account="PAPER", port=immune.IB_PAPER_PORT
    )  # must not raise


# ── FIX-2026-09-23-03: halt/latch persist across restarts ───────────────


def test_halt_persists_across_restart():
    """Regression: FIX-2026-09-23-03"""
    s = _producer(enabled=True)
    s.on_consecutive_losses(3)
    assert s.authorized() == (False, "consecutive_losses")
    assert safety_mod._SAFETY_STATE_FILE.exists()
    s2 = _producer(enabled=True)
    assert s2.halted is True
    assert s2.pause_reason == "consecutive_losses"
    assert s2.authorized() == (False, "consecutive_losses")


def test_kill_latch_persists_and_survives_resume():
    """Regression: FIX-2026-09-23-03"""
    s = _producer(enabled=True)
    s.kill_bypass = False  # exercise the real kill path
    s.on_daily_pnl(-600.0)
    assert s.authorized() == (False, "kill_daily_loss_limit")
    assert s.latched is True
    s2 = _producer(enabled=True)
    assert s2.latched is True
    s2.resume()
    assert s2.latched is True  # resume() never clears a kill latch
    assert s2.authorized()[0] is False


def test_release_kill_clears_persisted_latch():
    """Regression: FIX-2026-09-23-03"""
    s = _producer(enabled=True)
    s.kill_bypass = False
    s.on_daily_pnl(-600.0)
    s.authorized()
    assert s.latched is True
    s.release_kill()
    s2 = _producer(enabled=True)
    assert s2.latched is False
    assert s2.halted is False


# ── FIX-2026-09-23-04: kill flattens positions ─────────────────────────


def test_kill_hook_cancels_and_flattens():
    """Regression: FIX-2026-09-23-04"""
    from hanoon_prime.ib_adapter import IBStreamingBot

    bot = IBStreamingBot.__new__(IBStreamingBot)  # no IB connection needed
    calls: list = []

    class FakeExecutor:
        def cancel_all(self):
            calls.append("cancel_all")

        def close_all_positions(self, streamer):
            calls.append(("flatten", streamer))
            return 2

    sentinel = object()
    bot.executor = FakeExecutor()
    bot.streamer = sentinel
    bot._on_kill("kill_daily_loss_limit")
    assert calls[0] == "cancel_all"
    assert calls[1] == ("flatten", sentinel)


def test_safety_kill_runs_hook_once():
    """Regression: FIX-2026-09-23-04"""
    s = _producer(enabled=True)
    s.kill_bypass = False
    seen: list = []
    s.on_kill(seen.append)
    s.on_daily_pnl(-600.0)
    s.authorized()
    assert seen == ["kill_daily_loss_limit"]


# ── FIX-2026-09-23-05: 60-minute auto-resume ───────────────────────────


def test_halt_auto_resumes_after_pause_minutes():
    """Regression: FIX-2026-09-23-05"""
    s = _producer(enabled=True)
    s.on_consecutive_losses(3)
    assert s.authorized() == (False, "consecutive_losses")
    # Streak cleared by the operator; 61 minutes elapsed since the halt.
    s.on_consecutive_losses(0)
    s._halt_started_at = time.time() - (immune.PAUSE_DURATION_MIN * 60 + 60)
    assert s.authorized() == (True, "")
    assert s.halted is False


def test_halt_does_not_expire_early():
    """Regression: FIX-2026-09-23-05"""
    s = _producer(enabled=True)
    s.on_consecutive_losses(3)
    s.authorized()
    s.on_consecutive_losses(0)
    s._halt_started_at = time.time() - 30 * 60  # only 30 min elapsed
    ok, _reason = s.authorized()
    assert ok is False
    assert s.halted is True


def test_kill_latch_never_auto_resumes():
    """Regression: FIX-2026-09-23-05"""
    s = _producer(enabled=True)
    s.kill_bypass = False
    s.on_daily_pnl(-600.0)
    s.authorized()
    assert s.latched is True
    s._halt_started_at = time.time() - 25 * 3600  # 25 hours ago
    ok, _reason = s.authorized()
    assert ok is False
    assert s.latched is True


def test_expired_halt_restores_then_auto_resumes():
    """Regression: FIX-2026-09-23-05"""
    s = _producer(enabled=True)
    s.on_consecutive_losses(3)
    s.authorized()
    s.on_consecutive_losses(0)
    # Simulate a restart 2h later: the persisted halt is already expired.
    s._halt_started_at = time.time() - 2 * 3600
    s._persist_state()
    s2 = _producer(enabled=True)
    assert s2.halted is True  # restored as halted...
    assert s2.authorized() == (True, "")  # ...but expires on first check
    assert s2.halted is False
