"""tests/test_safety_producer — live halt + pause policy (slow-cortex owned).

Safety nets drop to portfolio-level, low-cadence ownership: the slow cortex
feeds them from the account feed and publishes `authorized` into
policy_state. Halt blocks new entries, never stops the bot, notify + journal.
"""

from hanoon_prime.brain.policy.safety import SafetyProducer
from hanoon_prime.immune import (
    CONSECUTIVE_LOSSES_PAUSE,
    DAILY_LOSS_LIMIT,
    MAX_CONCURRENT_POSITIONS,
)

_CAPTURED: list[tuple[str, str]] = []


def make(journal: list | None = None) -> SafetyProducer:
    _CAPTURED.clear()

    def notify(reason: str) -> None:
        _CAPTURED.append(("notify", reason))

    s = SafetyProducer(journal=journal, notify=notify)
    s.begin_call()
    return s


def captured() -> list[tuple[str, str]]:
    return list(_CAPTURED)


def test_defaults_authorized():
    s = make()
    ok, reason = s.authorized()
    assert ok is True and reason == ""


def test_daily_loss_trips_halt():
    s = make()
    s.on_daily_pnl(-(DAILY_LOSS_LIMIT + 1.0))
    ok, reason = s.authorized()
    assert ok is False and "daily" in reason
    assert s.halted is True


def test_consecutive_losses_trips_halt():
    s = make()
    s.on_consecutive_losses(CONSECUTIVE_LOSSES_PAUSE)
    assert s.authorized()[0] is False


def test_position_count_trips_halt():
    s = make()
    s.on_position_count(MAX_CONCURRENT_POSITIONS + 1)
    assert s.authorized()[0] is False


def test_resume_clears_halt():
    s = make()
    s.on_daily_pnl(-(DAILY_LOSS_LIMIT + 1.0))
    assert s.authorized()[0] is False
    s.resume()
    assert s.halted is False
    s.on_daily_pnl(50.0)  # the underlying condition must also clear
    assert s.authorized()[0] is True


def test_disabled_never_halts():
    s = make()
    s.set_enabled(False)
    s.on_daily_pnl(-(DAILY_LOSS_LIMIT * 3))
    assert s.authorized()[0] is True


def test_halt_journals_and_notifies():
    events: list[dict] = []
    s = make(journal=events)
    s.on_daily_pnl(-(DAILY_LOSS_LIMIT + 1.0))
    s.authorized()
    halt_events = [e for e in events if e.get("event") == "halt"]
    assert halt_events, "halt must be journaled"
    assert captured() and captured()[0][0] == "notify"


def test_halt_is_sticky_within_pulse_without_reregging():
    s = make()
    s.on_daily_pnl(-(DAILY_LOSS_LIMIT + 1.0))
    s.authorized()
    s.on_daily_pnl(-(DAILY_LOSS_LIMIT + 50.0))
    s.authorized()
    assert len([c for c in captured() if c[0] == "notify"]) == 1


def test_begin_call_new_pulse_stays_halted_until_resume():
    s = make()
    s.on_position_count(MAX_CONCURRENT_POSITIONS + 1)
    s.authorized()
    s.begin_call()
    ok, reason = s.authorized()
    assert ok is False and reason == "too_many_positions"


def test_reason_contains_pause_reason():
    s = make()
    s.on_position_count(MAX_CONCURRENT_POSITIONS + 1)
    _, reason = s.authorized()
    assert reason != ""
