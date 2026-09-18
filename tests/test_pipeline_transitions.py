"""tests/test_pipeline_transitions.py — transition-only alerts + vitals log."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hanoon_prime.ib_cycle import BotCycleMixin


def _bot() -> BotCycleMixin:
    """Bare mixin instance with just the transition/vitals surface wired."""
    bot = object.__new__(BotCycleMixin)
    bot.monitor = MagicMock()
    bot.vitals_log = MagicMock()
    bot.vitals_log.record = MagicMock()
    return bot


class TestPipelineTransitions:
    """Alerts fire only on healthy<->broken transitions, never on repeats."""

    def test_first_observation_sets_baseline_without_alert(self):
        bot = _bot()
        bot.monitor.snapshot.return_value = {"failing": {"bars_fresh": "stale"}}
        with (
            patch("hanoon_prime.ib_cycle.error_notify") as err,
            patch("hanoon_prime.ib_cycle.recovered") as rec,
        ):
            bot._notify_pipeline_transitions()
        err.assert_not_called()
        rec.assert_not_called()
        assert bot._pipe_failures_last == {"bars_fresh": "stale"}

    def test_healthy_to_broken_sends_error_notify(self):
        bot = _bot()
        bot._pipe_failures_last = {}
        bot.monitor.snapshot.return_value = {
            "failing": {"brain_advancing": "40 cycles", "bars_fresh": "stale"}
        }
        with (
            patch("hanoon_prime.ib_cycle.error_notify") as err,
            patch("hanoon_prime.ib_cycle.recovered") as rec,
        ):
            bot._notify_pipeline_transitions()
        err.assert_called_once()
        assert "brain_advancing" in err.call_args[0][1]
        rec.assert_not_called()

    def test_broken_to_healthy_sends_recovered(self):
        bot = _bot()
        bot._pipe_failures_last = {"bars_fresh": "stale"}
        bot.monitor.snapshot.return_value = {"failing": {}}
        with (
            patch("hanoon_prime.ib_cycle.error_notify") as err,
            patch("hanoon_prime.ib_cycle.recovered") as rec,
        ):
            bot._notify_pipeline_transitions()
        rec.assert_called_once()
        err.assert_not_called()

    def test_same_state_sends_nothing(self):
        bot = _bot()
        bot._pipe_failures_last = {"bars_fresh": "stale"}
        bot.monitor.snapshot.return_value = {"failing": {"bars_fresh": "stale"}}
        with (
            patch("hanoon_prime.ib_cycle.error_notify") as err,
            patch("hanoon_prime.ib_cycle.recovered") as rec,
        ):
            bot._notify_pipeline_transitions()
        err.assert_not_called()
        rec.assert_not_called()


class TestRecordVitals:
    """_record_vitals persists monitor snapshots to vitals_log."""

    def test_records_snapshot(self):
        bot = _bot()
        bot.monitor.snapshot.return_value = {"healthy": True, "vitals": {}}
        bot._record_vitals()
        bot.vitals_log.record.assert_called_once()

    def test_no_vitals_log_is_silent(self):
        bot = _bot()
        bot.vitals_log = None
        bot._record_vitals()
        bot.monitor.snapshot.assert_not_called()

    def test_snapshot_error_is_swallowed(self):
        bot = _bot()
        bot.monitor.snapshot.side_effect = RuntimeError("boom")
        bot._record_vitals()
        bot.vitals_log.record.assert_not_called()
