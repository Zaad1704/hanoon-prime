"""Master session gate: unified ids, window classification, effective_state."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hanoon_prime.brain.policy.trading_policy import TradingConfig  # noqa: E402
from hanoon_prime.monitor.sleep_manager import SESSION_IDS, SleepManager  # noqa: E402

ET = ZoneInfo("America/New_York")


def _dt(day: int, h: int, mi: int) -> datetime:
    return datetime(2026, 9, 14, h, mi, tzinfo=ET)  # Monday 2026-09-14


class TestWindowClassification:
    def test_pre_market_boundary(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 4, 0))
        assert st.active is True
        assert st.session == "pre_market"

    def test_pre_market_just_before_rth(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 9, 29))
        assert st.session == "pre_market"

    def test_rth_starts_at_0930(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 9, 30))
        assert st.active is True
        assert st.session == "rth"

    def test_rth_ends_at_1600(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 15, 59))
        assert st.session == "rth"

    def test_post_market_inactive(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 16, 0))
        assert st.active is False
        assert st.session == "post_market"

    def test_post_market_ends_at_2000(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 19, 59))
        assert st.session == "post_market"

    def test_overnight_inactive(self) -> None:
        st = SleepManager().get_state(now=_dt(14, 2, 0))
        assert st.active is False
        assert st.session == "overnight"

    def test_weekend_inactive(self) -> None:
        st = SleepManager().get_state(now=datetime(2026, 9, 12, 12, 0, tzinfo=ET))
        assert st.active is False
        assert st.session == "weekend"

    def test_ib_disconnected_inactive(self) -> None:
        st = SleepManager().get_state(ib_connected=False, now=_dt(14, 10, 0))
        assert st.active is False

    def test_session_ids_match_config_suffixes(self) -> None:
        cfg = TradingConfig()
        suffixes = {a[len("session_") :] for a in dir(cfg) if a.startswith("session_")}
        assert SESSION_IDS == suffixes


class TestEffectiveState:
    def test_config_disable_turns_system_off(self) -> None:
        cfg = TradingConfig()
        cfg.session_pre_market = False
        st = SleepManager().effective_state(cfg, now=_dt(14, 5, 0))
        assert st.active is False
        assert st.reason == "session_disabled"

    def test_enabled_pre_market_stays_active(self) -> None:
        st = SleepManager().effective_state(TradingConfig(), now=_dt(14, 5, 0))
        assert st.active is True
        assert st.session == "pre_market"

    def test_post_market_off_even_with_config(self) -> None:
        st = SleepManager().effective_state(TradingConfig(), now=_dt(14, 17, 0))
        assert st.active is False


from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from hanoon_prime import ib_cycle as _ibc  # noqa: E402
from hanoon_prime.ib_cycle import BotCycleMixin  # noqa: E402
from hanoon_prime.monitor.sleep_manager import SleepState  # noqa: E402


def _mixin() -> BotCycleMixin:
    mixin = BotCycleMixin.__new__(BotCycleMixin)
    mixin.streamer = MagicMock()
    mixin.streamer.ticker_subs = {}
    mixin.streamer.last_seen = {}
    mixin.streamer.drain_signals.return_value = []
    mixin.streamer.update_bar.return_value = False
    mixin.hippocampus = MagicMock()
    mixin.hippocampus._open_positions = {}
    mixin.hippocampus._consecutive_losses = 0
    mixin.executor = MagicMock()
    mixin.executor.get_newly_closed_trades.return_value = []
    juli = MagicMock()
    juli.brain = MagicMock()
    juli.brain._consolidation = None
    juli._candidates = []
    juli._state = {}
    juli._recent_verdicts = []
    juli.tick.return_value = ([], [])
    juli.budget.get_all_tracked.return_value = set()
    mixin.juli = juli
    mixin.monitor = MagicMock()
    mixin._closing = set()
    mixin._last_beat = 0.0
    mixin._last_policy_sync = 0.0
    mixin._sleeping = False
    mixin.ib = MagicMock()
    mixin.ib.pendingTickers.return_value = []
    mixin._exit_reasons = {}
    mixin.journal = MagicMock()
    return mixin


class TestDeepSleepLoop:
    def test_asleep_cycle_skips_brain(self, monkeypatch) -> None:
        monkeypatch.setattr(
            _ibc._SLEEP_MGR,
            "effective_state",
            lambda enabled, ib_connected=True, now=None: SleepState(
                active=False, session="overnight", reason="Off hours"
            ),
        )
        mixin = _mixin()
        mixin._cycle(0.0, None)
        mixin.juli.tick.assert_not_called()
        mixin.monitor.record_cycle.assert_called()

    def test_awake_cycle_runs_brain(self, monkeypatch) -> None:
        monkeypatch.setattr(
            _ibc._SLEEP_MGR,
            "effective_state",
            lambda enabled, ib_connected=True, now=None: SleepState(
                active=True, session="rth", reason="Market open"
            ),
        )
        mixin = _mixin()
        mixin._cycle(0.0, None)
        mixin.juli.tick.assert_called_once()

    def test_asleep_cycle_still_honors_flatten(self, monkeypatch) -> None:
        monkeypatch.setattr(
            _ibc._SLEEP_MGR,
            "effective_state",
            lambda enabled, ib_connected=True, now=None: SleepState(
                active=False, session="overnight", reason="Off hours"
            ),
        )
        mixin = _mixin()
        mixin._check_manual_flatten = lambda: True  # type: ignore[method-assign]
        mixin._cycle(0.0, None)
        mixin.juli.tick.assert_not_called()
        mock = mixin.monitor.record_cycle
        assert len(mock.call_args_list) >= 1
