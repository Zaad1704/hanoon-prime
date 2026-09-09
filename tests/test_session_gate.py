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
