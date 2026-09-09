"""Account-feed hygiene: IB's PnL stream yields nan before the first
update, and NaN must never leak into policy_state/telemetry JSON.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hanoon_prime.brain.policy.verdict import ENTER, Verdict
from hanoon_prime.ib_cycle import BotCycleMixin, CycleMeta


def _make_mixin() -> BotCycleMixin:
    """BotCycleMixin with fakes; pnl-relevant attrs only."""
    mixin = BotCycleMixin.__new__(BotCycleMixin)
    mixin.streamer = MagicMock()
    mixin.streamer.ticker_subs = {}
    mixin.streamer.last_seen = {}
    mixin.hippocampus = MagicMock()
    mixin.hippocampus._open_positions = {}
    mixin.hippocampus._consecutive_losses = 0
    mixin.executor = MagicMock()
    mixin.executor.get_newly_closed_trades.return_value = []
    juli = MagicMock()
    juli._candidates = []
    juli._state = {}
    juli.budget.get_all_tracked.return_value = set()
    mixin.juli = juli
    mixin.monitor = MagicMock()
    mixin._closing = set()
    mixin._last_beat = 0.0
    mixin.ib = MagicMock()
    mixin.ib.pendingTickers.return_value = []
    mixin._exit_reasons = {}
    mixin.journal = MagicMock()
    return mixin


class TestAccountFeedSanitization:
    def test_publish_account_feed_sanitizes_nan(self):
        """publish_account_feed coerces nan dailyPnL to 0.0."""
        mixin = _make_mixin()
        pnl = SimpleNamespace(dailyPnL=float("nan"))
        mixin._publish_account_feed(pnl)
        feed = mixin.juli._state["account_feed"]
        assert feed["daily_pnl"] == 0.0
        assert math.isfinite(feed["daily_pnl"])

    def test_publish_account_feed_carries_last_equity_between_syncs(self):
        """Non-sync cycles keep the last known equity (never drop it)."""
        mixin = _make_mixin()
        mixin._account_equity = 123_456.0
        mixin._account_equity_synced = True
        mixin._last_policy_sync = time.monotonic()  # suppress the 30s refresh
        mixin._publish_account_feed(SimpleNamespace(dailyPnL=0.0))
        feed = mixin.juli._state["account_feed"]
        assert feed["equity"] == 123_456.0
        assert feed["equity_synced"] is True

    def test_finish_cycle_sanitizes_nan_pnl(self):
        """finish_cycle never stores nan on the hippocampus."""
        mixin = _make_mixin()
        pnl = SimpleNamespace(dailyPnL=float("nan"))
        meta = CycleMeta(poll=1.0, started=0.0, market_open=False)
        ver = Verdict(ticker="TSLA", action=ENTER)
        mixin._finish_cycle([], [ver], pnl, meta)
        assert mixin.hippocampus._daily_pnl == 0.0
