"""Seeding regression: every desired ticker must backfill ≥20 bars.

The bot was only seeding tickers that were *new subscriptions in the same
cycle* (`missing`). A ticker that was already subscribed — or whose first
seed attempt failed — was never backfilled and silently dropped to
live-bars-only data, so freshly-scanned candidates stayed at ~5 bars
during premarket.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

from hanoon_prime.ib_cycle import SEED_RETRY_MAX, BotCycleMixin


class _Tk:
    """Dummy ticker sub object."""


def _make_mixin() -> BotCycleMixin:
    """BotCycleMixin instance with the minimal attrs _sync_subs touches."""
    mixin = BotCycleMixin.__new__(BotCycleMixin)
    mixin.streamer = MagicMock()
    mixin.streamer.ticker_subs = {"BABA": _Tk()}
    mixin.streamer.last_seen = {}
    mixin.hippocampus = MagicMock()
    mixin.hippocampus._open_positions = {}
    mixin.executor = MagicMock()
    juli = MagicMock()
    juli._candidates = [MagicMock(symbol="BABA")]
    juli.budget.get_all_tracked.return_value = {"BABA"}
    mixin.juli = juli
    return mixin


class TestSeedBackfill:
    def test_already_subscribed_candidate_is_seeded(self):
        """A candidate that is past its first subscription still gets history."""
        mixin = _make_mixin()
        mixin.streamer.seed_history = MagicMock()
        mixin._sync_subs()
        mixin.streamer.seed_history.assert_called_once_with("BABA")

    def test_failed_seed_is_retried(self):
        """A transient seed failure must not permanently strand a ticker."""
        mixin = _make_mixin()
        calls = 0

        def _flaky(ticker: str) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("pacing")
            mixin.streamer.seed_history_success = True

        mixin.streamer.seed_history = MagicMock(side_effect=_flaky)
        mixin._sync_subs()
        assert calls == 1
        assert not mixin.__dict__.get("_seeded_subs")
        mixin._sync_subs()
        assert calls == 2
        assert mixin.__dict__.get("_seeded_subs") == {"BABA"}
        mixin.streamer.seed_history.assert_has_calls([call("BABA"), call("BABA")])

    def test_failed_seed_gives_up_after_bound(self):
        """Persistent failure stops after SEED_RETRY_MAX (no hammering)."""
        mixin = _make_mixin()
        mixin.streamer.seed_history = MagicMock(side_effect=RuntimeError("pacing"))
        for _ in range(SEED_RETRY_MAX + 2):
            mixin._sync_subs()
        assert mixin.streamer.seed_history.call_count == SEED_RETRY_MAX
        assert mixin.__dict__.get("_seeded_subs") == {"BABA"}
