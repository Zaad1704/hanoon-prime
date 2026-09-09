"""tests/test_trade_notify.py — hold milestones + flat-book postmortem notify."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hanoon_prime._telegram import postmortem, trade_closed
from hanoon_prime.ib_cycle import HOLD_FIRST_MIN, HOLD_REPEAT_MIN, BotCycleMixin


def _fake_bot() -> BotCycleMixin:
    """Bare mixin instance with just the attrs _notify_holds touches."""
    bot = object.__new__(BotCycleMixin)
    bot._hold_notified = {}
    bot.hippocampus = MagicMock()
    bot.juli = MagicMock()
    return bot


class TestNotifyHolds:
    """_notify_holds emits milestone messages for open positions."""

    def test_first_milestone_notifies(self):
        bot = _fake_bot()
        pos = MagicMock()
        pos.direction = 1
        bot.hippocampus._open_positions = {"TSLA": pos}
        bot.juli.brain.exits.hold_minutes.return_value = HOLD_FIRST_MIN + 1
        with patch("hanoon_prime.ib_cycle.trade_hold") as fn:
            bot._notify_holds()
        fn.assert_called_once()
        assert fn.call_args[0][:2] == ("TSLA", HOLD_FIRST_MIN + 1)
        assert bot._hold_notified["TSLA"] == HOLD_FIRST_MIN + 1

    def test_no_notify_before_milestone(self):
        bot = _fake_bot()
        pos = MagicMock()
        pos.direction = -1
        bot.hippocampus._open_positions = {"TSLA": pos}
        bot.juli.brain.exits.hold_minutes.return_value = HOLD_FIRST_MIN - 1
        with patch("hanoon_prime.ib_cycle.trade_hold") as fn:
            bot._notify_holds()
        fn.assert_not_called()

    def test_repeats_after_interval(self):
        bot = _fake_bot()
        pos = MagicMock()
        pos.direction = 1
        bot.hippocampus._open_positions = {"TSLA": pos}
        bot._hold_notified["TSLA"] = 15.0
        bot.juli.brain.exits.hold_minutes.return_value = 15.0 + HOLD_REPEAT_MIN + 1
        with patch("hanoon_prime.ib_cycle.trade_hold") as fn:
            bot._notify_holds()
        fn.assert_called_once()
        assert bot._hold_notified["TSLA"] == 15.0 + HOLD_REPEAT_MIN + 1

    def test_empty_book_is_resilient(self):
        bot = _fake_bot()
        bot.hippocampus._open_positions = {}
        bot._notify_holds()  # no positions → no raise


class TestSendPostmortem:
    """_send_postmortem forwards HALIM's insight verbatim when flat."""

    def test_sends_verbatim_insight(self):
        bot = _fake_bot()
        insight = {"insight": "capture early gains", "regime": "rth"}
        bot.juli.brain.state.get.return_value = insight
        with patch("hanoon_prime.ib_cycle.postmortem") as fn:
            bot._send_postmortem()
        fn.assert_called_once_with(insight)

    def test_skips_when_empty(self):
        bot = _fake_bot()
        bot.juli.brain.state.get.return_value = {}
        with patch("hanoon_prime.ib_cycle.postmortem") as fn:
            bot._send_postmortem()
        fn.assert_not_called()


class TestReflectClosedPostmortem:
    """_reflect_closed posts the postmortem exactly when the book goes flat."""

    def test_flat_after_batch_sends_postmortem(self):
        bot = _fake_bot()
        bot.executor = MagicMock()
        bot.executor.get_newly_closed_trades.return_value = [
            {
                "ticker": "TSLA",
                "pnl": 12.0,
                "return_pct": 1.2,
                "direction": 1,
                "source": "ib_fill",
            }
        ]
        bot.juli.brain.state.get.return_value = {"insight": "done"}
        bot.hippocampus._open_positions = {}
        bot.streamer = MagicMock()
        bot._closing = set()
        bot._watched = set()
        bot._exit_reasons = {}
        bot._hold_notified = {}
        with patch("hanoon_prime.ib_cycle.postmortem") as fn:
            bot._reflect_closed()
        fn.assert_called_once_with({"insight": "done"})

    def test_positions_left_skips_postmortem(self):
        bot = _fake_bot()
        bot.executor = MagicMock()
        bot.executor.get_newly_closed_trades.return_value = [
            {
                "ticker": "TSLA",
                "pnl": -5.0,
                "return_pct": -0.5,
                "direction": 1,
                "source": "ib_fill",
            }
        ]
        bot.juli.brain.state.get.return_value = {"insight": "done"}
        pos = MagicMock()
        pos.direction = 1
        bot.hippocampus._open_positions = {"NVDA": pos}
        bot.streamer = MagicMock()
        bot._closing = set()
        bot._watched = set()
        bot._exit_reasons = {}
        bot._hold_notified = {}
        with patch("hanoon_prime.ib_cycle.postmortem") as fn:
            bot._reflect_closed()
        fn.assert_not_called()


class TestHalimSeparateChat:
    """HALIM post-mortems route to their own chat when configured."""

    def test_postmortem_routes_to_halim_chat(self, monkeypatch):
        monkeypatch.setenv("HALIM_TELEGRAM_CHAT_ID", "-100123456789")
        with patch("hanoon_prime._telegram.send") as fn:
            postmortem({"insight": "done"})
        fn.assert_called_once()
        args, kwargs = fn.call_args
        assert kwargs["chat_id"] == "-100123456789"
        assert "HALIM POST-MORTEM" in args[0]

    def test_postmortem_falls_back_to_main_chat(self, monkeypatch):
        from hanoon_prime._telegram import _get_chat_id

        monkeypatch.delenv("HALIM_TELEGRAM_CHAT_ID", raising=False)
        with patch("hanoon_prime._telegram.send") as fn:
            postmortem({"insight": "done"})
        args, kwargs = fn.call_args
        assert kwargs["chat_id"] == _get_chat_id()  # main chat fallback


class TestTradeClosedDetails:
    """trade_closed keeps the IB P&L line and appends extra context."""

    def test_trade_closed_appends_extra(self):
        with patch("hanoon_prime._telegram.send") as fn:
            trade_closed(
                "TSLA",
                "LONG",
                0.1061,
                reason="target",
                extra="Account $242,783 | IB day -554.00\nJULI WR 66.5% (n=200)",
            )
        body = fn.call_args[0][0]
        assert "✅ WIN TSLA" in body
        assert "P&L: $+0.1061" in body
        assert "Reason: target" in body
        assert "Account $242,783" in body
        assert "IB day -554.00" in body
        assert "JULI WR 66.5% (n=200)" in body
