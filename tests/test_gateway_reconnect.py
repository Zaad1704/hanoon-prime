"""tests/test_gateway_reconnect — verifies _reconnect uses stored params, not ib.host."""

from __future__ import annotations

import time

from hanoon_prime import ib_cycle
from hanoon_prime.ib_cycle import GATEWAY_FEED_STALE_SECS, BotCycleMixin


class _FakeIB:
    def __init__(self) -> None:
        self._connected = False

    def isConnected(self) -> bool:
        return self._connected


class _FakeStreamer:
    def __init__(self) -> None:
        self.last_data_ts: dict[str, float] = {}

    def touch(self, tickers: set) -> None:
        pass


class _FakeHippo:
    def __init__(self) -> None:
        self._open_positions: set = set()


class _FakeState:
    def __init__(self, active: bool) -> None:
        self.active = active


class _SleepProxy:
    """Stand-in for ib_cycle._SLEEP_MGR (active session = expect live data)."""

    def __init__(self, active: bool) -> None:
        self.active = active

    def effective_state(self, _cfg) -> _FakeState:
        return _FakeState(self.active)


class FakeBot(BotCycleMixin):
    """Minimal host implementing the reconnect protocol for testing."""

    def __init__(self) -> None:
        self.ib = _FakeIB()
        self.streamer = _FakeStreamer()
        self.hippocampus = _FakeHippo()
        self._last_conn: tuple[str, int, int] = ("127.0.0.1", 4002, 7)
        self._gw_was_connected = True
        self._gw_attempts = 0
        self.__connect_calls: list[tuple[str, int, int]] = []
        self._fake_connect_exc: Exception | None = None

    # stubs wired by test
    def connect(self, _host: str, _port: int, _client_id: int) -> None:
        self.__connect_calls.append((_host, _port, _client_id))
        if self._fake_connect_exc is not None:
            raise self._fake_connect_exc

    # expose for assertions
    @property
    def connect_calls(self) -> list[tuple[str, int, int]]:
        return list(self.__connect_calls)

    # touch called on success
    def _resubscribe_all(self) -> None:
        pass


def test_reconnect_uses_last_conn_params() -> None:
    bot = FakeBot()
    bot._last_conn = ("10.0.0.1", 4001, 13)
    ok = bot._reconnect()
    assert ok is False  # connect doesn't raise, but isConnected still False
    assert bot.connect_calls == [("10.0.0.1", 4001, 13)]


def test_reconnect_returns_false_on_connection_error() -> None:
    bot = FakeBot()
    bot._fake_connect_exc = ConnectionError("refused")
    ok = bot._reconnect()
    assert ok is False
    assert len(bot.connect_calls) == 1


def test_reconnect_success_touches_streamer() -> None:
    class TouchTracker(_FakeStreamer):
        touched = False

        def touch(self, _tickers: set) -> None:
            self.touch_tracker = True

    tracker = TouchTracker()
    bot = FakeBot()
    bot.streamer = tracker

    # make the "connect" succeed by flipping isConnected
    def _connect(_host: str, _port: int, _client_id: int) -> None:
        bot.ib._connected = True

    bot.connect = _connect  # type: ignore[assignment]
    ok = bot._reconnect()
    assert ok is True
    assert getattr(tracker, "touch_tracker", False) is True


def test_feed_stale_skips_during_inactive_session(monkeypatch) -> None:
    monkeypatch.setattr(ib_cycle, "_SLEEP_MGR", _SleepProxy(active=False))
    bot = FakeBot()
    bot.streamer.last_data_ts["AAPL"] = time.time() - GATEWAY_FEED_STALE_SECS - 60
    assert bot._feed_stale() is False


def test_feed_stale_true_when_data_silent_in_active_session(monkeypatch) -> None:
    monkeypatch.setattr(ib_cycle, "_SLEEP_MGR", _SleepProxy(active=True))
    bot = FakeBot()
    bot.streamer.last_data_ts["AAPL"] = time.time() - GATEWAY_FEED_STALE_SECS - 10
    assert bot._feed_stale() is True


def test_feed_stale_false_with_fresh_data(monkeypatch) -> None:
    monkeypatch.setattr(ib_cycle, "_SLEEP_MGR", _SleepProxy(active=True))
    bot = FakeBot()
    bot.streamer.last_data_ts["AAPL"] = time.time()
    assert bot._feed_stale() is False


def test_feed_stale_false_when_nothing_received(monkeypatch) -> None:
    monkeypatch.setattr(ib_cycle, "_SLEEP_MGR", _SleepProxy(active=True))
    bot = FakeBot()
    assert bot._feed_stale() is False


def test_supervise_forces_reconnect_on_stale_feed(monkeypatch) -> None:
    monkeypatch.setattr(ib_cycle, "_SLEEP_MGR", _SleepProxy(active=True))
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    bot9 = FakeBot()
    bot9.ib._connected = True  # socket "up" but silent at IB level
    bot9.streamer.last_data_ts["AAPL"] = time.time() - GATEWAY_FEED_STALE_SECS - 5
    resubscribed: list[str] = []

    def _connect(_host: str, _port: int, _client_id: int) -> None:
        bot9.ib._connected = True

    def _resubscribe() -> None:
        resubscribed.append("x")

    bot9.connect = _connect  # type: ignore[assignment]
    bot9._resubscribe_all = _resubscribe  # type: ignore[method-assign]
    bot9._supervise_gateway()
    assert resubscribed, "stale-but-connected feed must be force-reconnected"
    assert bot9._gw_was_connected is True  # re-set by successful _reconnect()


def test_supervise_skips_reconnect_when_feed_fresh(monkeypatch) -> None:
    monkeypatch.setattr(ib_cycle, "_SLEEP_MGR", _SleepProxy(active=True))
    bot = FakeBot()
    bot.ib._connected = True
    bot.streamer.last_data_ts["AAPL"] = time.time()
    called: list[str] = []

    def _reconnect() -> bool:
        called.append("reconnect")
        return True

    bot._reconnect = _reconnect  # type: ignore[method-assign]
    bot._supervise_gateway()
    assert called == []
    assert bot._gw_was_connected is True
