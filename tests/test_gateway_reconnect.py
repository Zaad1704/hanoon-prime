"""tests/test_gateway_reconnect — verifies _reconnect uses stored params, not ib.host."""

from __future__ import annotations

from hanoon_prime.ib_cycle import BotCycleMixin


class _FakeIB:
    def __init__(self) -> None:
        self._connected = False

    def isConnected(self) -> bool:
        return self._connected


class _FakeStreamer:
    def touch(self, tickers: set) -> None:
        pass


class _FakeHippo:
    def __init__(self) -> None:
        self._open_positions: set = set()


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
