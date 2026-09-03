"""tests/test_ib_streamer.py — tests for 1-minute bar aggregation.

Since ib_insync is not importable in this environment, we use fake/mock
IB client objects. Tests verify:

1. StreamBuffer append + trimming to LOOKBACK_BARS
2. update_bar aggregates sub-second ticks into 1-minute OHLCV bars
3. seed_history requests 1-min bars (not daily — the ATR collapse fix)
4. buffer_atr returns realistic values from aggregated bars
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hanoon_prime.ib_streamer import IBStreamer, StreamBuffer
from hanoon_prime.immune import EDGE_LOOKBACK, LOOKBACK_BARS


class FakeTicker:
    """Mimics ib_insync Ticker for testing."""

    def __init__(
        self,
        close,
        high,
        low,
        volume=100,
        bid_size=10,
        ask_size=10,
        has_bidass=True,
        ts=60_000.0,
    ):
        self._close = float(close)
        self._high = float(high)
        self._low = float(low)
        self.volume = volume
        self.bidSize = bid_size
        self.askSize = ask_size
        self._hb = has_bidass
        self._ts = ts
        self.last = float(close)

    def hasBidAsk(self):
        return self._hb

    @property
    def close(self):
        return self._close

    @property
    def high(self):
        return self._high

    @property
    def low(self):
        return self._low

    @property
    def time(self):
        m = MagicMock()
        m.timestamp.return_value = self._ts
        return m


@pytest.fixture
def streamer():
    """IBStreamer with a fake IB client and pre-subscribed ticker."""
    s = IBStreamer(MagicMock())
    s.buffers["TSLA"] = StreamBuffer("TSLA")
    s.ticker_subs["TSLA"] = FakeTicker(100.0, 100.5, 99.5, volume=100, ts=60_000.0)
    s.depth_subs["TSLA"] = None
    return s


class TestStreamBuffer:
    """StreamBuffer basic operations."""

    def test_append_trims_to_lookback(self):
        buf = StreamBuffer("T")
        for i in range(LOOKBACK_BARS + 10):
            buf.append(
                float(i),
                float(i + 0.1),
                float(i - 0.1),
                1.0,
                0.5,
                1.0,
                1.0,
            )
        assert len(buf.close) == LOOKBACK_BARS
        assert len(buf.high) == LOOKBACK_BARS
        assert len(buf.low) == LOOKBACK_BARS

    def test_ready_threshold(self):
        buf = StreamBuffer("T")
        assert not buf.ready()
        for _ in range(EDGE_LOOKBACK):
            buf.append(1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0)
        assert buf.ready()

    def test_ready_false_after_edge_lookback_minus_1(self):
        buf = StreamBuffer("T")
        for _ in range(EDGE_LOOKBACK - 1):
            buf.append(1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0)
        assert not buf.ready()

    def test_arrays_returns_numpy(self):
        buf = StreamBuffer("T")
        buf.append(100.0, 101.0, 99.0, 1000.0, 500.0, 10.0, 10.0)
        arrs = buf.arrays()
        assert "close" in arrs
        assert "high" in arrs
        assert "low" in arrs
        assert "volume" in arrs
        assert "bid_sizes" in arrs
        assert "ask_sizes" in arrs


class TestMinuteAggregation:
    """1-minute bar aggregation — the core ATR collapse fix."""

    def test_first_tick_starts_minute_no_flush(self, streamer):
        """First tick in a minute starts accumulation, no bar flushed."""
        s = streamer
        # ts 60_010 → minute 60_010 // 60 = 1000
        s.ticker_subs["TSLA"] = FakeTicker(100.0, 101.0, 99.0, ts=60_010.0)
        assert s.update_bar("TSLA") is False
        assert len(s.buffers["TSLA"].close) == 0

    def test_same_minute_accumulates(self, streamer):
        """Multiple ticks in the same minute are accumulated, not flushed."""
        s = streamer
        s.ticker_subs["TSLA"] = FakeTicker(100.0, 101.0, 99.0, volume=100, ts=60_000.0)
        s.update_bar("TSLA")  # minute 1000, first tick
        s.ticker_subs["TSLA"] = FakeTicker(101.0, 102.0, 99.5, volume=200, ts=60_010.0)
        assert s.update_bar("TSLA") is False  # same minute — accumulated
        assert len(s.buffers["TSLA"].close) == 0

    def test_minute_cross_flushes_bar(self, streamer):
        """When the minute boundary crosses, the completed bar is flushed."""
        s = streamer
        # Minute 1000: ts=60_000
        s.ticker_subs["TSLA"] = FakeTicker(100.0, 101.0, 99.0, volume=100, ts=60_000.0)
        s.update_bar("TSLA")
        # Minute 1001: ts=60_060
        s.ticker_subs["TSLA"] = FakeTicker(102.0, 103.0, 101.0, volume=200, ts=60_060.0)
        assert s.update_bar("TSLA") is True  # previous bar flushed
        assert len(s.buffers["TSLA"].close) == 1

    def test_ohlc_from_aggregated_ticks(self, streamer):
        """Aggregated bar OHLC reflects min/max/close of constituent ticks."""
        s = streamer
        # Minute 1000: tick 1 (close=100, high=105, low=95, vol=100)
        s.ticker_subs["TSLA"] = FakeTicker(100.0, 105.0, 95.0, volume=100, ts=60_000.0)
        s.update_bar("TSLA")
        # Minute 1000: tick 2 (close=103, high=104, low=98, vol=200) — same minute
        s.ticker_subs["TSLA"] = FakeTicker(103.0, 104.0, 98.0, volume=200, ts=60_020.0)
        s.update_bar("TSLA")  # accumulated
        # Minute 1001: new minute — flush
        s.ticker_subs["TSLA"] = FakeTicker(105.0, 107.0, 104.0, volume=10, ts=60_060.0)
        s.update_bar("TSLA")
        buf = s.buffers["TSLA"]
        assert len(buf.close) == 1
        assert buf.close[0] == 103.0  # last close of minute 1000
        assert buf.high[0] == 105.0  # max(105, 104)
        assert buf.low[0] == 95.0  # min(95, 98)
        assert buf.volume[0] == 300.0  # 100 + 200

    def test_atr_from_aggregated_bars_not_collapsed(self, streamer):
        """ATR from real 1-min bars must be meaningful, not ~0.001."""
        s = streamer
        base = 60_000.0
        for i in range(EDGE_LOOKBACK + 5):
            ts = base + i * 60  # each bar in a different minute
            close = 100.0 + i * 0.5
            s.ticker_subs["TSLA"] = FakeTicker(
                close, close + 2.0, close - 2.0, volume=1000, ts=ts
            )
            s.update_bar("TSLA")
        atr = s.buffer_atr("TSLA")
        assert atr > 0.5  # realistic volatility, not 0.001

    def test_buffer_trims_to_lookback_after_many_bars(self, streamer):
        """After feeding many 1-min bars, buffer stays at LOOKBACK_BARS."""
        s = streamer
        base = 60_000.0
        for i in range(LOOKBACK_BARS + 10):
            ts = base + i * 60
            s.ticker_subs["TSLA"] = FakeTicker(100.0, 101.0, 99.0, volume=100, ts=ts)
            s.update_bar("TSLA")
        assert len(s.buffers["TSLA"].close) <= LOOKBACK_BARS


class TestSeedHistory:
    """seed_history must use 1-minute bars (ATR collapse fix)."""

    @patch("hanoon_prime.ib_streamer.ib")
    def test_seed_history_requests_1min_bars(self, mock_ib_mod):
        """seed_history must request '1 min' bars with '2 D' duration."""
        fake_ib = MagicMock()
        mock_ib_mod.Stock = MagicMock(return_value=MagicMock())
        s = IBStreamer(fake_ib)
        s.contracts["TSLA"] = MagicMock()
        s.buffers["TSLA"] = StreamBuffer("TSLA")
        fake_ib.reqHistoricalData.return_value = []
        s.seed_history("TSLA")
        kwargs = fake_ib.reqHistoricalData.call_args.kwargs
        assert kwargs["barSizeSetting"] == "1 min"
        assert kwargs["durationStr"] == "2 D"

    @patch("hanoon_prime.ib_streamer.ib")
    def test_seed_history_does_not_request_daily_bars(self, mock_ib_mod):
        """Ensure the old '1 day' setting is gone."""
        fake_ib = MagicMock()
        mock_ib_mod.Stock = MagicMock(return_value=MagicMock())
        s = IBStreamer(fake_ib)
        s.contracts["TSLA"] = MagicMock()
        s.buffers["TSLA"] = StreamBuffer("TSLA")
        fake_ib.reqHistoricalData.return_value = []
        s.seed_history("TSLA")
        kwargs = fake_ib.reqHistoricalData.call_args.kwargs
        assert kwargs["barSizeSetting"] != "1 day"


class TestEdgeCases:
    """update_bar edge cases."""

    def test_unknown_ticker_returns_false(self, streamer):
        assert streamer.update_bar("NOPE") is False

    def test_no_bid_ask_returns_false(self):
        s = IBStreamer(MagicMock())
        s.buffers["TSLA"] = StreamBuffer("TSLA")
        s.ticker_subs["TSLA"] = FakeTicker(100.0, 100.5, 99.5, has_bidass=False)
        assert s.update_bar("TSLA") is False

    def test_buffer_atr_returns_default_when_empty(self):
        s = IBStreamer(MagicMock())
        assert s.buffer_atr("NOPE") == 1.0
