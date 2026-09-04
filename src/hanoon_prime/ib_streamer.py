"""hanoon_prime.ib_streamer — IB Gateway streaming data layer.

Manages live market data: reqMktData, reqMktDepth, reqHistoricalData.
Tracks executions and commissions for journal carbon copy.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .eyes import rolling_atr
from .ib_compat import ib
from .immune import DEPTH_ROWS, EDGE_LOOKBACK, LOOKBACK_BARS

log = logging.getLogger(__name__)

@dataclass
class StreamBuffer:
    """Circular buffer holding recent bars for one ticker."""

    ticker: str
    close: list[float] = field(default_factory=list)
    high: list[float] = field(default_factory=list)
    low: list[float] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)
    buy_vol: list[float] = field(default_factory=list)
    bid_sizes: list[float] = field(default_factory=list)
    ask_sizes: list[float] = field(default_factory=list)

    def append(self, close: float, high: float, low: float,
               volume: float, buy_vol: float, bid_size: float, ask_size: float) -> None:
        """Append one bar, trimming to LOOKBACK_BARS."""
        vals = (close, high, low, volume, buy_vol, bid_size, ask_size)
        attrs = [self.close, self.high, self.low, self.volume,
                 self.buy_vol, self.bid_sizes, self.ask_sizes]
        for lst, val in zip(attrs, vals, strict=True):
            lst.append(val)
        for lst in attrs:
            if len(lst) > LOOKBACK_BARS:
                del lst[:-LOOKBACK_BARS]

    def ready(self) -> bool:
        """True if enough bars for Cortex evaluation."""
        return len(self.close) >= EDGE_LOOKBACK

    def arrays(self) -> dict[str, Any]:
        """Return numpy arrays for the brain."""
        keys = ("close", "high", "low", "volume", "buy_volume")
        attrs = (self.close, self.high, self.low, self.volume, self.buy_vol)
        result: dict[str, Any] = dict(zip(keys, [np.array(a) for a in attrs]))
        result["bid_sizes"] = np.array(self.bid_sizes)
        result["ask_sizes"] = np.array(self.ask_sizes)
        return result

class IBStreamer:
    """Manages all IB Gateway streaming subscriptions for one bot."""

    def __init__(self, ib_client: Any) -> None:
        self.ib = ib_client
        self.buffers: dict[str, StreamBuffer] = {}
        self.contracts: dict[str, Any] = {}
        self.ticker_subs: dict[str, Any] = {}
        self.depth_subs: dict[str, Any] = {}
        self._minutely: dict[str, list[Any]] = {}
        self.executions: list[dict[str, Any]] = []
        self.commissions: dict[str, float] = {}

    def subscribe(self, ticker: str) -> None:
        """Subscribe to live market data + order book depth."""
        contract = ib.Stock(ticker, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        self.contracts[ticker] = contract
        self.buffers[ticker] = StreamBuffer(ticker)
        self.ticker_subs[ticker] = self.ib.reqMktData(contract, "", False, False)
        try:
            self.depth_subs[ticker] = self.ib.reqMktDepth(contract, DEPTH_ROWS, False)
        except Exception as e:
            log.warning("DOM unavailable for %s: %s", ticker, e)
            self.depth_subs[ticker] = None
        log.info("Subscribed to %s (mkt data + DOM)", ticker)

    def seed_history(self, ticker: str) -> None:
        """Fetch 1-min historical bars for lookback seeding."""
        contract = self.contracts[ticker]
        bars = self.ib.reqHistoricalData(contract, endDateTime="", durationStr="2 D",
                                          barSizeSetting="1 min", whatToShow="TRADES",
                                          useRTH=True, formatDate=1)
        self.ib.sleep(1)
        buf = self.buffers[ticker]
        for bar in reversed(bars):
            bv = self._est_buy_vol(bar.close, bar.high, bar.low, bar.volume)
            buf.append(bar.close, bar.high, bar.low, bar.volume, bv, bar.close, bar.volume)

    @staticmethod
    def _est_buy_vol(close: float, high: float, low: float, vol: float) -> float:
        """Estimate buy volume fraction from OHLCV."""
        rng = max(high - low, 1e-12)
        return vol * max(0.0, min(1.0, (close - low) / rng))

    def get_last_price(self, ticker: str) -> float:
        """Return the latest trade price from the live IB stream."""
        tk = self.ticker_subs.get(ticker)
        if tk is None:
            return 0.0
        return float(tk.last or tk.close or tk.bid or 0.0)

    def _extract_bar(self, ticker: str) -> Optional[tuple[Any, ...]]:
        """Pull latest OHLCV + DOM snapshot from streaming Tickers."""
        tk = self.ticker_subs.get(ticker)
        if tk is None or not tk.hasBidAsk():
            return None
        close = next((float(v) for v in (tk.close, tk.last) if v and not np.isnan(v)), None)
        if close is None:
            return None
        high = float(tk.high) if (tk.high and not np.isnan(tk.high)) else close
        low = float(tk.low) if (tk.low and not np.isnan(tk.low)) else close
        vol = float(tk.volume or 0)
        bv = self._est_buy_vol(close, high, low, vol)
        dt = self.depth_subs.get(ticker)
        dom_bids = list(dt.domBids) if dt and dt.domBids else []
        dom_asks = list(dt.domAsks) if dt and dt.domAsks else []
        bid_sz = sum(d.size for d in dom_bids[:3]) if dom_bids else float(tk.bidSize or 1)
        ask_sz = sum(d.size for d in dom_asks[:3]) if dom_asks else float(tk.askSize or 1)
        ts = tk.time.timestamp() if tk.time else time.time()
        return close, high, low, vol, bv, bid_sz, ask_sz, ts

    def update_bar(self, ticker: str) -> bool:
        """Aggregate live ticks into 1-min bars. True if a bar was appended."""
        r = self._extract_bar(ticker)
        if r is None:
            return False
        close, high, low, vol, bv, bid, ask, ts = r
        m = int(ts // 60)
        a = self._minutely.get(ticker)
        if a and a[0] == m:
            a[1] = max(a[1], high)
            a[2] = min(a[2], low)
            a[3] = close
            a[4] += vol
            a[5] += bv
            a[6] = bid
            a[7] = ask
            return False
        if a is not None:
            self.buffers[ticker].append(a[3], a[1], a[2], a[4], a[5], a[6], a[7])
        self._minutely[ticker] = [m, high, low, close, vol, bv, bid, ask]
        return a is not None

    def get_arrays(self, ticker: str) -> dict[str, Any]:
        """Return numpy arrays from the buffer for brain consumption."""
        return self.buffers[ticker].arrays()

    def ready(self, ticker: str) -> bool:
        """Check if buffer has enough data."""
        return self.buffers[ticker].ready()

    def buffer_atr(self, ticker: str) -> float:
        """Compute ATR(14) from the current buffer."""
        buf = self.buffers.get(ticker)
        if not buf or len(buf.close) < 2:
            return 1.0
        atr = rolling_atr(np.array(buf.high), np.array(buf.low), np.array(buf.close))
        return max(atr, 1e-8) if not np.isnan(atr) else 1.0

    def record_execution(self, trade: Any, fill: Any) -> None:
        """Record an IB execution (fill) for journal carbon copy."""
        try:
            e = fill.execution
            self.executions.append({"ticker": fill.contract.symbol, "action": e.side,
                                    "shares": e.shares, "price": e.price,
                                    "timestamp": e.time.timestamp() if e.time else time.time()})
        except Exception:
            pass

    def record_commission(self, trade: Any, fill: Any, report: Any) -> None:
        """Record IB commission report for journal carbon copy."""
        try:
            sym = fill.contract.symbol if fill.contract else ""
            self.commissions[sym] = self.commissions.get(sym, 0.0) + float(report.commission)
        except Exception:
            pass

    def cancel_all(self) -> None:
        """Cancel all MD + depth subscriptions."""
        for sub in self.ticker_subs.values():
            try:
                self.ib.cancelMktData(sub)
            except Exception:
                pass
        for sub in (s for s in self.depth_subs.values() if s is not None):
            try:
                self.ib.cancelMktDepth(sub)
            except Exception:
                pass
        self.ticker_subs.clear()
        self.depth_subs.clear()
