"""hanoon_prime.tape — rolling Time & Sales buffer for MM absorption.

Per-ticker print + quote rings with signed CVD windows. Classification
uses print price vs concurrent bid/ask (aggressive buy at/through the
ask, aggressive sell at/through the bid). Pure data layer — detection
lives in ``absorption.py``. No verdict strings (R1).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .brain.learning_config import (
    ABSORPTION_BID_HOLD_POLLS,
    ABSORPTION_CVD_FAST,
    ABSORPTION_CVD_SLOW,
    TAPE_PRINT_MAXLEN,
    TAPE_QUOTE_MAXLEN,
)


@dataclass(frozen=True)
class Print:
    """One trade print with aggressor side from L1 classification."""

    ts: float
    price: float
    size: float
    side: int  # +1 aggressive buy, -1 aggressive sell, 0 unknown/mid


@dataclass
class TapeBuffer:
    """Rolling prints + quotes for one ticker (circular deques)."""

    ticker: str
    prints: deque[Print] = field(
        default_factory=lambda: deque(maxlen=TAPE_PRINT_MAXLEN)
    )
    quotes: deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=TAPE_QUOTE_MAXLEN)
    )  # (ts, bid, ask)
    _last_print_ts: float = 0.0

    def record_quote(self, ts: float, bid: float, ask: float) -> None:
        """Append one bid/ask snapshot (drives hold + CVD classification)."""
        if bid > 0.0 and ask > 0.0:
            self.quotes.append((ts, bid, ask))

    def record_print(
        self, ts: float, price: float, size: float, bid: float, ask: float
    ) -> None:
        """Append one print, classifying aggressor vs concurrent L1."""
        if price <= 0.0 or size <= 0.0:
            return
        if ts <= self._last_print_ts:
            return  # drop non-monotonic / duplicate ticks
        self._last_print_ts = ts
        side = 0
        if bid > 0.0 and ask > 0.0:
            if price >= ask:
                side = 1
            elif price <= bid:
                side = -1
        self.prints.append(Print(ts, price, size, side))

    def cvd(self, window_sec: float, now: float | None = None) -> float:
        """Signed cumulative volume delta over the window in [-1, +1].

        +1 = all aggressive buy volume, -1 = all aggressive sell volume.
        Empty window (or only mid prints) returns 0.0.
        """
        now = time.time() if now is None else now
        cutoff = now - window_sec
        buy = sell = 0.0
        for p in self.prints:
            if p.ts < cutoff:
                continue
            if p.side > 0:
                buy += p.size
            elif p.side < 0:
                sell += p.size
        total = buy + sell
        if total <= 0.0:
            return 0.0
        return (buy - sell) / total

    def side_volume(
        self, side: int, window_sec: float, now: float | None = None
    ) -> float:
        """Absolute print volume for one aggressor side over the window."""
        now = time.time() if now is None else now
        cutoff = now - window_sec
        return sum(p.size for p in self.prints if p.ts >= cutoff and p.side == side)

    def level_held(self, side: int, now: float | None = None) -> bool:
        """True when the last N quote polls kept the level on `side`.

        ``side > 0`` checks the ask (buy-side hold); ``side < 0`` checks
        the bid (sell-side hold). Fewer polls than the hold length → False
        (not enough evidence of absorption).
        """
        now = time.time() if now is None else now
        n = ABSORPTION_BID_HOLD_POLLS
        if len(self.quotes) < n:
            return False
        recent = list(self.quotes)[-n:]
        if recent[-1][0] < now - ABSORPTION_CVD_FAST:
            return False  # stale quotes — not a live level
        idx = 2 if side > 0 else 1  # ask or bid column
        levels = [q[idx] for q in recent]
        if any(x <= 0.0 for x in levels):
            return False
        if side < 0:
            return levels[-1] >= levels[0]  # bid not broken down
        return levels[-1] <= levels[0]  # ask not broken up

    def metrics(self, now: float | None = None) -> dict[str, float]:
        """Absorption inputs for this ticker (empty when tape is cold)."""
        now = time.time() if now is None else now
        if not self.prints or now - self.prints[-1].ts > ABSORPTION_CVD_SLOW:
            return {}
        return {
            "cvd_fast": self.cvd(ABSORPTION_CVD_FAST, now),
            "cvd_slow": self.cvd(ABSORPTION_CVD_SLOW, now),
            "vol_buy": self.side_volume(1, ABSORPTION_CVD_FAST, now),
            "vol_sell": self.side_volume(-1, ABSORPTION_CVD_FAST, now),
            "bid_held": 1.0 if self.level_held(-1, now) else 0.0,
            "ask_held": 1.0 if self.level_held(1, now) else 0.0,
        }


class TapeBook:
    """Per-ticker TapeBuffer registry (one book per bot)."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._bufs: dict[str, TapeBuffer] = {}

    def for_ticker(self, ticker: str) -> TapeBuffer:
        """Get or create the buffer for ``ticker``."""
        buf = self._bufs.get(ticker)
        if buf is None:
            buf = TapeBuffer(ticker)
            self._bufs[ticker] = buf
        return buf

    def drop(self, ticker: str) -> None:
        """Remove a ticker's buffer (unsubscribe / GC)."""
        self._bufs.pop(ticker, None)

    def clear(self) -> None:
        """Drop every buffer (session reset)."""
        self._bufs.clear()

    def metrics(self, ticker: str, now: float | None = None) -> dict[str, float]:
        """Tape metrics for ``ticker`` (empty dict when unknown/cold)."""
        buf = self._bufs.get(ticker)
        if buf is None:
            return {}
        return buf.metrics(now)


__all__ = ["Print", "TapeBuffer", "TapeBook"]
