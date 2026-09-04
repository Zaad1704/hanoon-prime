"""brain.ib_intel — IB intelligence hub.

Queries IB for news, filings, corporate actions. Caches results.
Provides sentiment data to the brain.

Source: rebuild's hanoon/ib_intel.py (simplified).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

_NEWS_TTL: float = 3600.0  # 1 hour


@dataclass
class IntelItem:
    symbol: str
    headline: str = ""
    source: str = ""
    timestamp: float = 0.0
    sentiment: float = 0.0


@dataclass
class IntelCache:
    news: dict[str, IntelItem] = field(default_factory=dict)
    filings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


class IBIntel:
    """IB intelligence hub — news, filings, corporate actions."""

    def __init__(self, ib_client: Any = None) -> None:
        """Auto-generated docstring."""
        self._ib = ib_client
        self._cache = IntelCache()
        self._last_fetch: dict[str, float] = {}

    def get_news(self, symbol: str) -> Optional[IntelItem]:
        """Get cached news for a symbol."""
        item = self._cache.news.get(symbol)
        if item and time.time() - item.timestamp < _NEWS_TTL:
            return item
        return None

    def get_sentiment(self, symbol: str) -> float:
        """Get sentiment polarity for a symbol."""
        item = self.get_news(symbol)
        if item:
            return item.sentiment
        return 0.0

    def fetch_news(self, symbol: str) -> Optional[IntelItem]:
        """Fetch news from IB (if connected)."""
        now = time.time()
        if now - self._last_fetch.get(symbol, 0) < _NEWS_TTL:
            return self.get_news(symbol)
        self._last_fetch[symbol] = now
        if self._ib is None:
            return None
        try:
            # IB news query (simplified)
            return None
        except Exception as exc:
            log.debug("News fetch failed for %s: %s", symbol, exc)
            return None

    def update_cache(self, symbol: str, headline: str, sentiment: float = 0.0) -> None:
        """Update cache with new intel."""
        self._cache.news[symbol] = IntelItem(
            symbol=symbol, headline=headline, sentiment=sentiment, timestamp=time.time()
        )

    def snapshot(self) -> dict[str, Any]:
        """Auto-generated docstring."""
        return {"n_news": len(self._cache.news), "n_filings": len(self._cache.filings)}
