"""brain.news_sources — live headline providers + sentiment feed (System 2).

BRAIN-FIRST: news is sensory EVIDENCE for the brain, never a gate. The
feed runs on the slow path (ConsolidationEngine thread), converts fresh
headlines into bounded polarity, and publishes a ±0.03 modifier to
BrainState. Every failure is silent — no provider is load-bearing.

Sources (free, rebuild senses/news/providers.py parity): Yahoo Finance
search API (no key) and Finnhub company-news (FINNHUB_API_KEY free tier).
Providers use stdlib urllib with hard timeouts; headlines cache per
ticker (TTL) and reuse the SAME keyword polarity model as brain/news.py
so learned outcomes and live headlines speak one language.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .news import SentimentPolarity

log = logging.getLogger(__name__)

_TTL: float = 300.0  # per-ticker headline cache (5 min)
_POLL_INTERVAL: float = 120.0  # background refresh cadence
_MAX_HEADLINES: int = 8  # per ticker per provider
_HTTP_TIMEOUT: float = 4.0
_YF_BASE = "https://query1.finance.yahoo.com/v1/finance/search?q="
_FINNHUB_BASE = "https://finnhub.io/api/v1/company-news"


@dataclass
class Headline:
    """One news item from any provider."""

    text: str
    source: str
    ts: float = 0.0

    @property
    def polarity(self) -> float:
        """Keyword polarity of the headline text (shared model)."""
        return _sentiment.polarity(self.text)


_headline_cache: dict[str, tuple[float, list[Headline]]] = {}
_cache_lock = threading.Lock()
# Shared polarity model (same keywords as the outcome learner).
_sentiment = SentimentPolarity()


def _http_get(url: str) -> str:
    """Bounded GET → body text (empty on any failure)."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "hanoon-prime/2.0 research"}
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            body: str = resp.read(65536).decode("utf-8", "ignore")
            return body
    except Exception:
        return ""


# ── Providers ────────────────────────────────────────────────────────


def _fetch_yahoo(ticker: str) -> list[Headline]:
    """Yahoo Finance search API — no key, generous limits."""
    body = _http_get(
        _YF_BASE + urllib.parse.quote(ticker) + f"&newsCount={_MAX_HEADLINES}"
    )
    if not body:
        return []
    try:
        items = json.loads(body).get("news", [])[:_MAX_HEADLINES]
        return [
            Headline(
                text=str(i.get("title", ""))[:300],
                source="yahoo",
                ts=float(i.get("providerPublishTime", 0) or 0),
            )
            for i in items
            if i.get("title")
        ]
    except Exception:
        return []


def _fetch_finnhub(ticker: str) -> list[Headline]:
    """Finnhub company-news — requires FINNHUB_API_KEY (free tier)."""
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        return []
    day = 24 * 3600
    now = time.time()
    url = (
        f"{_FINNHUB_BASE}?symbol={urllib.parse.quote(ticker)}"
        f"&from={time.strftime('%Y-%m-%d', time.gmtime(now - 2 * day))}"
        f"&to={time.strftime('%Y-%m-%d', time.gmtime(now))}"
        f"&token={key}"
    )
    body = _http_get(url)
    if not body:
        return []
    try:
        items = json.loads(body)[:_MAX_HEADLINES]
        return [
            Headline(
                text=str(i.get("headline", ""))[:300],
                source="finnhub",
                ts=float(i.get("datetime", 0) or 0),
            )
            for i in items
            if i.get("headline")
        ]
    except Exception:
        return []


_PROVIDERS = (_fetch_yahoo, _fetch_finnhub)


def headlines_for(ticker: str, max_age: float = _TTL) -> list[Headline]:
    """Fresh headlines for a ticker (cached across the TTL)."""
    now = time.time()
    with _cache_lock:
        cached = _headline_cache.get(ticker.upper())
        if cached and now - cached[0] < max_age:
            return cached[1]
    items: list[Headline] = []
    for provider in _PROVIDERS:
        try:
            items.extend(provider(ticker))
        except Exception:
            continue
        if len(items) >= _MAX_HEADLINES:
            break
    with _cache_lock:
        _headline_cache[ticker.upper()] = (now, items)
    return items


def ticker_sentiment(ticker: str) -> tuple[float, int]:
    """Mean headline polarity + headline count for a ticker in [-1, 1]."""
    items = headlines_for(ticker)
    if not items:
        return 0.0, 0
    pols = [h.polarity for h in items if h.text]
    if not pols:
        return 0.0, 0
    return sum(pols) / len(pols), len(pols)


class NewsFeedEngine:
    """System 2 news organ: refresh sentiment for tracked tickers.

    Publishes ``news_sentiment`` ({ticker: polarity}) into BrainState.
    Bounded ±0.03 downstream — evidence, never a gate.
    """

    def __init__(self, state: Any, interval: float = _POLL_INTERVAL) -> None:
        self._state = state
        self._interval = interval
        self._last_run = 0.0

    def maybe_refresh(self) -> None:
        """Called from the System 2 cycle — throttled to its interval."""
        now = time.time()
        if now - self._last_run < self._interval:
            return
        self._last_run = now
        try:
            alpha = self._state.get("latest_alpha", {}) or {}
            tickers = list(alpha.keys())[:5]
        except Exception:
            return
        snapshot = {}
        for t in tickers:
            try:
                pol, n = ticker_sentiment(t)
                if n:
                    snapshot[t] = round(pol, 3)
            except Exception:
                continue
        if snapshot:
            self._state.update(news_sentiment=snapshot)


__all__ = ["Headline", "NewsFeedEngine", "headlines_for", "ticker_sentiment"]
