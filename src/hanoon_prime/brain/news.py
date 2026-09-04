"""brain.news — News learner + sentiment polarity.

Tracks news sentiment patterns and their trade outcomes.
Learns which news signals are predictive over time.

Source: rebuild's news_learner.py + sentiment.py (simplified).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_STATE_PATH = Path("runtime/news_learner.json")
_SENTIMENT_TTL: float = 300.0  # 5 min cache
_MOD_BOUND: float = 0.03

# Keyword polarity dictionary
_BULLISH = {
    "surge",
    "rally",
    "upgrade",
    "beat",
    "strong",
    "growth",
    "breakout",
    "bullish",
    "soar",
    "jump",
    "record",
    "profit",
}
_BEARISH = {
    "crash",
    "downgrade",
    "miss",
    "weak",
    "loss",
    "decline",
    "sell",
    "bearish",
    "plunge",
    "drop",
    "fear",
    "risk",
}


class NewsLearner:
    """Learn which news patterns predict wins."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._patterns: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load()

    def record_trade(
        self, ticker: str, sentiment: float, horizon: str, won: bool, pnl: float
    ) -> None:
        """Record trade outcome with sentiment context."""
        bucket = self._sentiment_bucket(sentiment)
        key = f"{ticker}_{bucket}_{horizon}"
        with self._lock:
            if key not in self._patterns:
                self._patterns[key] = {"n": 0, "wins": 0}
            self._patterns[key]["n"] += 1
            if won:
                self._patterns[key]["wins"] += 1
            self._save()

    def get_advice(self, sentiment: float, horizon: str = "scalp") -> dict[str, Any]:
        """Get sentiment-based advice."""
        bucket = self._sentiment_bucket(sentiment)
        total = 0
        wins = 0
        for key, data in self._patterns.items():
            if f"_{bucket}_{horizon}" in key:
                total += data["n"]
                wins += data["wins"]
        if total < 5:
            return {"modifier": 0.0, "confidence": 0.0, "n": 0}
        wr = wins / total
        mod = (wr - 0.5) * 0.1
        mod = max(-_MOD_BOUND, min(_MOD_BOUND, mod))
        return {"modifier": mod, "confidence": min(1.0, total / 50), "n": total}

    def _sentiment_bucket(self, sentiment: float) -> str:
        """Auto-generated docstring."""
        if sentiment > 0.3:
            return "strong_positive"
        if sentiment > 0.1:
            return "positive"
        if sentiment < -0.3:
            return "strong_negative"
        if sentiment < -0.1:
            return "negative"
        return "neutral"

    def _save(self) -> None:
        """Auto-generated docstring."""
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._patterns))
            tmp.replace(_STATE_PATH)
        except Exception as exc:
            log.debug("News save failed: %s", exc)

    def _load(self) -> None:
        """Auto-generated docstring."""
        if not _STATE_PATH.exists():
            return
        try:
            self._patterns = json.loads(_STATE_PATH.read_text())
        except Exception as exc:
            log.debug("News load failed: %s", exc)


class SentimentPolarity:
    """Convert text headlines to bounded numeric polarity."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._cache: dict[str, tuple[float, float]] = {}

    def polarity(self, text: str) -> float:
        """Compute sentiment polarity from text. Returns [-1, 1]."""
        if not text:
            return 0.0
        now = time.time()
        cached = self._cache.get(text[:100])
        if cached and now - cached[1] < _SENTIMENT_TTL:
            return cached[0]
        words = set(re.findall(r"\w+", text.lower()))
        bull = len(words & _BULLISH)
        bear = len(words & _BEARISH)
        total = bull + bear
        if total == 0:
            return 0.0
        score = (bull - bear) / total
        self._cache[text[:100]] = (score, now)
        return score


_learner: Optional[NewsLearner] = None
_polarity: Optional[SentimentPolarity] = None


def get_news_learner() -> NewsLearner:
    """Auto-generated docstring."""
    global _learner
    if _learner is None:
        _learner = NewsLearner()
    return _learner


def get_sentiment_polarity() -> SentimentPolarity:
    """Auto-generated docstring."""
    global _polarity
    if _polarity is None:
        _polarity = SentimentPolarity()
    return _polarity
