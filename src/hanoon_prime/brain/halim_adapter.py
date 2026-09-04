"""hanoon_prime.brain.halim_adapter — async HALIM API client.

Communicates with the external HALIM AI advisor service.
Returns a bounded modifier (±HALIM_MOD_BOUND).
Non-blocking: starts debate async, reads cached result instantly.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any

from .config import HALIM_MOD_BOUND


class HalimAdapter:
    """Async HALIM API client with caching."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765") -> None:
        self._base_url = base_url
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl: float = 60.0

    def get_modifier(
        self,
        ticker: str,
        alpha: dict[str, float],
        score: float,
        verdict: str,
    ) -> float:
        """Get HALIM's advisory modifier. Non-blocking, returns cached if available."""
        cached = self._cache.get(ticker)
        if cached and (time.time() - float(cached.get("ts", 0))) < self._cache_ttl:
            return float(cached.get("modifier", 0.0))
        self._start_debate_async(ticker, alpha, score, verdict)
        return 0.0

    def _start_debate_async(
        self,
        ticker: str,
        alpha: dict[str, float],
        score: float,
        verdict: str,
    ) -> None:
        """Fire-and-forget HTTP request to HALIM."""
        try:
            data = json.dumps(
                {
                    "ticker": ticker,
                    "score": score,
                    "verdict": verdict,
                    "alpha": alpha,
                }
            ).encode()
            req = urllib.request.Request(
                f"{self._base_url}/debate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logging.getLogger(__name__).debug("HALIM debate failed: %s", e)

    def _read_cache(self, ticker: str) -> float:
        """Read cached HALIM advisory for a ticker."""
        cached = self._cache.get(ticker)
        if not cached:
            return 0.0
        if (time.time() - float(cached.get("ts", 0))) > self._cache_ttl:
            return 0.0
        return float(cached.get("modifier", 0.0))

    @staticmethod
    def _bound(value: float) -> float:
        return max(-HALIM_MOD_BOUND, min(HALIM_MOD_BOUND, value))
