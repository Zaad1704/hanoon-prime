"""hanoon_prime.brain.thalamus — Sensory gateway and data quality filter.

Filters raw scanner results before they reach the brain.
Checks: price, volume, spread, data freshness.
Computes Thalamic Salience Index for data budget prioritization.

Biological analogy: Sensory relay station — only passes
relevant stimuli to higher brain regions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

MIN_PRICE: float = 5.0
MIN_DAILY_VOLUME: int = 1_000_000
MAX_SPREAD_PCT: float = 0.005  # 0.5%
MIN_DATA_AGE: float = 5.0  # seconds


@dataclass
class ThalamusVerdict:
    """Result of thalamus screening."""

    ticker: str
    eligible: bool
    salience: float = 0.0
    reason: str = ""


class Thalamus:
    """Sensory gateway — filters and ranks candidates."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}

    def screen(
        self,
        ticker: str,
        bid: float,
        ask: float,
        last: float,
        volume: float,
        daily_volume: float,
    ) -> ThalamusVerdict:
        """Screen a candidate through all filters."""
        if last < MIN_PRICE:
            return ThalamusVerdict(ticker, False, reason="price_too_low")
        if daily_volume < MIN_DAILY_VOLUME:
            return ThalamusVerdict(ticker, False, reason="low_volume")
        if bid <= 0 or ask <= 0:
            return ThalamusVerdict(ticker, False, reason="no_bid_ask")
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return ThalamusVerdict(ticker, False, reason="invalid_mid")
        spread_pct = (ask - bid) / mid
        if spread_pct > MAX_SPREAD_PCT:
            return ThalamusVerdict(
                ticker,
                False,
                reason=f"spread_{spread_pct:.4f}",
            )
        salience = self._compute_salience(
            bid=bid,
            ask=ask,
            last=last,
            volume=volume,
            daily_volume=daily_volume,
        )
        self._seen[ticker] = time.time()
        return ThalamusVerdict(ticker, True, salience=salience)

    def _compute_salience(
        self,
        bid: float,
        ask: float,
        last: float,
        volume: float,
        daily_volume: float,
    ) -> float:
        """Thalamic Salience Index — higher = more interesting."""
        vol_norm = min(volume / max(daily_volume, 1.0), 1.0) * 0.4
        mid = (bid + ask) / 2.0
        spread_inv = (1.0 / max((ask - bid) / mid, 0.0001)) * 0.3
        momentum = abs(last - bid) / max(mid, 0.01) * 0.3
        return vol_norm + spread_inv + momentum

    def rank(
        self,
        screeners: list[ThalamusVerdict],
    ) -> list[ThalamusVerdict]:
        """Rank eligible candidates by salience."""
        eligible = [s for s in screeners if s.eligible]
        eligible.sort(key=lambda s: s.salience, reverse=True)
        return eligible

    def is_stale(self, ticker: str, max_age: float = MIN_DATA_AGE) -> bool:
        """Check if we haven't seen this ticker recently."""
        last_seen = self._seen.get(ticker, 0.0)
        return time.time() - last_seen > max_age
