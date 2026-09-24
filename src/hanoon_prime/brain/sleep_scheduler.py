"""hanoon_prime.brain.sleep_scheduler — auto-trigger + weighting for sleep replay.

Sleep does offline consolidation during downtime; this scheduler decides
WHEN (30min of inactivity or session close) and HOW the replay list is
biased — losers get 3× the replay drive of winners so the network learns
from its mistakes rather than re-soothing itself with wins, and random
historical traces interleave in (CLS interleaved learning).
"""

from __future__ import annotations

import random
import time
from typing import Any

from .learning_config import (
    SLEEP_COOLDOWN_SEC,
    SLEEP_INTERLEAVE_MAX,
    SLEEP_LOSS_WEIGHT,
    SLEEP_THRESHOLD_SEC,
    SLEEP_WIN_WEIGHT,
)


class SleepScheduler:
    """Decides when sleep replay runs and how the replay list is weighted."""

    def __init__(
        self,
        threshold_sec: float = SLEEP_THRESHOLD_SEC,
        cooldown_sec: float = SLEEP_COOLDOWN_SEC,
    ) -> None:
        """Start with a clean trigger history."""
        self._threshold = threshold_sec
        self._cooldown = cooldown_sec
        self._last_trigger: float = float("-inf")

    def check(
        self,
        last_trade_ts: float,
        session_close: bool = False,
        now: float | None = None,
    ) -> bool:
        """True every cooldown window once the machine has been idle."""
        now = now if now is not None else time.time()
        idle = (now - float(last_trade_ts)) >= self._threshold
        if not (idle or session_close):
            return False
        if (now - self._last_trigger) < self._cooldown:
            return False
        self._last_trigger = now
        return True

    def replay_weights(
        self,
        recent: list[tuple[dict[str, float], bool]],
        historical: list[tuple[dict[str, float], bool]] | None = None,
    ) -> list[tuple[dict[str, float], float, bool]]:
        """Weighted replay list: losers 3× winners, interleaved with history.

        ``recent`` and ``historical`` are ``(pattern, won)`` pairs; each
        pattern comes out tagged with its replay drive AND its explicit
        win/loss polarity, so the sleep engine never infers reward sign
        from drive magnitude (FIX-2026-09-23-08).
        """
        picks = [
            (pattern, SLEEP_LOSS_WEIGHT if not won else SLEEP_WIN_WEIGHT, won)
            for pattern, won in recent
        ]
        if historical is not None:
            pool = list(historical)
            random.shuffle(pool)
            for pattern, won in pool[:SLEEP_INTERLEAVE_MAX]:
                picks.append(
                    (pattern, SLEEP_LOSS_WEIGHT if not won else SLEEP_WIN_WEIGHT, won)
                )
        return picks

    @property
    def last_trigger(self) -> float:
        """Timestamp of the most recent triggered replay."""
        return self._last_trigger
