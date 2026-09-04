"""hanoon_prime.monitor.exit_scoring — Re-score positions for exit decisions.

Re-scores each open position through JULI's live indicators to determine
health. If health drops below threshold, an exit signal is generated.
This runs on the monitor daemon thread, not the main loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..brain.indicators import compute_all_alpha


@dataclass
class ExitHealth:
    """Structured exit health result."""

    score: float = 0.5
    verdict: int = 0  # 0=STAY, 1=EXIT
    reason: str = ""


class ExitScorer:
    """Re-score positions through live brain for exit decisions."""

    def __init__(self, threshold: float = 0.3) -> None:
        self._threshold = threshold

    def score_exit(
        self,
        ticker: str,
        entry_price: float,
        current_price: float,
        direction: int,
        close: Any = None,
        high: Any = None,
        low: Any = None,
        volume: Any = None,
    ) -> ExitHealth:
        """Re-score a position and decide if it should be exited."""
        STAY, EXIT = 0, 1
        if entry_price <= 0 or current_price <= 0:
            return ExitHealth(score=0.5, verdict=STAY, reason="no_price")
        score = 0.5
        if close is not None and len(close) >= 10:
            alpha = compute_all_alpha(close, high, low, volume)
            vals = [abs(v) for v in alpha.values() if isinstance(v, (int, float))]
            score = sum(vals) / max(len(vals), 1)
        pnl_pct = (current_price - entry_price) / entry_price * direction
        if pnl_pct < -0.03:
            return ExitHealth(score=score, verdict=EXIT, reason="stop_loss")
        if pnl_pct > 0.05 and score < self._threshold:
            return ExitHealth(score=score, verdict=EXIT, reason="profit_lock")
        if score < self._threshold * 0.5:
            return ExitHealth(score=score, verdict=EXIT, reason="health_collapse")
        return ExitHealth(score=score, verdict=STAY, reason="ok")
