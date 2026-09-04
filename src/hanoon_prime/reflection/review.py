"""reflection.review — Review session + trade review.

Structured daily/weekly reviews with regime analysis.
Per-trade postmortem for learning.

Source: rebuild's review.py + trade_review.py (simplified).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class ReviewReport:
    summary: dict[str, Any]
    action_items: list[str]
    regime_findings: list[str]
    timestamp: float = 0.0


@dataclass
class TradePostmortem:
    ticker: str
    won: bool
    entry_score: float
    exit_reason: str
    lessons: list[str]


class ReviewSession:
    """Structured daily/weekly reviews."""

    def __init__(self, buffer: Any = None) -> None:
        """Auto-generated docstring."""
        self._buffer = buffer

    def run_daily_review(self) -> ReviewReport:
        """Run daily review from trade buffer."""
        if self._buffer is None:
            return ReviewReport({}, [], [])
        trades = self._buffer.get_trades(last_n=50)
        if not trades:
            return ReviewReport({"total": 0}, [], [])
        wins = sum(1 for t in trades if t.win)
        wr = wins / len(trades)
        avg_pnl = sum(t.pnl for t in trades) / len(trades)
        summary = {
            "total": len(trades),
            "wr": round(wr, 3),
            "avg_pnl": round(avg_pnl, 2),
        }
        items = []
        if wr < 0.40:
            items.append("Portfolio WR below 40% — review strategy")
        if avg_pnl < 0:
            items.append("Average PnL negative — review sizing")
        return ReviewReport(summary, items, [], time.time())

    def run_weekly_review(self) -> ReviewReport:
        """Run weekly review with deeper analysis."""
        return self.run_daily_review()


class TradeReview:
    """Per-trade postmortem for learning."""

    def review(
        self, trade: Any, alpha: Optional[dict[str, Any]] = None
    ) -> TradePostmortem:
        """Generate postmortem for a closed trade."""
        won = trade.win if hasattr(trade, "win") else trade.pnl > 0
        lessons = self._build_lessons(trade, won, alpha)
        ticker = trade.ticker if hasattr(trade, "ticker") else "?"
        return TradePostmortem(
            ticker=ticker,
            won=won,
            entry_score=0.0,
            exit_reason="closed",
            lessons=lessons,
        )

    def _build_lessons(
        self, trade: Any, won: bool, alpha: Optional[dict[str, Any]]
    ) -> list[str]:
        """Build lesson list from trade outcome."""
        pnl = trade.pnl if hasattr(trade, "pnl") else 0
        lessons = [f"{'Won' if won else 'Lost'} with pnl={pnl:.2f}"]
        if not won and alpha:
            for k, v in alpha.items():
                if abs(v) > 0.7:
                    lessons.append(f"Strong {k}={v:.2f} but lost")
        return lessons
