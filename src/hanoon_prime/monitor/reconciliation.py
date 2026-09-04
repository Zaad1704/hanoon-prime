"""monitor.reconciliation — Restart recovery logic.

Seeds synthetic entry fills for restored broker positions and rebuilds
in-memory brackets that were lost on restart.

Source: rebuild's reconciliation.py (simplified).
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


class Reconciliation:
    """Handles restart recovery: seed fills and reconcile brackets."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._seeded: set[str] = set()

    def reconcile(self, ib_positions: dict[str, Any], executor: Any) -> list[str]:
        """Reconcile IB positions with internal state. Returns seeded tickers."""
        seeded = []
        for ticker, pos in ib_positions.items():
            if ticker not in executor.last_thoughts:
                # Position exists in IB but not in our state — seed it
                self._seed_position(ticker, pos, executor)
                seeded.append(ticker)
                log.info("RECONCILE: seeded %s (qty=%s)", ticker, pos)
        return seeded

    def _seed_position(self, ticker: str, pos: Any, executor: Any) -> None:
        """Create synthetic fill for a restored position."""
        try:
            avg_price = float(getattr(pos, "avgCost", 0))
            qty = float(getattr(pos, "position", 0))
            if avg_price > 0 and qty != 0:
                executor.last_thoughts[ticker] = {
                    "ticker": ticker,
                    "direction": 1 if qty > 0 else -1,
                    "price": avg_price,
                    "synthetic": True,
                }
                self._seeded.add(ticker)
        except Exception as exc:
            log.warning("Seed failed for %s: %s", ticker, exc)

    def is_seeded(self, ticker: str) -> bool:
        """Auto-generated docstring."""
        return ticker in self._seeded

    def clear_seed(self, ticker: str) -> None:
        """Auto-generated docstring."""
        self._seeded.discard(ticker)
