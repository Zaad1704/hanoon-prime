"""hanoon_prime._execution — position entry & exit execution layer.

Handles translating Brain verdicts into journal entries and
position state. Split from ib_adapter.py to keep files < 200 lines.
"""
from __future__ import annotations

import time
from .brain import Brain, Thought
from .journal import Journal
from .constants import FEE_RATE, FIXED_FEE

import numpy as np


class ExecutionManager:
    """Manages open positions and executes Brain verdicts."""

    def __init__(self, brain: Brain, journal: Journal) -> None:
        self.brain = brain
        self.journal = journal
        self._open_positions: dict[str, dict] = {}

    def execute_entry(self, ticker: str, thought: Thought, entry_price: float) -> bool:
        """Execute an ENTER verdict. Returns True if position opened."""
        shares = self.brain.size_position(thought.trace["win_prob"], entry_price)
        if shares <= 0:
            return False

        direction = thought.direction
        self._open_positions[ticker] = {
            "entry_price": entry_price,
            "shares": shares,
            "direction": direction,
            "peak_price": entry_price,
            "stop_price": entry_price * (0.97 if direction > 0 else 1.03),
            "target_price": entry_price * (1.12 if direction > 0 else 0.88),
            "entry_time": time.time(),
            "score": thought.score,
            "win_prob": thought.trace.get("win_prob", 0.0),
        }

        self.journal.append({
            "event": "entry", "ticker": ticker,
            "price": round(entry_price, 4), "shares": round(shares, 2),
            "direction": direction, "score": thought.score,
            "win_prob": thought.trace.get("win_prob", 0.0),
        })
        print(f"ENTRY: {ticker} @ ${entry_price:.2f} × {shares:.1f} "
              f"dir={direction} score={thought.score:.3f}")
        return True

    def check_exits(self, get_price_func) -> None:
        """Check all open positions for exit conditions."""
        for ticker in list(self._open_positions.keys()):
            pos = self._open_positions[ticker]
            try:
                result = get_price_func(ticker, lookback=2)
                close = result[0]
                high = result[1]
                low = result[2]
            except Exception:
                continue

            verdict = self.brain.deliberate_exit(
                ticker=ticker,
                entry_price=pos["entry_price"],
                high=[float(np.max(high))],
                low=[float(np.min(low))],
                close=[float(close[-1])],
            )

            if verdict == "EXIT":
                self._process_exit(ticker, pos, float(close[-1]), "brain_verdict")

    def _process_exit(self, ticker, pos, exit_price, reason):
        """Close a position and log the outcome."""
        direction = pos["direction"]
        if direction > 0:
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]

        notional = pos["entry_price"] * pos["shares"]
        fees = 2 * (FIXED_FEE + FEE_RATE * notional)
        pnl_pct -= fees / notional if notional > 0 else 0.0
        won = pnl_pct > 0

        self.brain.record_trade(
            ticker=ticker, won=won, pnl_pct=pnl_pct, alpha_snapshot={},
        )

        self.journal.append({
            "event": "exit", "ticker": ticker,
            "price": round(exit_price, 4), "pnl_pct": round(pnl_pct, 6),
            "won": won, "exit_reason": reason,
            "holding_minutes": round((time.time() - pos["entry_time"]) / 60, 1),
        })
        print(f"EXIT:  {ticker} @ ${exit_price:.2f}  pnl={pnl_pct:+.2%} {'✅' if won else '❌'}")
        del self._open_positions[ticker]

    @property
    def open_count(self) -> int:
        return len(self._open_positions)
