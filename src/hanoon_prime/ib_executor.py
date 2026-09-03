"""hanoon_prime.ib_executor — IB-native order placement and exit monitoring.

IB is the SINGLE source of truth for everything:
  - Positions: read from ib.positions()
  - P&L: read from reqPnL()
  - Orders: managed via bracketOrder()
  - Exits: handled by IB's native bracket (stop/target children)

This module does NOT maintain local state. Every cycle:
  1. Query IB for actual positions
  2. Sync local view to match IB exactly
  3. Journal is a pure carbon copy of IB state

The system behaves like a human trader:
  - Views IB TWS for all data
  - Places orders through IB
  - Lets IB handle exits via bracket orders
  - Records IB state to journal (carbon copy)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._ib_sync import get_ib_pnl, journal_exit, journal_snapshot, read_ib_positions
from .edge import score_to_win_prob
from .hippocampus import Hippocampus
from .ib_compat import ib
from .immune import ATR_STOP_MULT, ATR_TARGET_MULT
from .memory import Journal
from .types import Position

log = logging.getLogger(__name__)


class IBExecutor:
    """Place bracket orders and monitor exits via live IB data."""

    def __init__(
        self, ib_client: Any, brain: Hippocampus,
        journal: Journal, tracked_tickers: set[str] | None = None,
    ) -> None:
        self.ib = ib_client
        self.brain = brain
        self.journal = journal
        self.tracked_tickers: set[str] = set(tracked_tickers or [])
        self.last_thoughts: dict[str, Any] = {}
        self._brackets: dict[str, tuple[float, float]] = {}

    def place_bracket(
        self, ticker: str, thought: Any, price: float, streamer: Any
    ) -> None:
        """Place atomic parent + TP + SL via IB's bracketOrder()."""
        atr = streamer.buffer_atr(ticker)
        if atr <= 0.0 or np.isnan(atr):
            log.warning("ATR invalid for %s — skipping", ticker)
            return
        shares = self.brain.size_position(
            score_to_win_prob(thought.score), price, atr
        )
        if shares <= 0:
            log.warning("Sizing returned 0 for %s — skipping", ticker)
            return
        d = thought.direction
        stop = round(price - d * ATR_STOP_MULT * atr, 2)
        target = round(price + d * ATR_TARGET_MULT * atr, 2)
        shares = max(1, int(shares))
        action = "BUY" if d > 0 else "SELL"
        contract = streamer.contracts[ticker]
        bracket = self.ib.bracketOrder(
            action, shares, round(price, 2), target, stop
        )
        for order in bracket:
            order.tif = "DAY"
            self.ib.placeOrder(contract, order)
        self._brackets[ticker] = (stop, target)
        log.info(
            "BRACKET %s %s @ %.2f stop=%.2f target=%.2f qty=%d",
            action, ticker, round(price, 2), stop, target, shares,
        )

    def sync_from_ib(self, streamer: Any) -> None:
        """Sync everything from IB — the single source of truth."""
        if not self._ping_ib():
            return
        ib_positions = read_ib_positions(
            self.ib, self.tracked_tickers, self._brackets
        )
        self._sync_and_cancel_orders(ib_positions, streamer)
        self._record_closed_positions(ib_positions, streamer)
        self.brain._open_positions = ib_positions
        journal_snapshot(self.journal, self.ib, ib_positions, self._brackets)

    def _ping_ib(self) -> bool:
        """Verify IB connection is alive before syncing."""
        try:
            return bool(self.ib.isConnected())
        except Exception:
            return False

    def _sync_and_cancel_orders(
        self, ib_positions: dict[str, Position], streamer: Any
    ) -> None:
        """Sync brackets from IB; cancel orphans; trail winning stops."""
        try:
            for trade in self.ib.trades():
                o = trade.order
                if not o or o.parentId:
                    continue
                sym = trade.contract.symbol if trade.contract else ""
                if sym not in self.tracked_tickers:
                    continue
                self._sync_one_bracket(trade, o, sym, ib_positions, streamer)
        except Exception as e:
            log.warning("sync orders from IB failed: %s", e)

    def _sync_one_bracket(
        self, trade: Any, order: Any, sym: str,
        ib_positions: dict[str, Position], streamer: Any,
    ) -> None:
        """Sync one bracket — cancel orphans, trail stops."""
        children = getattr(order, "children", None) or []
        stop = max((c.auxPrice for c in children if c.auxPrice), default=0.0)
        target = max((c.lmtPrice for c in children if c.lmtPrice), default=0.0)
        if sym not in ib_positions:
            self._cancel_if_active(trade, order)
            return
        if stop and target:
            self._brackets[sym] = (float(stop), float(target))
            self._trail_stop(sym, ib_positions[sym], streamer, float(stop), trade)

    def _cancel_if_active(self, trade: Any, order: Any) -> None:
        """Cancel an orphan order only if IB reports it active."""
        try:
            if not trade.isDone():
                self.ib.cancelOrder(order)
        except Exception as e:
            log.debug("cancel skip: %s", e)

    def _trail_stop(
        self, sym: str, pos: Position, streamer: Any,
        stop: float, trade: Any,
    ) -> None:
        """Trail stop loss for in-favor positions."""
        cur = streamer.get_last_price(sym)
        d = pos.direction
        if not cur or not d:
            return
        atr = streamer.buffer_atr(sym)
        if (d > 0 and cur - stop > atr) or (d < 0 and stop - cur > atr):
            new_stop = round(cur - d * ATR_STOP_MULT * atr, 2)
            for child in (trade.order.children or []):
                if getattr(child, "auxPrice", 0):
                    child.auxPrice = new_stop
                    self.ib.placeOrder(trade.contract, trade.order)
                    return

    def _record_closed_positions(
        self, ib_positions: dict[str, Position], streamer: Any
    ) -> None:
        """Record exits for positions IB closed."""
        for t in set(self._brackets) - set(ib_positions):
            self._record_exit(t, streamer)

    def _record_exit(self, ticker: str, streamer: Any) -> None:
        """Record exit when IB confirms position closed."""
        self._brackets.pop(ticker, None)
        pos = self.brain._open_positions.pop(ticker, None)
        if pos is None:
            return
        pnl = get_ib_pnl(self.ib, ticker, pos)
        log.info("EXIT %s (IB closed at P&L=%.4f)", ticker, pnl)
        self.brain.record_trade(
            ticker=ticker, won=pnl > 0, pnl_pct=pnl,
            direction=pos.direction,
        )
        journal_exit(self.journal, ticker, pnl, pos)

    def check_exits(self, streamer: Any) -> None:
        """IB handles exits via bracket orders — no local check needed."""
        pass

    def cancel_all(self) -> None:
        """Cancel all open orders in IB."""
        try:
            cancel = getattr(self.ib, "cancelAllOrders", None)
            (cancel or self.ib.reqGlobalCancel)()
        except Exception as e:
            log.warning("cancel all orders failed: %s", e)
        self._brackets.clear()


__all__ = ["IBExecutor"]
