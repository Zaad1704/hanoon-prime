"""hanoon_prime.ib_executor — JULI's execution layer.

JULI is the BRAIN — it does everything:
  - Places bracket orders (parent + stop + target)
  - Trails BOTH stop and target as position moves in favor
  - Monitors all orders and their children
  - Cancels orphaned orders
  - Records exits to journal (carbon copy of IB)

IB is just the HANDS — it only executes what JULI tells it.
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

def _find_stop_target(children: Any) -> tuple[float, float]:
    """Extract stop and target prices from bracket children."""
    stop = max((c.auxPrice for c in children if c.auxPrice), default=0.0)
    target = max((c.lmtPrice for c in children if c.lmtPrice), default=0.0)
    return float(stop), float(target)

def _should_trail(d: int, cur: float, level: float, atr: float) -> bool:
    """Check if order should be trailed (moved 1 ATR in favor)."""
    if d > 0:
        return cur - level > atr
    return level - cur > atr

class IBExecutor:
    """JULI's execution layer — monitors IB and manages all orders."""

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
        self._pending_parent: set[str] = set()

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
            return
        d = thought.direction
        stop = round(price - d * ATR_STOP_MULT * atr, 2)
        target = round(price + d * ATR_TARGET_MULT * atr, 2)
        shares = max(1, int(shares))
        action = "BUY" if d > 0 else "SELL"
        contract = streamer.contracts[ticker]
        bracket = self.ib.bracketOrder(action, shares, round(price, 2), target, stop)
        for order in bracket:
            order.tif = "DAY"
            self.ib.placeOrder(contract, order)
        self._brackets[ticker] = (stop, target)
        self._pending_parent.add(ticker)
        log.info("BRACKET %s %s @ %.2f stop=%.2f target=%.2f qty=%d",
                 action, ticker, round(price, 2), stop, target, shares)

    def sync_from_ib(self, streamer: Any) -> None:
        """JULI syncs everything from IB — IB is source of truth."""
        if not self._ping_ib():
            return
        ib_positions = read_ib_positions(self.ib, self.tracked_tickers, self._brackets)
        self._monitor_all_orders(ib_positions, streamer)
        self._record_closed_positions(ib_positions, streamer)
        self.brain._open_positions = ib_positions
        journal_snapshot(self.journal, self.ib, ib_positions, self._brackets)

    def _ping_ib(self) -> bool:
        """Verify IB connection is alive."""
        try:
            return bool(self.ib.isConnected())
        except Exception:
            return False

    def _monitor_all_orders(
        self, ib_positions: dict[str, Position], streamer: Any
    ) -> None:
        """JULI monitors ALL parent orders — cancel orphans, trail both."""
        try:
            for trade in self.ib.trades():
                o = trade.order
                if not o or o.parentId:
                    continue
                sym = trade.contract.symbol if trade.contract else ""
                if sym not in self.tracked_tickers:
                    continue
                self._handle_parent(trade, o, sym, ib_positions, streamer)
        except Exception as e:
            log.warning("monitor orders failed: %s", e)

    def _handle_parent(
        self, trade: Any, order: Any, sym: str,
        ib_positions: dict[str, Position], streamer: Any,
    ) -> None:
        """Handle one parent order — cancel orphans, trail both."""
        stop, target = _find_stop_target(getattr(order, "children", None) or [])
        if sym in self._pending_parent:
            if sym in ib_positions:
                self._pending_parent.discard(sym)
            else:
                return  # parent placed, not yet filled — don't cancel
        elif sym not in ib_positions:
            self._cancel_if_active(trade, order)
            return
        if stop and target:
            self._brackets[sym] = (stop, target)
            self._trail_both(sym, ib_positions[sym], streamer, stop, target, trade)

    def _cancel_if_active(self, trade: Any, order: Any) -> None:
        """Cancel an orphan order only if IB reports it active."""
        try:
            if not trade.isDone():
                self.ib.cancelOrder(order)
        except Exception as e:
            log.debug("cancel skip: %s", e)

    def _trail_both(
        self, sym: str, pos: Position, streamer: Any,
        stop: float, target: float, trade: Any,
    ) -> None:
        """JULI trails BOTH stop and target as position moves in favor."""
        cur = streamer.get_last_price(sym)
        d, atr = pos.direction, streamer.buffer_atr(sym)
        if not cur or not d or atr <= 0.0:
            return
        if _should_trail(d, cur, stop, atr):
            self._modify_child(trade, "stop", cur - d * ATR_STOP_MULT * atr)
        if _should_trail(d, cur, target, atr):
            self._modify_child(trade, "target", cur + d * ATR_TARGET_MULT * atr)

    def _modify_child(self, trade: Any, kind: str, new_price: float) -> None:
        """Modify a child order (stop or target) in IB."""
        price = round(new_price, 2)
        for child in (trade.order.children or []):
            if kind == "stop" and getattr(child, "auxPrice", None):
                child.auxPrice = price
                self.ib.placeOrder(trade.contract, trade.order)
                log.info("TRAIL STOP %s -> %.2f", trade.contract.symbol, price)
                return
            if kind == "target" and getattr(child, "lmtPrice", None):
                child.lmtPrice = price
                self.ib.placeOrder(trade.contract, trade.order)
                log.info("TRAIL TARGET %s -> %.2f", trade.contract.symbol, price)
                return

    def _record_closed_positions(self, ib_positions: dict[str, Position], streamer: Any) -> None:
        for t in set(self._brackets) - set(ib_positions):
            self._record_exit(t, streamer)

    def _record_exit(self, ticker: str, streamer: Any) -> None:
        self._brackets.pop(ticker, None)
        pos = self.brain._open_positions.pop(ticker, None)
        if pos is None:
            return
        pnl = get_ib_pnl(self.ib, ticker, pos)
        log.info("EXIT %s (IB closed at P&L=%.4f)", ticker, pnl)
        self.brain.record_trade(
            ticker=ticker, won=pnl > 0, pnl_pct=pnl, direction=pos.direction
        )
        journal_exit(self.journal, ticker, pnl, pos)

    def cancel_all(self) -> None:
        """Cancel all open orders in IB."""
        try:
            cancel = getattr(self.ib, "cancelAllOrders", None)
            (cancel or self.ib.reqGlobalCancel)()
        except Exception as e:
            log.warning("cancel all failed: %s", e)
        self._brackets.clear()
        self._pending_parent.clear()

__all__ = ["IBExecutor"]
