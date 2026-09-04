"""hanoon_prime.ib_executor — JULI's execution layer.
Places bracket orders, trails stops/targets, monitors IB orders.
IB is the HANDS — executes what the brain decides.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ._ib_sync import get_ib_pnl, journal_exit, journal_snapshot, read_ib_positions
from ._protect import protect_position, sweep_zombies
from ._telegram import trade_closed, trade_opened
from .edge import score_to_win_prob
from .hippocampus import Hippocampus
from .immune import ATR_STOP_MULT, ATR_TARGET_MULT
from .memory import Journal
from .types import Position

log = logging.getLogger(__name__)


def _find_stop_target(children: Any) -> tuple[float, float]:
    """Extract stop and target prices from bracket children."""
    sp = [c.auxPrice for c in children if c.auxPrice is not None]
    tp = [c.lmtPrice for c in children if c.lmtPrice is not None]
    return (float(max(sp)) if sp else 0.0, float(max(tp)) if tp else 0.0)


def _should_trail(d: int, cur: float, level: float, atr: float) -> bool:
    """Check if order should be trailed (moved 1 ATR in favor)."""
    return (cur - level > atr) if d > 0 else (level - cur > atr)


class IBExecutor:
    """JULI's execution layer — monitors IB and manages all orders."""

    def __init__(self, ib_client: Any, brain: Hippocampus, journal: Journal, tracked_tickers: set[str] | None = None) -> None:
        self.ib = ib_client
        self.brain = brain
        self.journal = journal
        self.tracked_tickers: set[str] = set(tracked_tickers or [])
        self._brackets: dict[str, tuple[float, float]] = {}
        self._pending_parent: set[str] = set()
        self._last_snapshot: float = 0.0
        self._closed_trades: list[dict[str, Any]] = []
        self.last_thoughts: dict[str, Any] = {}

    def place_bracket(self, ticker: str, thought: Any, price: float, streamer: Any) -> None:
        """Place atomic parent + TP + SL via IB's bracketOrder()."""
        atr = streamer.buffer_atr(ticker)
        if atr <= 0.0 or np.isnan(atr):
            log.warning("ATR invalid for %s", ticker)
            return
        shares = self.brain.size_position(score_to_win_prob(thought.score), price, atr)
        if shares <= 0:
            return
        d = thought.direction
        stop, target = round(price - d * ATR_STOP_MULT * atr, 2), round(price + d * ATR_TARGET_MULT * atr, 2)
        shares = max(1, int(shares))
        action = "BUY" if d > 0 else "SELL"
        contract = streamer.contracts[ticker]
        for order in self.ib.bracketOrder(action, shares, round(price, 2), target, stop):
            order.tif = "DAY"
            self.ib.placeOrder(contract, order)
        self._brackets[ticker] = (stop, target)
        self._pending_parent.add(ticker)
        log.info("BRACKET %s %s @ %.2f stop=%.2f target=%.2f qty=%d", action, ticker, round(price, 2), stop, target, shares)

    def sync_from_ib(self, streamer: Any) -> None:
        """Sync everything from IB — IB is source of truth."""
        if not self.ib.isConnected():
            return
        sweep_zombies(self.ib)
        protect_position(self.ib, self.tracked_tickers, self._brackets, self._pending_parent, streamer)
        ib_positions = read_ib_positions(self.ib, self.tracked_tickers, self._brackets)
        for t in set(self._brackets) - set(ib_positions):
            self._record_exit(t, ib_positions, streamer)
        self.brain._open_positions = ib_positions
        now = time.monotonic()
        if now - self._last_snapshot >= 10.0:
            self._last_snapshot = now
            journal_snapshot(self.journal, self.ib, ib_positions, self._brackets)

    def _record_exit(self, ticker: str, ib_positions: dict[str, Position], streamer: Any) -> None:
        """Record a closed position to journal and brain."""
        self._brackets.pop(ticker, None)
        pos = self.brain._open_positions.pop(ticker, None)
        if pos is None:
            return
        pnl = get_ib_pnl(self.ib, ticker, pos)
        log.info("EXIT %s (IB closed at P&L=%.4f)", ticker, pnl)
        trade_closed(ticker, "LONG" if pos.direction > 0 else "SHORT", pnl)
        self.brain.record_trade(ticker=ticker, won=pnl > 0, pnl_pct=pnl, direction=pos.direction)
        journal_exit(self.journal, ticker, pnl, pos)
        self._closed_trades.append({"ticker": ticker, "pnl": pnl, "return_pct": pnl, "direction": pos.direction, "entry_price": pos.entry_price, "shares": pos.shares})

    def monitor_orders(self, ib_positions: dict[str, Position], streamer: Any) -> None:
        """Monitor ALL parent orders — cancel orphans, trail both."""
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

    def _handle_parent(self, trade: Any, order: Any, sym: str, ib_positions: dict[str, Position], streamer: Any) -> None:
        """Handle one parent order — cancel orphans, trail both."""
        stop, target = _find_stop_target(getattr(order, "children", None) or [])
        if sym in self._brackets:
            stored = self._brackets[sym]
            stop = stored[0] if stored[0] > 0 else stop
            target = stored[1] if stored[1] > 0 else target
        if sym in self._pending_parent:
            if sym in ib_positions:
                self._pending_parent.discard(sym)
                pos = ib_positions[sym]
                action = "BUY" if pos.direction > 0 else "SELL"
                trade_opened(sym, action, int(abs(pos.shares)), pos.entry_price, stop, target)
            else:
                return
        elif sym not in ib_positions:
            self._cancel_if_active(trade, order)
            return
        if stop and target:
            self._brackets[sym] = (stop, target)
            cur = streamer.get_last_price(sym)
            d, atr = ib_positions[sym].direction, streamer.buffer_atr(sym)
            if cur and d and atr > 0:
                if _should_trail(d, cur, stop, atr):
                    self._modify_child(trade, "stop", cur - d * ATR_STOP_MULT * atr)
                if _should_trail(d, cur, target, atr):
                    self._modify_child(trade, "target", cur + d * ATR_TARGET_MULT * atr)

    def _cancel_if_active(self, trade: Any, order: Any) -> None:
        """Cancel orphan order if IB reports it active."""
        try:
            if not trade.isDone():
                self.ib.cancelOrder(order)
        except Exception as e:
            log.debug("cancel skip: %s", e)

    def _modify_child(self, trade: Any, kind: str, new_price: float) -> None:
        """Modify a child order (stop or target) in IB."""
        price = round(new_price, 2)
        for child in getattr(trade.order, "children", None) or []:
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

    def get_newly_closed_trades(self) -> list[dict[str, Any]]:
        """Return and clear newly closed trades."""
        trades = list(self._closed_trades)
        self._closed_trades.clear()
        return trades

    def close_position(self, ticker: str, streamer: Any) -> None:
        """Close position via market order (brain exit signal)."""
        pos = self.brain._open_positions.get(ticker)
        if pos is None:
            return
        contract = streamer.contracts.get(ticker)
        if contract is None:
            return
        action = "SELL" if pos.direction > 0 else "BUY"
        try:
            from ib_insync import MarketOrder
            self.ib.placeOrder(contract, MarketOrder(action, abs(pos.shares), tif="DAY"))
            log.info("BRAIN EXIT %s %s %d", action, ticker, abs(pos.shares))
        except Exception as e:
            log.warning("close_position failed %s: %s", ticker, e)

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