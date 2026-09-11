"""hanoon_prime.ib_executor — JULI's execution layer (Brackets, trails, monitors IB orders.)"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Optional

import numpy as np

from ._guard import cancel_sell_legs, open_order_for, short_positions
from ._ib_sync import get_ib_pnl, journal_exit, journal_snapshot, read_ib_positions
from ._protect import protect_position
from ._telegram import trade_closed, trade_opened
from .brain.horizons import HORIZONS
from .brain.policy.trading_policy import TRADING_CONFIG
from .brain.risk import SizingResult
from .edge import score_to_win_prob
from .hippocampus import Hippocampus
from .ib_bracket import _brackets_from_trades
from .ib_compat import ib as _ib
from .ib_order_sweep import sweep_zombies
from .immune import ALLOW_EXTENDED_HOURS, ATR_STOP_MULT, ATR_TARGET_MULT
from .memory import Journal
from .types import ExitLevels

log = logging.getLogger(__name__)


class IBExecutor:
    """JULI's execution layer — monitors IB and manages all orders."""

    def __init__(
        self,
        ib_client: Any,
        brain: Hippocampus,
        journal: Journal,
        tracked_tickers: set[str] | None = None,
    ) -> None:
        self.ib = ib_client
        self.brain = brain
        self.journal = journal
        self.tracked_tickers: set[str] = set(tracked_tickers or [])
        self._brackets: dict[str, tuple[float, float]] = {}
        self._pending_parent: set[str] = set()
        self._synthetic: set[str] = set()  # reconciled positions (not real IB fills)
        self._horizons: dict[str, str] = {}  # ticker → trading horizon
        self._last_snapshot: float = 0.0
        self._closed_trades: list[dict[str, Any]] = []
        self.last_thoughts: dict[str, Any] = {}
        self.on_fill_confirmed: Callable[[str, float], None] | None = None
        # IB account context mirrored from ib_cycle each cycle (IB source).
        self._account_feed: dict[str, Any] = {}
        self._winrate_provider: Callable[[], tuple[float, int]] | None = None
        # Netting guard state: ticker → qty last requested to flatten.
        self._flattening: dict[str, int] = {}

    def place_bracket(
        self,
        ticker: str,
        thought: Any,
        price: float,
        streamer: Any,
        sizing: Optional[SizingResult] = None,
        horizon: str = "scalp",
    ) -> None:
        """Place atomic parent + TP + SL via IB's bracketOrder().

        When a ``SizingResult`` from the brain is supplied with
        ``risk_pass=True`` its penny-cap'd stop/target/shares are used
        directly (live sizing respects the realized-EV gate). Otherwise the
        legacy hippocampus sizing is used as a fallback.
        """
        atr = streamer.buffer_atr(ticker)
        if atr <= 0.0 or np.isnan(atr) or np.isnan(price):
            log.warning("ATR/price invalid for %s (atr=%.4f price=%.4f)", ticker, atr, price)
            return
        d = thought.direction
        if sizing is not None and getattr(sizing, "risk_pass", False):
            shares = int(sizing.shares)
            stop = float(sizing.stop_price)
            target = float(sizing.target_price)
            # Defense-in-depth: the risk engine guards inputs, but a NaN
            # stop/target would send "Limit Price=nan" to IB (Error 320).
            # Reject rather than paper-trade bad risk parameters.
            if not math.isfinite(stop) or not math.isfinite(target):
                log.warning(
                    "ABORT %s: NaN stop/target from sizing (stop=%.4f target=%.4f)",
                    ticker, stop, target,
                )
                return
        else:
            raw = self.brain.size_position(score_to_win_prob(thought.score), price, atr)
            shares = max(1, int(raw))
            stop = max(0.01, round(price - d * ATR_STOP_MULT * atr, 2))
            target = max(0.01, round(price + d * ATR_TARGET_MULT * atr, 2))
        if shares <= 0:
            return
        action = "BUY" if d > 0 else "SELL"
        if action == "SELL" and not TRADING_CONFIG.is_direction_allowed(action):
            log.info("SHORT order blocked (long_only policy)")
            return
        cancel_sell_legs(self.ib, ticker)
        contract = streamer.contracts[ticker]
        self._horizons[ticker] = horizon
        for order in self.ib.bracketOrder(
            action, shares, round(price, 2), target, stop
        ):
            order.tif = "DAY"
            order.outsideRth = ALLOW_EXTENDED_HOURS
            self.ib.placeOrder(contract, order)
        self._brackets[ticker] = (stop, target)
        self._pending_parent.add(ticker)
        log.info(
            "BRACKET %s %s @ %.2f stop=%.2f target=%.2f qty=%d",
            action,
            ticker,
            round(price, 2),
            stop,
            target,
            shares,
        )

    def sync_from_ib(self, streamer: Any, closing: set[str] | None = None) -> None:
        """Sync everything from IB — IB is source of truth."""
        if not self.ib.isConnected():
            return
        sweep_zombies(self.ib, self._pending_parent)
        # Adopt orphan positions (not placed by this bot session)
        self._adopt_orphan_positions(streamer, closing or set())
        protect_position(
            self.ib,
            self.tracked_tickers,
            self._brackets,
            self._pending_parent,
            streamer,
        )
        _brackets_from_trades(self.ib, self.tracked_tickers, self._brackets)
        ib_positions = read_ib_positions(self.ib, self.tracked_tickers, self._brackets)
        # Guard: if IB returns empty but brain knows about positions, the
        # query flaked (e.g. during rapid OCA placement).  Never fire false
        # exits — keep the previous brain state and retry next cycle.
        if not ib_positions and self.brain._open_positions:
            log.debug(
                "sync_from_ib: IB returned 0 positions but brain has %d — skipping exit scan",
                len(self.brain._open_positions),
            )
            return
        self._guard_netting(streamer, closing or set())
        self._notify_open_fills(ib_positions)
        # Fire exit for any tracked position that IB no longer reports.
        # Use _open_positions (not just _brackets) so adopted/orphan
        # positions without OCA protection are still learned from.
        for t in set(self.brain._open_positions) - set(ib_positions):
            self._record_exit(t, streamer)
        self.brain._open_positions = ib_positions
        now = time.monotonic()
        if now - self._last_snapshot >= 10.0:
            self._last_snapshot = now
            journal_snapshot(self.journal, self.ib, ib_positions, self._brackets)

    def _guard_netting(self, streamer: Any, closing: set[str]) -> None:
        """Flatten accidental shorts while long_only (netting-reversal guard)."""
        if TRADING_CONFIG.direction_mode != "long_only":
            return
        shorts = short_positions(self.ib, self.tracked_tickers)
        for sym, size in shorts.items():
            qty = int(size)
            if sym in closing or qty <= 0:
                continue
            if self._flattening.get(sym) == qty:
                continue
            if open_order_for(self.ib, sym, "BUY"):
                continue
            contract = streamer.contracts.get(sym)
            if contract is None:
                continue
            cancel_sell_legs(self.ib, sym)
            try:
                self.ib.placeOrder(
                    contract,
                    _ib.MarketOrder(
                        "BUY", qty, tif="DAY", outsideRth=ALLOW_EXTENDED_HOURS
                    ),
                )
            except Exception as exc:
                log.warning("netting guard flatten %s failed: %s", sym, exc)
                continue
            self._brackets.pop(sym, None)
            self._pending_parent.discard(sym)
            self._flattening[sym] = qty
            log.info("NETTING GUARD: flattened accidental short %s qty=%d", sym, qty)
        for sym in list(self._flattening):
            if shorts.get(sym) is None:
                self._flattening.pop(sym, None)

    def _adopt_orphan_positions(
        self, streamer: Any, closing: set[str] | None = None
    ) -> None:
        """Adopt IB positions not placed by this bot session.

        Reads all IB positions, seeds synthetic entries for orphans,
        adds them to tracked_tickers, subscribes market data, and
        registers them for exit monitoring so the brain learns from
        their full lifecycle. Positions currently being closed
        (``closing``) are skipped — never adopt a position that is in
        the middle of a flatten, or we re-protect what we're selling.
        """
        closing = closing or set()
        try:
            ib_positions = self.ib.positions()
        except Exception:
            return
        for pos in ib_positions:
            sym = pos.contract.symbol
            qty = int(pos.position)
            if qty == 0:
                continue
            if sym in self.last_thoughts:
                continue
            if sym in closing:
                log.debug("RECONCILE: skip %s (closing)", sym)
                continue
            if sym not in self.tracked_tickers:
                self.tracked_tickers.add(sym)
                log.info("RECONCILE: adopted %s (qty=%d, avg=%.2f)",
                         sym, qty, pos.avgCost)
                try:
                    # Subscribe only — history seeding happens one ticker
                    # per cycle off the hot path in _sync_subs, so adoption
                    # of many orphans no longer blocks the main loop.
                    streamer.subscribe(sym)
                except Exception as e:
                    log.debug("RECONCILE sub %s failed: %s", sym, e)
            self.last_thoughts[sym] = {
                "ticker": sym,
                "direction": 1 if qty > 0 else -1,
                "price": pos.avgCost,
                "shares": abs(qty),
                "synthetic": True,
            }
            self._synthetic.add(sym)
            self._horizons[sym] = "scalp"

    def _ping_ib(self) -> bool:
        """Verify IB connection is alive (safety before sync)."""
        try:
            return bool(self.ib.isConnected())
        except Exception:
            return False

    def _record_exit(self, ticker: str, _streamer: Any) -> None:
        """Record a closed position to journal and brain."""
        self._brackets.pop(ticker, None)
        self._horizons.pop(ticker, None)
        pos = self.brain._open_positions.pop(ticker, None)
        if pos is None:
            return
        pnl = get_ib_pnl(self.ib, ticker, pos)
        is_synthetic = ticker in self._synthetic
        self._synthetic.discard(ticker)
        trade_closed(
            ticker,
            "LONG" if pos.direction > 0 else "SHORT",
            pnl,
            reason="reconciled" if is_synthetic else "",
            extra=self._close_summary(),
        )
        if is_synthetic:
            log.info("EXIT %s (reconciled close, P&L=%.4f) — learn from exit",
                     ticker, pnl)
        else:
            log.info("EXIT %s (IB closed at P&L=%.4f)", ticker, pnl)
            self.brain.record_trade(
                ticker=ticker, won=pnl > 0, pnl_pct=pnl, direction=pos.direction
            )
            journal_exit(self.journal, ticker, pnl, pos)
        self._closed_trades.append(
            {
                "ticker": ticker,
                "pnl": pnl,
                "return_pct": pnl,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "shares": pos.shares,
                "source": "reconciled_exit" if is_synthetic else "ib_fill",
            }
        )

    def _close_summary(self) -> str:
        """Enrich close notifications with IB account + realized context."""
        feed = self._account_feed or {}
        daily = feed.get("daily_pnl")
        parts = []
        if feed.get("equity") is not None:
            parts.append(f"Account ${feed['equity']:,.0f} | IB day {daily:+.2f}")
        if self._winrate_provider is not None:
            try:
                rate, n = self._winrate_provider()
                if n:
                    parts.append(f"JULI WR {rate * 100:.1f}% (n={n})")
            except Exception as exc:
                log.debug("winrate unavailable: %s", exc)
        return "\n".join(parts)

    def _confirm_fill_hook(self, sym: str, entry_price: float) -> None:
        """Call the cycle's fill-confirmed accounting (if wired)."""
        if self.on_fill_confirmed is None:
            return
        try:
            self.on_fill_confirmed(sym, entry_price)
        except Exception as exc:
            log.debug("fill-confirm accounting failed for %s: %s", sym, exc)

    def _notify_open_fills(self, ib_positions: dict[str, Any]) -> None:
        """Fire fill-confirmed entry notifications for pending brackets."""
        for sym in list(self._pending_parent):
            pos = ib_positions.get(sym)
            if pos is None:
                continue
            self._pending_parent.discard(sym)
            stop, target = self._brackets.get(sym, (0.0, 0.0))
            trade_opened(
                sym,
                "BUY" if pos.direction > 0 else "SELL",
                int(pos.shares),
                float(pos.entry_price or 0.0),
                ExitLevels(stop=float(stop or 0.0), target=float(target or 0.0)),
            )
            self._confirm_fill_hook(sym, float(pos.entry_price or 0.0))

    def monitor_orders(self, ib_positions: dict[str, Any], streamer: Any) -> None:
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

    def _handle_parent(
        self,
        trade: Any,
        order: Any,
        sym: str,
        ib_positions: dict[str, Any],
        streamer: Any,
    ) -> None:
        """Handle one parent order — cancel orphans, trail both."""
        sp = [
            c.auxPrice for c in getattr(order, "children", []) if c.auxPrice is not None
        ]
        tp = [
            c.lmtPrice for c in getattr(order, "children", []) if c.lmtPrice is not None
        ]
        stop, target = (float(max(sp)) if sp else 0.0), (float(max(tp)) if tp else 0.0)
        if sym in self._brackets:
            stored = self._brackets[sym]
            stop = stored[0] if stored[0] > 0 else stop
            target = stored[1] if stored[1] > 0 else target
        if sym in self._pending_parent:
            if sym in ib_positions:
                self._pending_parent.discard(sym)
                pos = ib_positions[sym]
                trade_opened(
                    sym,
                    "BUY" if pos.direction > 0 else "SELL",
                    int(abs(pos.shares)),
                    pos.entry_price,
                    ExitLevels(stop=stop, target=target),
                )
                self._confirm_fill_hook(sym, float(pos.entry_price or 0.0))
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
                if (cur - stop > atr) if d > 0 else (stop - cur > atr):
                    self._modify_child(trade, "stop", cur - d * ATR_STOP_MULT * atr)
                if (target - cur > atr) if d > 0 else (cur - target > atr):
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
        for child in getattr(trade.order, "children", []) or []:
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
            self.ib.placeOrder(
                contract,
                _ib.MarketOrder(
                    action, abs(pos.shares), tif="DAY", outsideRth=ALLOW_EXTENDED_HOURS
                ),
            )
            log.info("BRAIN EXIT %s %s %d", action, ticker, abs(pos.shares))
        except Exception as e:
            log.warning("close_position failed %s: %s", ticker, e)

    def close_all_positions(
        self,
        _streamer: Any,
        only: set[str] | None = None,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> int:
        """Flatten open positions.

        ``only`` restricts the flatten to specific tickers (horizon-aware
        EOD: intraday rungs close, overnight rungs hold).

        ``order_type`` controls whether IB receives a market or limit order.
        Market orders guarantee fills but may walk the book on illiquid
        names; limit orders protect price but may not fill.

        ``limit_price`` is used when ``order_type="limit"``.  When omitted
        for a limit flatten, the current market price is used as the limit.
        """
        count = 0
        try:
            ib_positions = self.ib.positions()
        except Exception:
            ib_positions = []
        for pos in ib_positions:
            sym = pos.contract.symbol
            if only is not None and sym not in only:
                continue
            qty = abs(int(pos.position))
            if qty == 0:
                continue
            action = "SELL" if pos.position > 0 else "BUY"
            try:
                if order_type == "limit":
                    if limit_price is not None:
                        lp = float(limit_price)
                    else:
                        lp = _streamer.get_last_price(sym) or 0.0
                    order = _ib.Order(
                        orderType="LMT",
                        action=action,
                        totalQuantity=qty,
                        lmtPrice=lp,
                        tif="DAY",
                        outsideRth=True,
                    )
                else:
                    order = _ib.Order(
                        orderType="MKT",
                        action=action,
                        totalQuantity=qty,
                        tif="DAY",
                        outsideRth=True,
                    )
                self.ib.placeOrder(pos.contract, order)
                log.info("FLATTEN %s %s %d (%s)", action, sym, qty, order_type)
                count += 1
            except Exception as e:
                log.warning("FLATTEN failed %s: %s", sym, e)
        return count

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
