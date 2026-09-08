"""hanoon_prime.ib_cycle — IB streaming cycle and execution helpers.

Extracted from ib_adapter to keep files under R3 limit (200 lines).
Contains BotCycleMixin with cycle, sync, execution, and safety methods
that IBStreamingBot mixes in. Also provides connect helpers.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from ._ib_sync import read_portfolio
from ._telegram import safety_halt, shutdown
from .brain.horizons import holds_through_close
from .brain.probe_recovery import probe_recovery as _PROBE_RECOVERY
from .config import TRADING_CONFIG
from .immune import (
    CONSECUTIVE_LOSSES_PAUSE,
    DAILY_LOSS_LIMIT,
    ENTRY_REUSE_COOLDOWN_SEC,
    MAX_CONCURRENT_POSITIONS,
    MAX_ENTRIES_PER_CYCLE,
    PENNY_PRICE,
    PENNY_SCORE_BAR,
)
from .monitor.portfolio_risk import PortfolioRiskManager
from .monitor.sleep_manager import SleepManager

RISK_SYNC_SECS: float = 30.0  # portfolio-risk equity refresh cadence
STALE_SUB_SECS: float = 60.0  # subscription GC: unsubscribe after this idle
CYCLE_FLOOR: float = 0.2  # minimum gap between cycles even when overran

log = logging.getLogger(__name__)
_SLEEP_MGR = SleepManager()
_PORTFOLIO_RISK = PortfolioRiskManager()


@dataclass
class CycleMeta:
    """Timing/session context bundled into _finish_cycle."""

    poll: float
    started: float
    market_open: bool = True


class SafetyNetStopped(Exception):
    """Safety net triggered — trading halted."""


def count_open_positions(ib_client: Any, tracked: set[str]) -> int:
    """Count open positions in tracked set."""
    try:
        return len([p for p in ib_client.positions() if p.contract.symbol in tracked])
    except Exception:
        return 0


def try_connect(ib_client: Any, host: str, port: int, cid: int) -> bool:
    """Attempt to connect to IB Gateway."""
    try:
        ib_client.connect(host, port, clientId=cid)
        return bool(ib_client.isConnected())
    except Exception as e:
        log.warning("Connect failed: %s", e)
        return False


class BotCycleMixin:
    """Mixin providing cycle loop, execution, and safety for IBStreamingBot."""

    # ── Gateway supervision (rebuild runner_gateway.py port) ────────────

    def _supervise_gateway(self) -> None:
        """Detect a dropped Gateway and reconnect with backoff + re-sync.

        Rebuild lesson (runner_gateway.py): after reconnecting you MUST
        re-request market data — IB silently drops subscriptions on
        disconnect, and without re-subscribe the watchdog sees stale
        ticks and panics. 1s → 2s → 4s → 8s → 16s → 30s (cap).
        """
        try:
            connected = bool(self.ib.isConnected())
        except Exception as exc:
            log.debug("isConnected check failed: %s", exc)
            connected = False
        if connected:
            self._gw_was_connected = True
            self._gw_attempts = 0
            return
        if self._gw_was_connected:
            self._gw_was_connected = False
            self._gw_attempts = 0
            log.warning("GATEWAY: connection lost — reconnecting")
        delay = min(30.0, 1.0 * (2 ** min(self._gw_attempts, 4)))
        time.sleep(delay)
        self._gw_attempts += 1
        if self._reconnect():
            self._resubscribe_all()

    def _reconnect(self) -> bool:
        """One reconnect attempt via the adapter's retrying connect()."""
        try:
            self.connect(self.ib.host, self.ib.port, self.ib.clientId)
        except Exception as exc:
            log.warning("GATEWAY: reconnect attempt failed: %s", exc)
            return False
        ok = False
        try:
            ok = bool(self.ib.isConnected())
        except Exception as exc:
            log.debug("post-reconnect check failed: %s", exc)
        if ok:
            self._gw_was_connected = True
            self._gw_attempts = 0
            # History seeded pre-disconnect is stale — force re-seed with
            # positions first (they take priority in _sync_subs ordering).
            self.__dict__.setdefault("_seeded_subs", set()).clear()
            self.streamer.touch(set(self.hippocampus._open_positions))
            log.warning("GATEWAY: reconnected — re-seeding position history")
        return ok

    def _resubscribe_all(self) -> None:
        """Re-request market data after a reconnect (IB drops subs silently)."""
        targets = set(self.executor.tracked_tickers) | set(self.streamer.ticker_subs)
        for t in targets:
            try:
                self.streamer.subscribe(t)
            except Exception as exc:
                log.debug("re-subscribe %s failed: %s", t, exc)
        for t in set(self.hippocampus._open_positions):
            try:
                self.streamer.seed_history(t)
            except Exception as exc:
                log.debug("re-seed %s failed: %s", t, exc)

    def _sweep_stale_orders(self) -> None:
        """Brain-aware stale-order reconcile (JULI fast path, every cycle).

        Never cancels flatten orders (symbols in ``_closing``) or GTC
        protection (STP+LMT from _protect.py) — those must survive until
        the position closes. Only DAY entry parents that are stale (>60s)
        AND whose symbol has no open position are swept individually.

        Rebuild lesson (runner_protocol.md): the old timer swept ANY DAY-tif
        parent after 60s — including flatten MKT orders for _closing symbols
        — so it cancelled in-flight closes, stranding positions open while
        still flagged closing. That freeze is cleared in ``_reconcile_closing``.
        """
        self._reconcile_closing()
        pending = ("PendingSubmit", "PreSubmitted")
        try:
            trades = list(self.ib.openTrades())
        except Exception as exc:
            log.debug("openTrades failed: %s", exc)
            return
        now = time.time()
        for trade in trades:
            order = getattr(trade, "order", None)
            if not order or order.parentId:
                continue
            sym = trade.contract.symbol if trade.contract else "?"
            # Flatten orders (closing) or protection (GTC) — never touch.
            if sym in self._closing or getattr(order, "tif", "") != "DAY":
                continue
            if trade.orderStatus.status not in pending:
                continue
            # Only cancel a stale entry parent when no open position needs it.
            if sym in self.hippocampus._open_positions:
                continue
            self._sweep_one(order, sym, now)

    def _halim_order_review(self) -> None:
        """HALIM slow-path order review backstop (throttled)."""
        now = time.time()
        last = getattr(self, "_last_halim_review", 0.0)
        if now - last < 120.0:
            return
        self._last_halim_review = now
        halim = getattr(self.juli.brain, "_consolidation", None)
        if halim is None:
            return
        try:
            trades = [t for t in self.ib.openTrades() if not t.isDone()]
        except Exception as exc:
            log.debug("halim review openTrades failed: %s", exc)
            return
        if not trades:
            return
        orders = self._open_order_summary(trades)
        positions = {
            sym: {"direction": p.direction, "qty": int(abs(p.shares))}
            for sym, p in self.hippocampus._open_positions.items()
        }
        for intent in halim.halim.review_open_orders(orders, positions):
            if intent.get("action") == "cancel":
                self._halim_apply_cancel(intent.get("ticker", ""), trades)

    def _open_order_summary(self, trades: list[Any]) -> list[dict[str, Any]]:
        """Compact summary of active trades for the HALIM order audit."""
        orders = []
        for t in trades:
            o = getattr(t, "order", None)
            if o is None:
                continue
            orders.append(
                {
                    "ticker": (t.contract.symbol if t.contract else "?"),
                    "action": o.action or "?",
                    "type": o.orderType or "?",
                    "qty": int(o.totalQuantity or 0),
                    "status": (t.orderStatus.status if t.orderStatus else "?"),
                }
            )
        return orders

    def _halim_apply_cancel(self, sym: str, trades: list[Any]) -> None:
        """Re-validate one HALIM cancel intent against JULI's safety rules."""
        if not sym or sym in self._closing or sym in self.hippocampus._open_positions:
            return
        for trade in trades:
            o = getattr(trade, "order", None)
            if o is None or o.parentId or not trade.isDone():
                continue
            if (trade.contract and trade.contract.symbol) != sym:
                continue
            if getattr(o, "tif", "") != "DAY":
                continue
            log.info("HALIM CANCEL %s: slow-path review", sym)
            try:
                self.ib.cancelOrder(o)
            except Exception as exc:
                log.warning("HALIM cancel %s failed: %s", sym, exc)
            break

    def _reconcile_closing(self) -> None:
        """Release _closing symbols whose close order died without a fill.

        A symbol lands in ``_closing`` when a flatten/exit order is placed.
        If IB cancels that order (timer sweep, server reject, disconnect)
        with filled=0, the position stays OPEN while every acting path
        skips it forever. Detect the dead state and release it.

        Safety: NEVER release a symbol that IB shows flat but ``_open_positions``
        still holds (sync read it open earlier the same cycle, then the fill
        landed). Dropping the flag there makes the exit evaluator re-fire a
        duplicate order that flips the position. Let sync_from_ib drop the
        stale position first.
        """
        if not self._closing:
            return
        try:
            ib_positions = {
                p.contract.symbol
                for p in self.ib.positions()
                if abs(int(p.position)) > 0
            }
            active = {t.contract.symbol for t in self.ib.openTrades() if not t.isDone()}
        except Exception as exc:
            log.debug("RECONCILE closing scan unavailable: %s", exc)
            return
        for sym in list(self._closing):
            if sym in ib_positions and sym in active:
                continue  # live close order still in flight
            if sym in ib_positions:
                self._closing.discard(sym)
                log.warning("RECONCILE: orphan released %s (close died)", sym)
                continue
            if sym in self.hippocampus._open_positions:
                continue  # stale in-memory position; sync_from_ib removes it
            self._closing.discard(sym)
            log.warning("RECONCILE: released %s from closing (pos flat)", sym)

    def _sweep_one(self, order: Any, sym: str, now: float) -> None:
        """Cancel one stale pending parent (tracked ≥ 60s)."""
        oid = order.orderId
        placed = self._order_placed_ts.get(oid)
        if placed is None:
            self._order_placed_ts[oid] = now
            return
        if now - placed <= 60.0:
            return
        log.info("SWEEP: stale pending parent %s (%.0fs)", sym, now - placed)
        try:
            self.ib.cancelOrder(order)
        except Exception as exc:
            log.warning("SWEEP cancel %s failed: %s", sym, exc)
        self._order_placed_ts.pop(oid, None)

    def _snapshot(self, sym: str) -> dict[str, Any] | None:
        """Build snapshot dict from live ticker data."""
        tk = self.streamer.ticker_subs.get(sym)
        if tk is None or not tk.hasBidAsk:
            return None
        base: dict[str, Any] = {
            "bid": float(tk.bid),
            "ask": float(tk.ask),
            "last": float(tk.last or tk.close or 0),
            "volume": float(tk.volume or 0),
            "daily_volume": float(tk.volume or 0),
        }
        if not self.streamer.ready(sym):
            return base
        a = self.streamer.get_arrays(sym)
        base["atr"] = self.streamer.buffer_atr(sym)
        for k in (
            "close",
            "volume",
            "high",
            "low",
            "buy_volume",
            "bid_sizes",
            "ask_sizes",
        ):
            base[f"{k}_arr"] = a[k]
        base["prices"], base["volumes"] = list(a["close"]), list(a["volume"])
        return base

    def _cycle(self, poll: float, pnl: Any) -> None:
        """One main loop iteration."""
        started = time.monotonic()
        try:
            self._supervise_gateway()
            self.executor.sync_from_ib(self.streamer, closing=self._closing)
            self._sweep_stale_orders()
            self._check_safety(pnl)
            self._sync_subs()
            if self.monitor.pop_heal():
                log.warning("PIPELINE HEAL: forcing re-subscribe")
                self._sync_subs()
            # Manual flatten request from webapp
            if self._check_manual_flatten():
                self._finish_cycle([], [], pnl, CycleMeta(poll, started, False))
                return
            # EOD flatten: force-close all positions in last N minutes of RTH
            if self._check_eod_flatten():
                self._finish_cycle([], [], pnl, CycleMeta(poll, started, False))
                return
            positions = set(self.hippocampus._open_positions.keys())
            market_open = _SLEEP_MGR.get_state().active
            pos_info = {
                t: {
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "stop_price": p.stop_price,
                }
                for t, p in self.hippocampus._open_positions.items()
            }
            self._halim_order_review()
            exit_s, decisions = self.juli.tick(
                positions, self._snapshot, self.streamer, self._closing, pos_info
            )
            self._finish_cycle(
                exit_s, decisions, pnl, CycleMeta(poll, started, market_open)
            )
        except Exception as e:
            log.error("Cycle error: %s", e, exc_info=True)

    def _finish_cycle(
        self,
        exit_s: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        pnl: Any,
        meta: CycleMeta,
    ) -> None:
        """Process tickers, exits, decisions, reflect, wait."""
        self._cycle_entries = 0  # reset per-cycle entry order cap
        self._last_bars = sum(
            1
            for tk in self.ib.pendingTickers()
            if self.streamer.update_bar(tk.contract.symbol if tk.contract else "")
        )
        self._drain_event_exits()
        for es in exit_s:
            t = es["ticker"]
            if t not in self._closing:
                self._closing.add(t)
                self.executor.close_position(t, self.streamer)
                self._exit_reasons[t] = es.get("type", "brain_exit")
                log.info("EXIT %s: %s", t, es.get("reason", ""))
        for dec in decisions:
            if meta.market_open and self._can_trade(dec):
                self._exec_decision(dec)
        self._reflect_closed()
        self.monitor.record_cycle(meta.market_open)
        if pnl is not None:
            daily = float(pnl.dailyPnL)
            self.hippocampus._daily_pnl = daily
        self._sync_portfolio_risk()
        elapsed = time.monotonic() - meta.started
        gap = max(CYCLE_FLOOR, meta.poll - elapsed)
        time.sleep(gap)
        self._heartbeat()
        npos = len(self.hippocampus._open_positions)
        log.info("CYCLE bars=%d open=%d d=%d x=%d", self._last_bars, npos, len(decisions), len(exit_s))

    def _sync_portfolio_risk(self) -> None:
        """Feed IB-reported equity into the portfolio risk manager (throttled).

        Reads NetLiquidation once per RISK_SYNC_SECS; the manager derives
        drawdown → risk scalar → entry block internally.
        """
        now = time.monotonic()
        if now - getattr(self, "_last_risk_sync", 0.0) < RISK_SYNC_SECS:
            return
        self._last_risk_sync = now
        try:
            summary = self.ib.accountSummary(self.account)
            net_liq = next(
                (float(i.value) for i in summary if i.tag == "NetLiquidation"),
                0.0,
            )
            _PORTFOLIO_RISK.update_equity(net_liq)
            _PORTFOLIO_RISK.update_positions(read_portfolio(self.ib))
            giveback = _PORTFOLIO_RISK.check_portfolio_giveback()
            for t in giveback.tickers:
                if t not in self._closing:
                    self._closing.add(t)
                    self._exit_reasons[t] = "portfolio_giveback"
                    self.executor.close_position(t, self.streamer)
                    log.warning(
                        "GIVEBACK EXIT %s (fade=%.0f%% peak=$%.0f now=$%.0f)",
                        t,
                        giveback.fade * 100,
                        giveback.peak,
                        giveback.unrealized,
                    )
        except Exception as e:
            log.debug("Portfolio risk sync skipped: %s", e)

    def _sync_subs(self) -> None:
        """Sync subscriptions: async mkt data for all, one seed per cycle.

        GC unsubscribes stale scanner tickers to free MD lines.
        Positions are always touched (never collected).
        """
        tracked = self.juli.budget.get_all_tracked()
        scanner = {c.symbol for c in self.juli._candidates[:20]}
        needed = tracked | scanner | set(self.hippocampus._open_positions.keys())
        self.executor.tracked_tickers = tracked
        self.streamer.touch(needed)
        missing = [s for s in sorted(needed) if s not in self.streamer.ticker_subs]
        for s in missing:
            try:
                self.streamer.subscribe(s)
            except Exception as e:
                log.warning("Sub %s fail: %s", s, e)
        self._gc_stale_subs()
        # Seed history one ticker per cycle (blocking call kept out of
        # the hot path); positions take priority over scanner candidates.
        seeded = self.__dict__.setdefault("_seeded_subs", set())
        pending = [
            s
            for s in sorted(set(self.hippocampus._open_positions))
            + sorted(set(missing))
            if s in self.streamer.ticker_subs and s not in seeded
        ]
        if pending:
            s = pending[0]
            try:
                self.streamer.seed_history(s)
                seeded.add(s)
            except Exception as e:
                log.debug("Seed %s fail: %s", s, e)

    def _gc_stale_subs(self) -> None:
        """Unsubscribe tickers not seen in STALE_SUB_SECS (frees MD lines)."""
        now = time.time()
        stale = [
            t
            for t, seen in self.streamer.last_seen.items()
            if now - seen > STALE_SUB_SECS
        ]
        for t in stale:
            if t in set(self.hippocampus._open_positions):
                self.streamer.touch({t})  # positions are never collected
                continue
            self.streamer.unsubscribe(t)

    def _check_manual_flatten(self) -> bool:
        """Check if webapp requested a manual flatten."""
        from .telemetry import _FLATTEN_REQUESTED

        if not _FLATTEN_REQUESTED:
            return False
        _FLATTEN_REQUESTED.clear()
        # Count actual IB positions, not just Juli-tracked ones
        try:
            ib_count = len([p for p in self.ib.positions() if abs(int(p.position)) > 0])
        except Exception:
            ib_count = 0
        if ib_count == 0:
            log.info("MANUAL FLATTEN: no IB positions to flatten")
            return False
        # Mark every position as closing so adoption/protection skips them
        # while the flatten market orders are in flight.
        try:
            for p in self.ib.positions():
                if abs(int(p.position)) > 0:
                    self._closing.add(p.contract.symbol)
        except Exception as e:
            log.debug("flatten closing-mark failed: %s", e)
        log.warning("MANUAL FLATTEN: closing %d IB positions", ib_count)
        closed = self.executor.close_all_positions(self.streamer)
        log.warning("MANUAL FLATTEN: sent market orders for %d positions", closed)
        return True

    def _check_eod_flatten(self) -> bool:
        """If EOD window, flatten scalp/multihour/swing positions.

        Horizon-aware (rebuild v4 lesson): positions classified multiday
        or longer are DESIGNED to hold through sessions — flattening them
        at the close defeats their thesis. Only intraday horizons force-
        close; longer rungs survive until their own exit policy fires.
        """
        from .config import TRADING_CONFIG

        if not TRADING_CONFIG.eod_flatten_enabled:
            return False
        if not _SLEEP_MGR.is_eod_window(TRADING_CONFIG.eod_flatten_minutes):
            return False
        remaining = _SLEEP_MGR.minutes_to_close()
        intraday = [
            t
            for t in self.hippocampus._open_positions
            if not holds_through_close(self.executor._horizons.get(t, "scalp"))
        ]
        if not intraday:
            log.info(
                "EOD: only overnight-safe horizons open (%.1f min to close)",
                remaining,
            )
            return False
        log.warning(
            "EOD FLATTEN: %.1f min to close, %d intraday positions — forcing market close",
            remaining,
            len(intraday),
        )
        closed = self.executor.close_all_positions(self.streamer, only=set(intraday))
        log.warning("EOD FLATTEN: sent limit orders for %d positions", closed)
        return bool(closed)

    def _can_trade(self, dec: dict[str, Any]) -> bool:
        """Check session and direction config before trading."""
        if self._halted:
            return False
        # Check direction mode (thought is a SimpleNamespace, not dict)
        thought = dec.get("thought")
        side = getattr(thought, "verdict", "BUY") if thought else "BUY"
        if not TRADING_CONFIG.is_direction_allowed(side):
            log.debug(
                "SKIP %s direction=%s mode=%s",
                dec["ticker"],
                side,
                TRADING_CONFIG.direction_mode,
            )
            return False
        # Raise-the-bar for sub-dollar tickers: the brain must be EXTREMELY
        # sure of a quick scalp before we accept a micro-cap setup. This is
        # not a hard price block — the higher bar can still be cleared.
        t = dec["ticker"]
        tk = self.streamer.ticker_subs.get(t)
        if tk is not None and tk.hasBidAsk:
            last = float(tk.last or tk.close or 0)
            mid = float((tk.bid + tk.ask) * 0.5) if tk.bid and tk.ask else 0.0
            px = mid or last
            if px < PENNY_PRICE and abs(dec.get("score", 0.0)) < PENNY_SCORE_BAR:
                log.info(
                    "SKIP %s sub_dollar_bar price=%.3f score=%.3f (need >= %.2f)",
                    t,
                    px,
                    abs(dec.get("score", 0.0)),
                    PENNY_SCORE_BAR,
                )
                return False
        # Check session (sleep_manager already gates overall, but double-check)
        state = _SLEEP_MGR.get_state()
        if not TRADING_CONFIG.is_session_active(state.session):
            log.debug("SKIP %s session=%s disabled", dec["ticker"], state.session)
            return False
        return True

    def _portfolio_gate_and_size(
        self, t: str, tk: Any, dec: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Portfolio risk gate + size adjustment (rebuild port).

        Returns the (possibly resized) sizing dict to execute, or None
        when the entry is blocked or sized to zero.
        """
        sizing = dec.get("sizing")
        if sizing is None or getattr(sizing, "shares", 0) <= 0:
            log.debug("SKIP %s sizing=0", t)
            return None
        price = float((tk.bid + tk.ask) * 0.5)
        shares = int(sizing.shares)
        allowed, reason = _PORTFOLIO_RISK.pre_trade_risk_gate(t, abs(shares * price))
        if not allowed:
            log.info("SKIP %s portfolio_risk (%s)", t, reason)
            return None
        adj = _PORTFOLIO_RISK.adjust_size(shares, price)
        if adj != shares:
            log.info(
                "SIZE %s %d->%d (scalar=%.2f)",
                t,
                shares,
                adj,
                _PORTFOLIO_RISK._risk_scalar,
            )
            sizing.shares = adj
            if adj <= 0:
                log.info("SKIP %s sized_to_zero", t)
                return None
        return {"sizing": sizing, "price": price}

    def _entry_throttled(self, t: str) -> bool:
        """Full-parallel-throttle: cap entries per cycle + per-ticker cooldown.

        Returns True if this entry should be deferred, False if it may
        place. Kept as its own method to stay under the R3 40-line limit.
        """
        if getattr(self, "_cycle_entries", 0) >= MAX_ENTRIES_PER_CYCLE:
            log.debug("THROTTLE %s: cycle order cap reached", t)
            return True
        last_enter = getattr(self, "_entry_reuse_ts", {}).get(t, 0.0)
        if time.time() - last_enter < ENTRY_REUSE_COOLDOWN_SEC:
            log.debug("THROTTLE %s: reuse cooldown", t)
            return True
        return False

    def _entry_safety_cleared(self, t: str, tk: Any, dec: dict[str, Any]) -> bool:
        """Death-spiral PROBE recovery (off-by-default; on in production):
        one quality-gated entry bypasses the halt to re-establish edge and
        resume learning. check_entry_allowed's halt is UNCHANGED while
        PROBE_RECOVERY_ENABLED is False. Returns False to block the entry.
        """
        if self.hippocampus.check_entry_allowed():
            return True
        thought = dec.get("thought")
        if _PROBE_RECOVERY.maybe_probe(
            getattr(thought, "score", 0.0),
            float(tk.bid),
            float(tk.ask),
            self.hippocampus.consecutive_losses,
        ):
            log.info("PROBE %s death_spiral_recovery", t)
            return True
        log.info("SKIP %s safety", t)
        return False

    def _exec_decision(self, dec: dict[str, Any]) -> None:
        """Execute an entry decision through IB (throttled, fill-confirmed)."""
        tk = self.streamer.ticker_subs.get(dec["ticker"])
        if tk is None or not tk.hasBidAsk or math.isnan(tk.bid) or math.isnan(tk.ask):
            return
        t = dec["ticker"]
        if self._entry_throttled(t):
            return
        if not self._entry_safety_cleared(t, tk, dec):
            return
        if t in self.hippocampus._open_positions:
            log.info("SKIP %s open", t)
            return
        exec_pack = self._portfolio_gate_and_size(t, tk, dec)
        if exec_pack is None:
            return
        horizon = str(dec.get("horizon", "scalp"))
        self.executor.place_bracket(
            t,
            dec["thought"],
            exec_pack["price"],
            self.streamer,
            sizing=exec_pack["sizing"],
            horizon=horizon,
        )
        self.executor.last_thoughts[t] = dec["thought"]
        self.juli.brain.register_position(t, exec_pack["price"], horizon=horizon)
        self._attach_position_watchers(t)
        # Throttle accounting only after a real bracket was placed.
        self._cycle_entries = getattr(self, "_cycle_entries", 0) + 1
        self._entry_reuse_ts = getattr(self, "_entry_reuse_ts", {})
        self._entry_reuse_ts[t] = time.time()

    def _attach_position_watchers(self, ticker: str) -> None:
        """Attach tick-driven exit watcher + per-position PnL stream.

        The watcher checks the bracket stop level on every tick (IB push,
        sub-cycle latency) and enqueues an exit signal — the main cycle
        drains it. Never blocks the socket thread.
        """
        if getattr(self, "_watched", None) is None:
            self._watched: set[str] = set()
        if ticker in self._watched:
            return
        self._watched.add(ticker)
        brackets = self.executor._brackets
        account = getattr(self, "account", "")

        def check(t: str, last: float, _tk: Any) -> str | None:
            """Return exit reason if stop breached, else None."""
            levels = brackets.get(t)
            if levels is None:
                return None
            stop, _target = levels
            return "hard_stop_breach" if last <= stop else None

        self.streamer.attach_exit_watcher(ticker, check)
        self.streamer.watch_pnl_single(ticker, account)

    def _drain_event_exits(self) -> None:
        """Process exit signals queued by tick watchers (non-blocking)."""
        for sig in self.streamer.drain_signals():
            if sig.get("type") != "exit":
                continue
            t = sig["ticker"]
            if t in self._closing or t not in self.hippocampus._open_positions:
                continue
            self._closing.add(t)
            self.executor.close_position(t, self.streamer)
            self._exit_reasons[t] = sig.get("reason", "event_exit")
            log.info("EXIT %s (event): %s", t, sig.get("reason", ""))

    def _reflect_closed(self) -> None:
        """Route closed trades to neuromorphic brain for learning."""
        for trade in self.executor.get_newly_closed_trades():
            won = trade["pnl"] > 0
            source = trade.get("source", "ib_fill")
            self.juli.brain.on_trade_close(
                ticker=trade["ticker"],
                won=won,
                pnl_pct=trade["return_pct"],
                direction=trade["direction"],
                source=source,
                exit_triggers=[
                    self._exit_reasons.pop(trade["ticker"], "manual_or_bracket")
                ],
            )
            self._closing.discard(trade["ticker"])
            self._watched.discard(trade["ticker"])
            self.streamer.unwatch_pnl_single(trade["ticker"])
            log.info(
                "REFLECT %s %s pnl=%.4f src=%s",
                trade["ticker"],
                "WIN" if won else "LOSS",
                trade["pnl"],
                source,
            )

    def _heartbeat(self) -> None:
        """Periodic status log."""
        now = time.monotonic()
        if now - self._last_beat < 60.0:
            return
        self._last_beat = now
        log.info(
            "HEARTBEAT open=%d journal=%d",
            len(self.hippocampus._open_positions),
            self.journal.count(),
        )

    def _check_safety(self, pnl: Any) -> None:
        """Check safety nets before trading (immune.py constants)."""
        if not self.hippocampus.safety_enabled:
            return
        n = count_open_positions(self.ib, self.executor.tracked_tickers)
        if n > MAX_CONCURRENT_POSITIONS:
            log.critical("SAFETY: %d > %d", n, MAX_CONCURRENT_POSITIONS)
            self._halt("too_many_positions")
            return
        if pnl is not None and float(pnl.dailyPnL) < -DAILY_LOSS_LIMIT:
            log.critical("SAFETY: P&L $%.2f", float(pnl.dailyPnL))
            self._halt("daily_loss_limit")
            return
        if self.hippocampus._consecutive_losses >= CONSECUTIVE_LOSSES_PAUSE:
            self._halt("consecutive_losses")

    def _halt(self, reason: str) -> None:
        """Emergency halt — block new entries, never stop the bot."""
        self._halted = True
        self.journal.append({"event": "halt", "reason": reason, "ts": time.time()})
        safety_halt(reason)

    def _cleanup(self, pnl: Any) -> None:
        """Shutdown all subsystems."""
        self.monitor.stop()
        self.juli.brain.stop()
        self.streamer.cancel_all()
        self.executor.cancel_all()
        self.juli.scanner.cancel_all()
        if pnl:
            self.ib.cancelPnL(self.account)
        if self.ib.isConnected():
            self.ib.disconnect()
        shutdown()


__all__ = ["SafetyNetStopped", "count_open_positions", "try_connect", "BotCycleMixin"]
