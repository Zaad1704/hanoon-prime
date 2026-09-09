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
from ._telegram import postmortem, shutdown, trade_hold
from .account_equity import resolve_account_equity
from .brain.horizons import holds_through_close
from .brain.policy.verdict import ENTER, Verdict
from .config import TRADING_CONFIG
from .monitor.sleep_manager import SleepManager, SleepState

RISK_SYNC_SECS: float = 30.0  # portfolio-risk equity refresh cadence
STALE_SUB_SECS: float = 60.0  # subscription GC: unsubscribe after this idle
CYCLE_FLOOR: float = 0.2  # minimum gap between cycles even when overran
SEED_RETRY_MAX: int = 3  # backfill retries before a ticker is left to live bars
HOLD_FIRST_MIN: float = 15.0  # first open-position hold notice (minutes)
HOLD_REPEAT_MIN: float = 60.0  # repeat hold notice every this many minutes

log = logging.getLogger(__name__)
_SLEEP_MGR = SleepManager()


def _is_hard_stop(es: dict[str, Any]) -> bool:
    """True when an exit decision is a protective price stop (never bar-derived)."""

    return str(es.get("reason", "")).startswith("hard_stop")


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
            state = _SLEEP_MGR.effective_state(TRADING_CONFIG)
            if not state.active:
                self._sleep_tick(state, poll)
                return
            if getattr(self, "_sleeping", False):
                self._sleeping = False
                log.info("SESSION WAKE: %s active", state.session)
            self._supervise_gateway()
            self.executor.sync_from_ib(self.streamer, closing=self._closing)
            self._sweep_stale_orders()
            self._sync_subs()
            if self.monitor.pop_heal():
                log.warning("PIPELINE HEAL: forcing re-subscribe")
                self._sync_subs()
            # Manual flatten request, then EOD flatten (both skip brain).
            if self._check_manual_flatten() or self._check_eod_flatten():
                self._finish_cycle([], [], pnl, CycleMeta(poll, started, False))
                return
            self._run_brain_cycle(poll, pnl, started)
            self._notify_holds()
        except Exception as e:
            log.error("Cycle error: %s", e, exc_info=True)

    def _sleep_tick(self, state: SleepState, poll: float) -> None:
        """Keepalive for a deactivated session: no brain, no orders, no HALIM.

        Gateway supervision and on-demand flatten stay live so risk
        controls still work while the system is fully asleep.
        """
        if not getattr(self, "_sleeping", False):
            self._sleeping = True
            log.info("SESSION SLEEP: %s inactive — whole system idle", state.session)
        try:
            if self._check_manual_flatten() or self._check_eod_flatten():
                self._finish_cycle(
                    [],
                    [],
                    None,
                    CycleMeta(poll, time.monotonic(), False),
                    session=state.session,
                )
                self._sleeping = True
                return
        except Exception as exc:
            log.error("Sleep flatten failed: %s", exc)
        try:
            self._supervise_gateway()
        except Exception as exc:
            log.debug("Gateway supervise during sleep: %s", exc)
        self._heartbeat()
        self.monitor.record_cycle(False, session=state.session)
        time.sleep(max(CYCLE_FLOOR, poll))

    def _run_brain_cycle(self, poll: float, pnl: Any, started: float) -> None:
        """Score the whole universe and finish the cycle (single funnel)."""
        positions = set(self.hippocampus._open_positions.keys())
        mkt_state = _SLEEP_MGR.effective_state(TRADING_CONFIG)
        if mkt_state.session == "pre_market":
            self.juli.brain.exits.reset_flat_pulses()
        pos_info = {
            t: {
                "direction": p.direction,
                "entry_price": p.entry_price,
                "stop_price": p.stop_price,
            }
            for t, p in self.hippocampus._open_positions.items()
        }
        self._halim_order_review()
        exit_s, verdicts = self.juli.tick(
            set(self.juli.budget.get_all_tracked()),
            self._snapshot,
            self.streamer,
            positions,
            self._closing,
            pos_info,
            session=mkt_state.session,
        )
        self._finish_cycle(
            exit_s,
            verdicts,
            pnl,
            CycleMeta(poll, started, mkt_state.active),
            session=mkt_state.session,
        )

    def _finish_cycle(
        self,
        exit_s: list[dict[str, Any]],
        verdicts: list[Verdict],
        pnl: Any,
        meta: CycleMeta,
        session: str = "rth",
    ) -> None:
        """Execute exits, journal/execute verdicts, reflect, wait."""
        self._last_bars = sum(
            1
            for tk in self.ib.pendingTickers()
            if self.streamer.update_bar(tk.contract.symbol if tk.contract else "")
        )
        if not self._last_bars or session == "pre_market":
            exit_s = [es for es in exit_s if _is_hard_stop(es)]  # protective only
        self._drain_event_exits()
        for es in exit_s:
            t = es["ticker"]
            if t not in self._closing:
                self._closing.add(t)
                self.executor.close_position(t, self.streamer)
                self._exit_reasons[t] = es.get("type", "brain_exit")
                log.info("EXIT %s: %s", t, es.get("reason", ""))
        if pnl is not None:
            daily = float(pnl.dailyPnL)
            self.hippocampus._daily_pnl = daily if math.isfinite(daily) else 0.0
        self._publish_account_feed(pnl)
        market_open = bool(meta and meta.market_open)
        for v in verdicts:
            self.journal.append({"event": "verdict", "ts": time.time(), **v.to_dict()})
            if meta and meta.market_open and v.action == ENTER:
                self._execute_verdict(v)
        self._reflect_closed()
        self.monitor.record_cycle(market_open, session=session)
        elapsed = time.monotonic() - meta.started
        gap = max(CYCLE_FLOOR, meta.poll - elapsed)
        time.sleep(gap)
        self._heartbeat()
        log.info("CYCLE bars=%d open=%d d=%d x=%d", self._last_bars, len(self.hippocampus._open_positions), len(verdicts), len(exit_s))

    def _publish_account_feed(self, pnl: Any) -> None:
        """Forward IB account facts to the slow cortex on BrainState."""
        daily = float(pnl.dailyPnL) if pnl is not None else 0.0
        if not math.isfinite(daily):
            daily = 0.0  # IB PnL stream starts as nan before the first update
        feed: dict[str, Any] = {"daily_pnl": daily, "ts": time.monotonic()}
        if time.monotonic() - getattr(self, "_last_policy_sync", 0.0) >= RISK_SYNC_SECS:
            self._last_policy_sync = time.monotonic()
            try:
                equity, synced = resolve_account_equity(self.ib, self.account)
                feed["positions"] = read_portfolio(self.ib)
                if equity is not None:
                    self._account_equity = equity
                    self._account_equity_synced = synced
            except Exception as exc:
                log.debug("Account sync skipped: %s", exc)
        carried = getattr(self, "_account_equity", None)
        if carried is not None:
            # Feed is rebuilt every cycle; carry the last known equity so
            # the slow-cortex pulse always sees it between 30s refresh ticks.
            feed["equity"] = carried
            feed["equity_synced"] = getattr(self, "_account_equity_synced", True)
        self.juli._state.update(
            account_feed=feed,
            consecutive_losses=getattr(self.hippocampus, "_consecutive_losses", 0),
        )

    def _execute_verdict(self, verdict: Verdict) -> None:
        """Execute one ENTER verdict (live bid/ask, sizing, open skip)."""
        ticker = verdict.ticker
        sizing = verdict.sizing
        if sizing is None or getattr(sizing, "shares", 0) <= 0:
            log.info("SKIP %s sizing=0", ticker)
            return
        tk = self.streamer.ticker_subs.get(ticker)
        if tk is None or not tk.hasBidAsk or math.isnan(tk.bid) or math.isnan(tk.ask):
            log.warning("VERDICT UNEXECUTABLE %s: no live bid/ask", ticker)
            return
        if ticker in self.hippocampus._open_positions:
            log.info("SKIP %s open", ticker)
            return
        price = float((tk.bid + tk.ask) * 0.5)
        self.executor.place_bracket(
            ticker,
            verdict.thought,
            price,
            self.streamer,
            sizing=verdict.sizing,
            horizon=verdict.horizon,
        )
        self.executor.last_thoughts[ticker] = verdict.thought
        self.juli.brain.note_entry(ticker)

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
        self._seed_next_backfill(needed)

    def _seed_next_backfill(self, needed: set[str]) -> None:
        """Seed one ticker per cycle: any desired ticker missing backfill.

        Positions take priority over scanner candidates. Every desired
        subscribed ticker is eligible (not only same-cycle subscriptions),
        and a failed seed retries up to SEED_RETRY_MAX instead of
        stranding the ticker on live-bars-only data.
        """
        seeded = self.__dict__.setdefault("_seeded_subs", set())
        retries = self.__dict__.setdefault("_seed_retries", {})
        ordered = list(
            dict.fromkeys(
                sorted(set(self.hippocampus._open_positions)) + sorted(needed)
            )
        )
        pending = [s for s in ordered if s in self.streamer.ticker_subs and s not in seeded]
        if not pending:
            return
        s = pending[0]
        try:
            self.streamer.seed_history(s)
            seeded.add(s)
            retries.pop(s, None)
        except Exception as e:
            retries[s] = retries.get(s, 0) + 1
            if retries[s] >= SEED_RETRY_MAX:
                seeded.add(s)  # give up; live bars still accumulate
            log.debug("Seed %s fail (%d/%d): %s", s, retries[s], SEED_RETRY_MAX, e)

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

    def _confirm_fill(self, ticker: str, entry_price: float) -> None:
        """Account an entry only after IB reports the fill (post-fill).

        Wired as executor.on_fill_confirmed; registers the exits-tracker
        entry and attaches tick/PnL watchers once the bracket actually
        fills — never at order placement.
        """
        try:
            self.juli.brain.register_position(
                ticker, entry_price, horizon=self.executor._horizons.get(ticker, "scalp")
            )
        except Exception as exc:
            log.debug("fill register failed for %s: %s", ticker, exc)
        self._attach_position_watchers(ticker)

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
        trades = self.executor.get_newly_closed_trades()
        for trade in trades:
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
            self._hold_notified.pop(trade["ticker"], None)
            self.streamer.unwatch_pnl_single(trade["ticker"])
            log.info(
                "REFLECT %s %s pnl=%.4f src=%s",
                trade["ticker"],
                "WIN" if won else "LOSS",
                trade["pnl"],
                source,
            )
        if trades and not self.hippocampus._open_positions:
            self._send_postmortem()

    def _send_postmortem(self) -> None:
        """After the book goes flat, forward HALIM's post-mortem verbatim."""
        try:
            insight = self.juli.brain.state.get("halim_last_insight") or {}  # array-safe
            if insight:
                postmortem(insight)
        except Exception as exc:
            log.debug("postmortem notify failed: %s", exc)

    def _notify_holds(self) -> None:
        """Send hold milestone notifications for open positions."""
        try:
            for t in tuple(self.hippocampus._open_positions):
                pos = self.hippocampus._open_positions[t]
                mins = self.juli.brain.exits.hold_minutes(t)
                last = self._hold_notified.get(t, 0.0)
                if (last <= 0.0 and mins >= HOLD_FIRST_MIN) or (
                    last > 0.0 and mins - last >= HOLD_REPEAT_MIN
                ):
                    self._hold_notified[t] = mins
                    side = "LONG" if pos.direction > 0 else "SHORT"
                    trade_hold(t, mins, side)
        except Exception as exc:
            log.debug("hold notify failed: %s", exc)

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
