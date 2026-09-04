"""hanoon_prime.ib_cycle — IB streaming cycle and execution helpers.

Extracted from ib_adapter to keep files under R3 limit (200 lines).
Contains BotCycleMixin with cycle, sync, execution, and safety methods
that IBStreamingBot mixes in. Also provides connect helpers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ._telegram import safety_halt, shutdown
from .config import TRADING_CONFIG
from .hippocampus import Hippocampus
from .ib_executor import IBExecutor
from .ib_streamer import IBStreamer
from .memory import Journal
from .monitor.sleep_manager import SleepManager

log = logging.getLogger(__name__)
_SLEEP_MGR = SleepManager()


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
            self.executor.sync_from_ib(self.streamer)
            self._check_safety(pnl)
            self._sync_subs()
            # Manual flatten request from webapp
            if self._check_manual_flatten():
                self._finish_cycle([], [], poll, started, pnl, False)
                return
            # EOD flatten: force-close all positions in last N minutes of RTH
            if self._check_eod_flatten():
                self._finish_cycle([], [], poll, started, pnl, False)
                return
            positions = set(self.hippocampus._open_positions.keys())
            market_open = _SLEEP_MGR.get_state().active
            exit_s, decisions = self.juli.tick(
                positions, self._snapshot, self.streamer, self._closing
            )
            self._finish_cycle(exit_s, decisions, poll, started, pnl, market_open)
        except Exception as e:
            log.error("Cycle error: %s", e, exc_info=True)

    def _finish_cycle(
        self,
        exit_s: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        poll: float,
        started: float,
        pnl: Any,
        market_open: bool = True,
    ) -> None:
        """Process tickers, exits, decisions, reflect, wait."""
        self._last_bars = sum(
            1
            for tk in self.ib.pendingTickers()
            if self.streamer.update_bar(tk.contract.symbol if tk.contract else "")
        )
        for es in exit_s:
            t = es["ticker"]
            if t not in self._closing:
                self._closing.add(t)
                self.executor.close_position(t, self.streamer)
                log.info("EXIT %s: %s", t, es.get("reason", ""))
        for dec in decisions:
            if market_open and self._can_trade(dec):
                self._exec_decision(dec)
        self._reflect_closed()
        if pnl is not None:
            self.hippocampus._daily_pnl = float(pnl.dailyPnL)
        elapsed = time.monotonic() - started
        if elapsed < poll:
            time.sleep(poll - elapsed)
        self._heartbeat()
        log.info(
            "CYCLE bars=%d open=%d d=%d x=%d",
            self._last_bars,
            len(self.hippocampus._open_positions),
            len(decisions),
            len(exit_s),
        )

    def _sync_subs(self) -> None:
        """Sync subscriptions with scanner and open positions."""
        tracked = self.juli.budget.get_all_tracked()
        scanner = {c.symbol for c in self.juli._candidates[:20]}
        needed = tracked | scanner | set(self.hippocampus._open_positions.keys())
        self.executor.tracked_tickers = tracked
        for s in [s for s in needed if s not in self.streamer.ticker_subs][:5]:
            try:
                self.streamer.subscribe(s)
                self.streamer.seed_history(s)
            except Exception as e:
                log.warning("Sub %s fail: %s", s, e)

    def _check_manual_flatten(self) -> bool:
        """Check if webapp requested a manual flatten."""
        from .telemetry import _FLATTEN_REQUESTED

        if not _FLATTEN_REQUESTED:
            return False
        _FLATTEN_REQUESTED.clear()
        pos_count = len(self.hippocampus._open_positions)
        if pos_count == 0:
            log.info("MANUAL FLATTEN: no positions to flatten")
            return False
        log.warning("MANUAL FLATTEN: closing %d positions", pos_count)
        closed = self.executor.close_all_positions(self.streamer)
        log.warning("MANUAL FLATTEN: sent orders for %d positions", closed)
        return True

    def _check_eod_flatten(self) -> bool:
        """If EOD window, Juli tries to flatten. If she fails, we force it."""
        from .config import TRADING_CONFIG

        if not TRADING_CONFIG.eod_flatten_enabled:
            return False
        if not _SLEEP_MGR.is_eod_window(TRADING_CONFIG.eod_flatten_minutes):
            return False
        remaining = _SLEEP_MGR.minutes_to_close()
        pos_count = len(self.hippocampus._open_positions)
        if pos_count == 0:
            log.info("EOD: no positions, all flat (%.1f min to close)", remaining)
            return False
        # Juli already closed some — check if any remain
        log.warning(
            "EOD FLATTEN: %.1f min to close, %d positions remain — forcing market close",
            remaining,
            pos_count,
        )
        closed = self.executor.close_all_positions(self.streamer)
        log.warning("EOD FLATTEN: sent market orders for %d positions", closed)
        return True

    def _can_trade(self, dec: dict[str, Any]) -> bool:
        """Check session and direction config before trading."""
        # Check direction mode
        side = dec.get("thought", {}).get("verdict", "BUY")
        if not TRADING_CONFIG.is_direction_allowed(side):
            log.debug(
                "SKIP %s direction=%s mode=%s",
                dec["ticker"],
                side,
                TRADING_CONFIG.direction_mode,
            )
            return False
        # Check session (sleep_manager already gates overall, but double-check)
        state = _SLEEP_MGR.get_state()
        if not TRADING_CONFIG.is_session_active(state.session):
            log.debug("SKIP %s session=%s disabled", dec["ticker"], state.session)
            return False
        return True

    def _exec_decision(self, dec: dict[str, Any]) -> None:
        """Execute an entry decision through IB."""
        tk = self.streamer.ticker_subs.get(dec["ticker"])
        if tk is None or not tk.hasBidAsk:
            return
        t = dec["ticker"]
        if not self.hippocampus.check_entry_allowed():
            log.info("SKIP %s safety", t)
            return
        if t in self.hippocampus._open_positions:
            log.info("SKIP %s open", t)
            return
        sizing = dec.get("sizing")
        if sizing is None or getattr(sizing, "shares", 0) <= 0:
            log.debug("SKIP %s sizing=0", t)
            return
        price = float((tk.bid + tk.ask) * 0.5)
        self.executor.place_bracket(t, dec["thought"], price, self.streamer)
        self.executor.last_thoughts[t] = dec["thought"]
        self.juli.brain.register_position(t, price)

    def _reflect_closed(self) -> None:
        """Route closed trades to neuromorphic brain for learning."""
        for trade in self.executor.get_newly_closed_trades():
            won = trade["pnl"] > 0
            self.juli.brain.on_trade_close(
                ticker=trade["ticker"],
                won=won,
                pnl_pct=trade["return_pct"],
                direction=trade["direction"],
            )
            self._closing.discard(trade["ticker"])
            log.info(
                "REFLECT %s %s pnl=%.4f",
                trade["ticker"],
                "WIN" if won else "LOSS",
                trade["pnl"],
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
        """Check safety nets before trading."""
        if not self.hippocampus.safety_enabled:
            return
        n = count_open_positions(self.ib, self.executor.tracked_tickers)
        if n > 3:
            log.critical("SAFETY: %d > 3", n)
            self._halt("too_many_positions")
            return
        if pnl is not None and float(pnl.dailyPnL) < -200.0:
            log.critical("SAFETY: P&L $%.2f", float(pnl.dailyPnL))
            self._halt("daily_loss_limit")
            return
        if self.hippocampus._consecutive_losses >= 3:
            self._halt("consecutive_losses")

    def _halt(self, reason: str) -> None:
        """Emergency halt — persist and shutdown."""
        self.journal.append({"event": "halt", "reason": reason, "ts": time.time()})
        safety_halt(reason)
        self._running = False

    def _cleanup(self, pnl: Any) -> None:
        """Shutdown all subsystems."""
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
