"""hanoon_prime.ib_adapter — IB Gateway streaming adapter.
IB is the SINGLE source of truth for everything.
"""
from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Any

from ._telegram import safety_halt, shutdown, startup
from .hippocampus import Hippocampus
from .ib_compat import _ib_available, ib
from .ib_executor import IBExecutor
from .ib_streamer import IBStreamer
from .immune import IB_CLIENT_ID, IB_HOST, IB_LIVE_PORT, IB_PAPER_PORT
from .juli import JuliBrain
from .memory import Journal

log = logging.getLogger(__name__)
MAX_RECONNECT, RECONNECT_DELAY = 5, 5
class SafetyNetStopped(Exception):
    pass
def _count_ib_positions(ib_client: Any, tracked: set[str]) -> int:
    try:
        return len([p for p in ib_client.positions() if p.contract.symbol in tracked])
    except Exception:
        return 0
def _try_connect(ib_client: Any, host: str, port: int, client_id: int) -> bool:
    try:
        ib_client.connect(host, port, clientId=client_id)
        return bool(ib_client.isConnected())
    except Exception as e:
        log.warning("Connection failed: %s", e)
        return False
class IBStreamingBot:
    """Live bot: IB Gateway stream -> JuliBrain -> bracket orders."""
    def __init__(self, account: str = "PAPER") -> None:
        if not _ib_available:
            raise ImportError("ib_insync required: pip install ib_insync")
        self.ib: Any = ib.IB()
        self.account = account
        self.brain = Hippocampus(safety_enabled=False)
        self.juli = JuliBrain(self.ib)
        repo_root = Path(__file__).resolve().parents[2]
        self.journal = Journal(repo_root / "runtime" / "journal_live.jsonl")
        self.streamer = IBStreamer(self.ib)
        self.executor = IBExecutor(self.ib, self.brain, self.journal)
        self._running, self._last_beat, self._closing = False, 0.0, set()
        self._setup_signals()
    def _setup_signals(self) -> None:
        def _h(s: int, _: Any) -> None:
            log.warning("Signal %s — shutting down.", s); self._running = False
        signal.signal(signal.SIGINT, _h); signal.signal(signal.SIGTERM, _h)
    def connect(self, host: str = IB_HOST, port: int = IB_PAPER_PORT,
                client_id: int = IB_CLIENT_ID) -> None:
        """Connect to IB Gateway with retry logic."""
        for attempt in range(1, MAX_RECONNECT + 1):
            log.info("Connecting %s:%s (client=%s, attempt %d)", host, port, client_id, attempt)
            if _try_connect(self.ib, host, port, client_id):
                self.ib.execDetailsEvent += self.streamer.record_execution
                self.ib.commissionReportEvent += self.streamer.record_commission
                log.info("Connected. Account: %s", self.account)
                return
            if attempt < MAX_RECONNECT:
                time.sleep(RECONNECT_DELAY)
        raise ConnectionError(f"Failed after {MAX_RECONNECT} attempts")
    def run(self, tickers: list[str] | None = None, poll_interval: float = 1.0) -> None:
        """Run with optional seed tickers. JuliBrain discovers the rest."""
        self._running = True
        seed = tickers or []
        log.info("JULI Prime starting (seed=%s)", seed)
        startup(seed)
        self.executor.tracked_tickers = set(seed)
        for t in seed:
            self.streamer.subscribe(t)
            self.streamer.seed_history(t)
        pnl = self._start_pnl_stream()
        log.info("All streams active. Entering event loop...")
        while self._running:
            self._process_cycle(seed, poll_interval, pnl)
        self._cleanup(pnl)
    def _start_pnl_stream(self) -> Any:
        try:
            if self.account == "PAPER": self.account = self.ib.managedAccounts()[0]
            return self.ib.reqPnL(self.account, "")
        except Exception as e:
            log.error("PnL subscription failed: %s", e)
            return None
    def _get_snapshot(self, sym: str) -> dict[str, Any] | None:
        """Market data snapshot — partial if not enough bars yet."""
        tk = self.streamer.ticker_subs.get(sym)
        if tk is None or not tk.hasBidAsk():
            return None
        base: dict[str, Any] = {"bid": float(tk.bid), "ask": float(tk.ask),
                "last": float(tk.last or tk.close or 0),
                "volume": float(tk.volume or 0), "daily_volume": float(tk.volume or 0)}
        if not self.streamer.ready(sym):
            return base
        a = self.streamer.get_arrays(sym)
        base["atr"] = self.streamer.buffer_atr(sym)
        base["prices"] = list(a["close"])
        base["volumes"] = list(a["volume"])
        base["close_arr"] = a["close"]
        base["vol_arr"] = a["volume"]
        base["high_arr"] = a["high"]
        base["low_arr"] = a["low"]
        base["buy_vol_arr"] = a["buy_volume"]
        base["bid_sizes"] = a["bid_sizes"]
        base["ask_sizes"] = a["ask_sizes"]
        return base
    def _process_cycle(self, _seed: list[str], poll_interval: float, pnl: Any) -> None:
        started = time.monotonic()
        try:
            self.executor.sync_from_ib(self.streamer)
            self._check_safety_nets(pnl)
            self._sync_subscriptions()
            positions = set(self.brain._open_positions.keys())
            decisions, exit_signals = self.juli.tick(positions, self._get_snapshot, self.streamer)
            bars = 0
            for tk in self.ib.pendingTickers():
                s = tk.contract.symbol if tk.contract else ""
                if self.streamer.update_bar(s):
                    bars += 1
            for exit_sig in exit_signals:
                t = exit_sig["ticker"]
                if t not in self._closing:
                    self._closing.add(t)
                    self.executor.close_position(t, self.streamer)
                    log.info("BRAIN EXIT %s: %s", t, exit_sig.get("reason", ""))
            for dec in decisions:
                self._execute_decision(dec)
            self._process_closed_trades()
            if pnl is not None:
                self.brain._daily_pnl = float(pnl.dailyPnL)
            self.ib.waitOnUpdate(timeout=poll_interval)
            if time.monotonic() - started < poll_interval:
                time.sleep(poll_interval - (time.monotonic() - started))
            self._heartbeat()
            log.info("CYCLE bars=%d open=%d decisions=%d exits=%d",
                     bars, len(self.brain._open_positions), len(decisions), len(exit_signals))
        except Exception as e:
            log.error("Cycle error: %s", e, exc_info=True)
    def _sync_subscriptions(self) -> None:
        tracked = self.juli.budget.get_all_tracked()
        scanner_syms = {c.symbol for c in self.juli._candidates[:20]}
        all_needed = tracked | scanner_syms | set(self.brain._open_positions.keys())
        self.executor.tracked_tickers = tracked
        for sym in [s for s in all_needed if s not in self.streamer.ticker_subs][:5]:
            try: self.streamer.subscribe(sym); self.streamer.seed_history(sym)
            except Exception as e: log.warning("Sub %s fail: %s", sym, e)
    def _execute_decision(self, dec: dict[str, Any]) -> None:
        ticker, thought = dec["ticker"], dec["thought"]
        tk = self.streamer.ticker_subs.get(ticker)
        if tk is None or not tk.hasBidAsk():
            return
        if not self.brain.check_entry_allowed():
            log.info("SKIP %s safety blocked", ticker); return
        if ticker in self.brain._open_positions:
            log.info("SKIP %s already open", ticker); return
        price = float((tk.bid + tk.ask) * 0.5)
        self.executor.place_bracket(ticker, thought, price, self.streamer)
        self.executor.last_thoughts[ticker] = thought
        self.juli.brain.register_position(ticker, price)
    def _process_closed_trades(self) -> None:
        """Route closed trades to brain reflection."""
        for trade in self.executor.get_newly_closed_trades():
            won = trade["pnl"] > 0
            self.juli.on_trade_close(ticker=trade["ticker"], won=won,
                pnl_pct=trade["return_pct"], direction=trade["direction"])
            self._closing.discard(trade["ticker"])
            log.info("REFLECT %s %s pnl=%.4f", trade["ticker"], "WIN" if won else "LOSS", trade["pnl"])
    def _heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_beat < 60.0: return
        self._last_beat = now
        log.info("HEARTBEAT open=%d journal=%d", len(self.brain._open_positions), self.journal.count())
    def _check_safety_nets(self, pnl: Any) -> None:
        if not self.brain.safety_enabled: return
        count = _count_ib_positions(self.ib, self.executor.tracked_tickers)
        if count > 3: log.critical("SAFETY NET: %d positions > 3 max", count); self._halt_bot("too_many_positions"); return
        if pnl is not None and float(pnl.dailyPnL) < -200.0:
            log.critical("SAFETY NET: IB daily P&L $%.2f", float(pnl.dailyPnL)); self._halt_bot("daily_loss_limit"); return
        if self.brain._consecutive_losses >= 3: self._halt_bot("consecutive_losses")
    def _halt_bot(self, reason: str) -> None:
        self.journal.append({"event": "halt", "reason": reason, "ts": time.time()}); safety_halt(reason); self._running = False
    def _cleanup(self, pnl: Any) -> None:
        self.streamer.cancel_all(); self.executor.cancel_all(); self.juli.scanner.cancel_all()
        if pnl: self.ib.cancelPnL(self.account)
        if self.ib.isConnected(): self.ib.disconnect()
        shutdown()
    def run_paper(self, tickers: list[str] | None = None) -> None:
        """Connect to IB paper port and run."""
        self.connect(port=IB_PAPER_PORT); self.run(tickers)
    def run_live(self, tickers: list[str] | None = None) -> None:
        """Connect to IB live port and run."""
        self.connect(port=IB_LIVE_PORT); self.run(tickers)
__all__ = ["IBStreamingBot", "SafetyNetStopped"]
