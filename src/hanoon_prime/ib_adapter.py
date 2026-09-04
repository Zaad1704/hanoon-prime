"""hanoon_prime.ib_adapter — IB Gateway streaming adapter."""
from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Any

from ._telegram import safety_halt, shutdown, startup
from .brain.shared_state import BrainState
from .brain.slow_cortex import SlowCortex
from .hippocampus import Hippocampus
from .ib_compat import _ib_available, ib
from .ib_executor import IBExecutor
from .ib_streamer import IBStreamer
from .immune import IB_CLIENT_ID, IB_HOST, IB_LIVE_PORT, IB_PAPER_PORT
from .juli import JuliBrain
from .memory import Journal

log = logging.getLogger(__name__)
MAX_RECONNECT, RECONNECT_DELAY = 5, 5

class SafetyNetStopped(Exception): pass

def _count_pos(ib_client: Any, tracked: set[str]) -> int:
    try: return len([p for p in ib_client.positions() if p.contract.symbol in tracked])
    except Exception: return 0

def _try_connect(ib_client: Any, host: str, port: int, cid: int) -> bool:
    try:
        ib_client.connect(host, port, clientId=cid)
        return bool(ib_client.isConnected())
    except Exception as e:
        log.warning("Connect failed: %s", e)
        return False

class IBStreamingBot:
    """Live bot: IB Gateway stream -> JuliBrain -> bracket orders."""

    def __init__(self, account: str = "PAPER") -> None:
        if not _ib_available:
            raise ImportError("ib_insync required")
        self.ib: Any = ib.IB()
        self.account = account
        self.brain = Hippocampus(safety_enabled=False)
        self.juli = JuliBrain(self.ib)
        self.brain_state = BrainState()
        self.slow_cortex = SlowCortex(self.brain_state)
        repo_root = Path(__file__).resolve().parents[2]
        self.journal = Journal(repo_root / "runtime" / "journal_live.jsonl")
        self.streamer = IBStreamer(self.ib)
        self.executor = IBExecutor(self.ib, self.brain, self.journal)
        self._running, self._last_beat = False, 0.0
        self._closing: set[str] = set()
        self._setup_signals()
    def _setup_signals(self) -> None:
        def _h(s: int, _: Any) -> None:
            log.warning("Signal %s", s); self._running = False
        signal.signal(signal.SIGINT, _h); signal.signal(signal.SIGTERM, _h)
    def connect(self, host: str = IB_HOST, port: int = IB_PAPER_PORT,
                client_id: int = IB_CLIENT_ID) -> None:
        """Connect to IB Gateway with retry logic."""
        for attempt in range(1, MAX_RECONNECT + 1):
            log.info("Connect %s:%s (attempt %d)", host, port, attempt)
            if _try_connect(self.ib, host, port, client_id):
                self.ib.execDetailsEvent += self.streamer.record_execution
                self.ib.commissionReportEvent += self.streamer.record_commission
                log.info("Connected. Account: %s", self.account); return
            if attempt < MAX_RECONNECT: time.sleep(RECONNECT_DELAY)
        raise ConnectionError(f"Failed after {MAX_RECONNECT} attempts")
    def run(self, tickers: list[str] | None = None, poll: float = 1.0) -> None:
        """Run with optional seed tickers."""
        self._running = True; seed = tickers or []
        log.info("Starting (seed=%s)", seed)
        self.slow_cortex.start(); startup(seed or None)
        self.executor.tracked_tickers = set(seed)
        for t in seed:
            self.streamer.subscribe(t); self.streamer.seed_history(t)
        pnl = self._start_pnl()
        log.info("All streams active. Entering event loop...")
        while self._running:
            self._cycle(seed, poll, pnl)
        self._cleanup(pnl)
    def _start_pnl(self) -> Any:
        try:
            if self.account == "PAPER":
                self.account = self.ib.managedAccounts()[0]
            return self.ib.reqPnL(self.account, "")
        except Exception as e:
            log.error("PnL failed: %s", e); return None

    def _snapshot(self, sym: str) -> dict[str, Any] | None:
        tk = self.streamer.ticker_subs.get(sym)
        if tk is None or not tk.hasBidAsk(): return None
        base: dict[str, Any] = {"bid": float(tk.bid), "ask": float(tk.ask),
            "last": float(tk.last or tk.close or 0),
            "volume": float(tk.volume or 0), "daily_volume": float(tk.volume or 0)}
        if not self.streamer.ready(sym): return base
        a = self.streamer.get_arrays(sym)
        base["atr"] = self.streamer.buffer_atr(sym)
        for k in ("close","volume","high","low","buy_volume","bid_sizes","ask_sizes"):
            base[f"{k}_arr"] = a[k]
        base["prices"], base["volumes"] = list(a["close"]), list(a["volume"])
        return base

    def _cycle(self, _seed: list[str], poll: float, pnl: Any) -> None:
        started = time.monotonic()
        try:
            self.executor.sync_from_ib(self.streamer)
            self._check_safety(pnl); self._sync_subs()
            positions = set(self.brain._open_positions.keys())
            exit_s, decisions = self.juli.tick(
                positions, self._snapshot, self.streamer, self._closing)
            bars = sum(1 for tk in self.ib.pendingTickers()
                       if self.streamer.update_bar(
                           tk.contract.symbol if tk.contract else ""))
            for es in exit_s:
                t = es["ticker"]
                if t not in self._closing:
                    self._closing.add(t)
                    self.executor.close_position(t, self.streamer)
                    log.info("EXIT %s: %s", t, es.get("reason", ""))
            for dec in decisions: self._exec_decision(dec)
            self._reflect_closed()
            if pnl is not None: self.brain._daily_pnl = float(pnl.dailyPnL)
            self.ib.waitOnUpdate(timeout=poll)
            elapsed = time.monotonic() - started
            if elapsed < poll: time.sleep(poll - elapsed)
            self._heartbeat()
            n = len(self.brain._open_positions)
            log.info("CYCLE bars=%d open=%d d=%d x=%d", bars, n, len(decisions), len(exit_s))
        except Exception as e:
            log.error("Cycle error: %s", e, exc_info=True)
    def _sync_subs(self) -> None:
        tracked = self.juli.budget.get_all_tracked()
        scanner = {c.symbol for c in self.juli._candidates[:20]}
        needed = tracked | scanner | set(self.brain._open_positions.keys())
        self.executor.tracked_tickers = tracked
        for s in [s for s in needed if s not in self.streamer.ticker_subs][:5]:
            try: self.streamer.subscribe(s); self.streamer.seed_history(s)
            except Exception as e: log.warning("Sub %s fail: %s", s, e)
    def _exec_decision(self, dec: dict[str, Any]) -> None:
        tk = self.streamer.ticker_subs.get(dec["ticker"])
        if tk is None or not tk.hasBidAsk(): return
        t = dec["ticker"]
        if not self.brain.check_entry_allowed(): log.info("SKIP %s safety", t); return
        if t in self.brain._open_positions: log.info("SKIP %s open", t); return
        price = float((tk.bid + tk.ask) * 0.5)
        self.executor.place_bracket(t, dec["thought"], price, self.streamer)
        self.executor.last_thoughts[t] = dec["thought"]
        self.juli.brain.register_position(t, price)
    def _reflect_closed(self) -> None:
        for trade in self.executor.get_newly_closed_trades():
            won = trade["pnl"] > 0
            self.slow_cortex.on_trade_close(
                ticker=trade["ticker"], won=won,
                pnl_pct=trade["return_pct"], direction=trade["direction"],
                qty=trade.get("qty", 1), avg_price=trade.get("avg_price", 0),
                fees=trade.get("commission", 0))
            self._closing.discard(trade["ticker"])
            log.info("REFLECT %s %s pnl=%.4f", trade["ticker"], "WIN" if won else "LOSS", trade["pnl"])
    def _heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_beat < 60.0: return
        self._last_beat = now
        log.info("HEARTBEAT open=%d journal=%d", len(self.brain._open_positions), self.journal.count())
    def _check_safety(self, pnl: Any) -> None:
        if not self.brain.safety_enabled: return
        n = _count_pos(self.ib, self.executor.tracked_tickers)
        if n > 3: log.critical("SAFETY: %d > 3", n); self._halt("too_many_positions"); return
        if pnl is not None and float(pnl.dailyPnL) < -200.0:
            log.critical("SAFETY: P&L $%.2f", float(pnl.dailyPnL))
            self._halt("daily_loss_limit"); return
        if self.brain._consecutive_losses >= 3: self._halt("consecutive_losses")
    def _halt(self, reason: str) -> None:
        self.journal.append({"event": "halt", "reason": reason, "ts": time.time()})
        safety_halt(reason); self._running = False

    def _cleanup(self, pnl: Any) -> None:
        self.slow_cortex.stop(); self.streamer.cancel_all()
        self.executor.cancel_all(); self.juli.scanner.cancel_all()
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