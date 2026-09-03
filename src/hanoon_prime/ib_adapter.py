"""hanoon_prime.ib_adapter — IB Gateway streaming adapter.
IB is the SINGLE source of truth for everything.
Journal is a carbon copy of IB state — not generated data.
Excluded from mypy strict — ib_insync is not fully typed.
"""
from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Any, Optional

from .cerebellum import compute_alpha
from .cortex import Thought
from .hippocampus import Hippocampus
from .ib_compat import _ib_available, ib
from .ib_executor import IBExecutor
from .ib_streamer import IBStreamer
from .immune import IB_CLIENT_ID, IB_HOST, IB_LIVE_PORT, IB_PAPER_PORT
from .memory import Journal

log = logging.getLogger(__name__)
MAX_RECONNECT = 5
RECONNECT_DELAY = 5
class SafetyNetStopped(Exception):
    """Raised when a hard safety net triggers."""
def _count_ib_positions(ib_client: Any, tracked: set[str]) -> int:
    """Count actual positions in IB for tracked tickers."""
    try:
        return len([p for p in ib_client.positions() if p.contract.symbol in tracked])
    except Exception:
        return 0
def _try_connect(ib_client: Any, host: str, port: int, client_id: int) -> bool:
    """Try to connect once. Returns True on success."""
    try:
        ib_client.connect(host, port, clientId=client_id)
        return bool(ib_client.isConnected())
    except Exception as e:
        log.warning("Connection failed: %s", e)
        return False
class IBStreamingBot:
    """Live bot: IB Gateway stream → brain → bracket orders."""
    def __init__(self, account: str = "PAPER") -> None:
        if not _ib_available:
            raise ImportError("ib_insync required: pip install ib_insync")
        self.ib: Any = ib.IB()
        self.account = account
        self.brain = Hippocampus(safety_enabled=False)
        repo_root = Path(__file__).resolve().parents[2]
        self.journal = Journal(repo_root / "runtime" / "journal_live.jsonl")
        self.streamer = IBStreamer(self.ib)
        self.executor = IBExecutor(self.ib, self.brain, self.journal)
        self._running = False
        self._last_beat: float = 0.0
        self._setup_signals()
    def _setup_signals(self) -> None:
        """Register SIGINT/SIGTERM for safe shutdown."""
        def _handler(signum: int, frame: Any) -> None:
            log.warning("Received signal %s. Initiating safe shutdown.", signum)
            self._running = False
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    def connect(self, host: str = IB_HOST, port: int = IB_PAPER_PORT,
                client_id: int = IB_CLIENT_ID) -> None:
        """Connect to IB Gateway with retry logic."""
        for attempt in range(1, MAX_RECONNECT + 1):
            log.info("Connecting to %s:%s (client=%s, attempt %d)",
                     host, port, client_id, attempt)
            if _try_connect(self.ib, host, port, client_id):
                self._subscribe_ib_events()
                log.info("Connected. Account: %s", self.account)
                return
            if attempt < MAX_RECONNECT:
                time.sleep(RECONNECT_DELAY)
        raise ConnectionError(f"Failed to connect after {MAX_RECONNECT} attempts")
    def _subscribe_ib_events(self) -> None:
        """Subscribe to IB events (executions, commissions)."""
        self.ib.execDetailsEvent += self.streamer.record_execution
        self.ib.commissionReportEvent += self.streamer.record_commission
    def _evaluate(self, ticker: str) -> Optional[Thought]:
        """Run Cortex on the latest StreamBuffer data."""
        if not self.streamer.ready(ticker):
            return None
        a = self.streamer.get_arrays(ticker)
        alpha = compute_alpha(close=a["close"], volume=a["volume"],
                              buy_volume=a["buy_volume"],
                              bid_sizes=a["bid_sizes"], ask_sizes=a["ask_sizes"])
        return self.brain.cortex.evaluate(alpha) if alpha else None
    def _on_entry(self, ticker: str, thought: Thought, tk: Any) -> None:
        """Handle BUY/SELL verdict: safety nets, then bracket entry."""
        if not self.brain.check_entry_allowed():
            log.info("SKIP %s safety_nets blocked", ticker)
            return
        if ticker in self.brain._open_positions:
            log.info("SKIP %s already open", ticker)
            return
        if not tk.hasBidAsk():
            log.info("SKIP %s no bid/ask", ticker)
            return
        price = float((tk.bid + tk.ask) * 0.5)
        self.executor.place_bracket(ticker, thought, price, self.streamer)
        self.executor.last_thoughts[ticker] = thought
    def run(self, tickers: list[str], poll_interval: float = 1.0) -> None:
        """Subscribe to IB streams, evaluate verdicts, place bracket orders."""
        self._running = True
        log.info("JULI Prime — live monitoring %s", tickers)
        self.executor.tracked_tickers = set(tickers)
        for t in tickers:
            self.streamer.subscribe(t)
            self.streamer.seed_history(t)
        pnl = self._start_pnl_stream()
        log.info("All streams active. Entering event loop...")
        while self._running:
            self._process_cycle(tickers, poll_interval, pnl)
        self._cleanup(pnl)
    def _start_pnl_stream(self) -> Any:
        """Start IB P&L streaming for safety net monitoring."""
        try:
            if self.account == "PAPER":
                self.account = self.ib.managedAccounts()[0]
            return self.ib.reqPnL(self.account, "")
        except Exception as e:
            log.error("PnL subscription failed: %s", e)
            return None
    def _process_cycle(self, tickers: list[str], poll_interval: float, pnl: Any) -> None:
        """One iteration — IB is source of truth for everything."""
        started = time.monotonic()
        self.executor.sync_from_ib(self.streamer)
        self._check_safety_nets(pnl)
        bars = 0
        for tk in self.ib.pendingTickers():
            s = tk.contract.symbol if tk.contract else ""
            if s in tickers and self.streamer.update_bar(s):
                self._handle_ticker(s, tk)
                bars += 1
        if pnl is not None:
            self.brain._daily_pnl = float(pnl.dailyPnL)
        self.ib.waitOnUpdate(timeout=poll_interval)
        if time.monotonic() - started < poll_interval:
            time.sleep(poll_interval - (time.monotonic() - started))
        self._heartbeat()
        if bars:
            log.info("CYCLE bars=%d open=%d", bars, len(self.brain._open_positions))
    def _heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_beat < 60.0:
            return
        self._last_beat = now
        log.info("HEARTBEAT open=%d journal=%d",
                 len(self.brain._open_positions), self.journal.count())
    def _check_safety_nets(self, pnl: Any) -> None:
        """Check safety nets against IB's actual state."""
        if not self.brain.safety_enabled:
            return
        count = _count_ib_positions(self.ib, self.executor.tracked_tickers)
        if count > 3:
            log.critical("SAFETY NET: %d positions > 3 max", count)
            self._halt_bot("too_many_positions")
            return
        if pnl is not None and float(pnl.dailyPnL) < -200.0:
            log.critical("SAFETY NET: IB daily P&L $%.2f", float(pnl.dailyPnL))
            self._halt_bot("daily_loss_limit")
            return
        if self.brain._consecutive_losses >= 3:
            self._halt_bot("consecutive_losses")
    def _halt_bot(self, reason: str) -> None:
        """Halt the bot and record to journal."""
        self.journal.append({"event": "safety_net_triggered", "source": "ib_gateway",
                             "reason": reason, "timestamp": time.time()})
        self._running = False
    def _handle_ticker(self, sym: str, tk: Any) -> None:
        """Evaluate ticker update and enter if verdict is non-HOLD."""
        thought = self._evaluate(sym)
        if not thought:
            return
        d = "+" if thought.direction > 0 else "-" if thought.direction < 0 else "0"
        log.info("THINK %s %s%s score=%.3f", sym, thought.verdict, d, thought.score)
        if thought.direction != 0 and thought.verdict != "HOLD":
            thought.z_scores = getattr(thought, "z_scores", {})
            self._on_entry(sym, thought, tk)
    def _cleanup(self, pnl: Any) -> None:
        """Cancel all IB subscriptions and disconnect safely."""
        log.info("Shutting down — cancelling all streams.")
        self.streamer.cancel_all()
        self.executor.cancel_all()
        if pnl is not None:
            self.ib.cancelPnL(self.account)
        if self.ib.isConnected():
            self.ib.disconnect()
        log.info("All IB subscriptions cancelled. JULI Prime stopped.")
    def run_paper(self, tickers: list[str]) -> None:
        """Connect to IB Gateway paper port (4002) and run."""
        self.connect(port=IB_PAPER_PORT)
        self.run(tickers)
    def run_live(self, tickers: list[str]) -> None:
        """Connect to IB Gateway live port (4001) and run."""
        self.connect(port=IB_LIVE_PORT)
        self.run(tickers)
__all__ = ["IBStreamingBot", "SafetyNetStopped"]
