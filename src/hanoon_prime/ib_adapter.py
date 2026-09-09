"""hanoon_prime.ib_adapter — IB Gateway streaming adapter.

IB is source of truth for: positions, P&L, fills, trade execution.
NeuromorphicBrain (via JuliBrain) is local source of truth for decisions.
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Any

from ._telegram import safety_halt, shutdown, startup
from ._telegram_chat import TelegramChat
from .brain.shared_state import BrainState
from .config import TRADING_CONFIG
from .hippocampus import Hippocampus
from .ib_compat import _ib_available, ib
from .ib_cycle import BotCycleMixin, SafetyNetStopped, try_connect
from .ib_executor import IBExecutor
from .ib_streamer import IBStreamer
from .immune import IB_CLIENT_ID, IB_HOST, IB_LIVE_PORT, IB_PAPER_PORT
from .juli import JuliBrain
from .memory import Journal
from .monitor.pipeline import PipelineMonitor

log = logging.getLogger(__name__)
MAX_RECONNECT, RECONNECT_DELAY = 5, 5


class IBStreamingBot(BotCycleMixin):
    """Live bot: IB Gateway stream -> NeuromorphicBrain -> bracket orders."""

    def __init__(self, account: str = "PAPER") -> None:
        if not _ib_available:
            raise ImportError("ib_insync required")
        self.ib: Any = ib.IB()
        self.account = account
        # Safety nets OFF by default — user activates via webapp telemetry.
        # When triggered, the bot blocks new entries but NEVER stops.
        self.hippocampus = Hippocampus(safety_enabled=False)
        self._halted: bool = False
        self.brain_state = BrainState()
        self.juli = JuliBrain(self.ib)
        repo_root = Path(__file__).resolve().parents[2]
        self.journal = Journal(repo_root / "runtime" / "journal_live.jsonl")
        consolidation = getattr(self.juli.brain, "_consolidation", None)
        if consolidation is not None:
            consolidation.safety.attach_journal(self.journal)
        self.streamer = IBStreamer(self.ib)
        self.executor = IBExecutor(self.ib, self.hippocampus, self.journal)
        self.executor.on_fill_confirmed = self._confirm_fill
        # Telegram chat (read-only queries answered from brain state)
        self._chat = TelegramChat(state_provider=self._chat_state)
        # Continuous pipeline health daemon (alerts + heal flags)
        self.monitor = PipelineMonitor(self, self.journal)
        self._running, self._last_beat = False, 0.0
        self._closing: set[str] = set()
        self._watched: set[str] = set()
        self._exit_reasons: dict[str, str] = {}
        self._hold_notified: dict[str, float] = {}
        self._last_bars: int = 0
        # Gateway supervision state (rebuild runner_gateway port)
        self._gw_was_connected: bool = True
        self._gw_attempts: int = 0
        self._order_placed_ts: dict[int, float] = {}
        self._setup_signals()

    def _setup_signals(self) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""

        def _h(s: int, _: Any) -> None:
            log.warning("Signal %s", s)
            self._running = False

        signal.signal(signal.SIGINT, _h)
        signal.signal(signal.SIGTERM, _h)

    def connect(
        self,
        host: str = IB_HOST,
        port: int = IB_PAPER_PORT,
        client_id: int = IB_CLIENT_ID,
    ) -> None:
        """Connect to IB Gateway with retry logic."""
        for attempt in range(1, MAX_RECONNECT + 1):
            log.info("Connect %s:%s (attempt %d)", host, port, attempt)
            if try_connect(self.ib, host, port, client_id):
                self.ib.execDetailsEvent += self.streamer.record_execution
                self.ib.commissionReportEvent += self.streamer.record_commission
                log.info("Connected. Account: %s", self.account)
                return
            if attempt < MAX_RECONNECT:
                time.sleep(RECONNECT_DELAY)
        raise ConnectionError(f"Failed after {MAX_RECONNECT} attempts")

    def run(self, tickers: list[str] | None = None, poll: float = 1.0) -> None:
        """Run with optional seed tickers."""
        self._running = True
        seed = tickers or []
        log.info("Starting (seed=%s)", seed)
        startup(seed or None)
        self.executor.tracked_tickers = set(seed)
        for t in seed:
            self.streamer.subscribe(t)
            self.streamer.seed_history(t)
        pnl = self._start_pnl()
        self._pnl_ref: Any = pnl
        self._chat.start()
        self.monitor.start()
        log.info("All streams active. Entering event loop...")
        while self._running:
            self._cycle(poll, pnl)
        self._cleanup(pnl)

    def _start_pnl(self) -> Any:
        """Request P&L stream from IB."""
        try:
            if self.account == "PAPER":
                self.account = self.ib.managedAccounts()[0]
            return self.ib.reqPnL(self.account, "")
        except Exception as e:
            log.error("PnL failed: %s", e)
            return None

    def run_paper(self, tickers: list[str] | None = None) -> None:
        """Connect to IB paper port and run."""
        self.connect(port=IB_PAPER_PORT)
        self.run(tickers)

    def run_live(self, tickers: list[str] | None = None) -> None:
        """Connect to IB live port and run."""
        self.connect(port=IB_LIVE_PORT)
        self.run(tickers)

    def _chat_state(self) -> dict[str, Any]:
        """Read-only snapshot for the Telegram chat (brain + IB state)."""
        try:
            snap = self.juli.brain.snapshot()
            mem = snap.get("memory", {}) or {}  # array-safe: dict-typed
            positions: dict[str, Any] = {}
            for t, pos in self.hippocampus._open_positions.items():
                positions[t] = {
                    "direction": getattr(pos, "direction", 1),
                    "shares": getattr(pos, "shares", 0),
                    "entry": getattr(pos, "entry_price", 0.0),
                    "horizon": self.executor._horizons.get(t, "scalp"),
                }
            pnl_obj = getattr(self, "_pnl_ref", None)
            return {
                "regime": snap.get("brain_state", {}).get("regime_label", "?"),
                "threshold": snap.get("threshold", 0.0),
                "decisions": snap.get("decision_count", 0),
                "open_positions": len(positions),
                "positions": positions,
                "horizons": sorted(TRADING_CONFIG.horizons),
                "daily_pnl": float(getattr(pnl_obj, "dailyPnL", 0.0) or 0.0),
                "wins": (
                    int(mem.get("recent_wr_n", 0) * mem.get("recent_wr", 0.0))
                    if isinstance(mem, dict)
                    else 0
                ),
                "losses": (
                    int(mem.get("recent_wr_n", 0) * (1 - mem.get("recent_wr", 0.0)))
                    if isinstance(mem, dict)
                    else 0
                ),
            }
        except Exception as exc:
            log.debug("Chat state failed: %s", exc)
            return {}


__all__ = ["IBStreamingBot", "SafetyNetStopped"]
