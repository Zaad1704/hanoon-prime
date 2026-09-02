"""hanoon_prime.ib_adapter — thin adapter from IB data to JULI's numpy arrays.

Architecture: IB → JULI → Entry/Hold/Exit + Learning.

This is the ONLY production entry point. It receives data from IB,
feeds it to the Brain, and executes verdicts. No business logic here —
just conversion and dispatch.

Requires ib_insync (installed with `pip install -e ".[ib]"`):
    pip install -e ".[ib]"

Usage:
    from hanoon_prime.ib_adapter import IBPaperTradingBot
    bot = IBPaperTradingBot("PAPER")
    bot.run_forever(tickers=["SPY", "AAPL"])
"""
from __future__ import annotations

import signal
import sys
import time
from typing import Optional

from .brain import Brain, Thought
from .journal import Journal
from ._execution import ExecutionManager

# ib_insync is optional — only needed for live IB connection.
_ib_insync_available = False
try:
    import ib_insync  # type: ignore
    _ib_insync_available = True
except ImportError:
    pass


class SafetyNetStopped(Exception):
    """Raised when a hard safety net triggers. Halts the bot immediately."""
    pass


class IBPaperTradingBot:
    """Paper/paper-trading bot: IB → Brain → Verdict → Execution."""

    def __init__(self, account: str = "PAPER", data_dir: Optional[str] = None) -> None:
        self.ib = ib_insync.IB() if _ib_insync_available else None  # type: ignore
        self.account = account
        self.data_dir = data_dir
        self.brain = Brain()
        self.journal = Journal("runtime/journal_prime.jsonl")
        self.exec_mgr = ExecutionManager(self.brain, self.journal)
        self._running = False

        if data_dir:
            return  # backtest mode — no signal handlers needed
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:
        print(f"\nReceived signal {signum}. Safe shutdown...")
        self._running = False

    def connect(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> None:
        """Connect to IB Gateway or TWS."""
        if self.data_dir or self.ib is None:
            return
        self.ib.connect(host, port, clientId=client_id)  # type: ignore

    def _get_ohlcv_arrays(self, ticker: str, lookback: int = 30):
        """Pull 1-min OHLCV from IB and convert to numpy arrays."""
        import numpy as np
        bars = self.ib.reqHistoricalData(  # type: ignore
            ticker, endDateTime="", durationStr="1 D",
            barSizeSetting="1 min", whatToShow="TRADES",
            useRTH=True, formatDate=1,
        )
        if not bars:
            return None
        arr = lambda attr: np.array([getattr(b, attr) for b in bars][-lookback:])
        return arr("close"), arr("high"), arr("low"), arr("volume")

    def _compute_buy_volume(self, close, high, low, volume):
        """Estimate buy/sell volume split from OHLCV."""
        import numpy as np
        rng = np.maximum(high - low, 1e-12)
        buy_frac = np.clip((close - low) / rng, 0.0, 1.0)
        return volume * buy_frac

    def evaluate_candidate(self, ticker: str) -> Optional[Thought]:
        """Evaluate one ticker through the full JULI pipeline."""
        import numpy as np
        data = self._get_ohlcv_arrays(ticker)
        if data is None:
            return None

        close, high, low, volume = data
        buy_vol = self._compute_buy_volume(close, high, low, volume)
        sell_vol = np.maximum(volume - buy_vol, 0)
        avg_vol = float(np.mean(volume)) if len(volume) > 0 else 1.0

        bid_size = float(np.mean(buy_vol[-10:])) / max(avg_vol, 1e-12)
        ask_size = float(np.mean(sell_vol[-10:])) / max(avg_vol, 1e-12)

        return self.brain.deliberate_entry(
            close=close.tolist(), volume=volume.tolist(),
            buy_volume=buy_vol.tolist(),
            bid_sizes=np.array([bid_size]), ask_sizes=np.array([ask_size]),
        )

    def _evaluate_with_data(self, ticker: str, close, high, low, volume):
        """Evaluate a candidate from pre-loaded arrays (backtest adapter mode)."""
        from .data import compute_buy_volume
        import numpy as np
        buy_vol = compute_buy_volume(close, high, low, volume)
        sell_vol = np.maximum(volume - buy_vol, 0)
        avg_vol = float(np.mean(volume)) if len(volume) > 0 else 1.0
        bid_size = float(np.mean(buy_vol[-10:])) / max(avg_vol, 1e-12)
        ask_size = float(np.mean(sell_vol[-10:])) / max(avg_vol, 1e-12)
        return self.brain.deliberate_entry(
            close=close.tolist(), volume=volume.tolist(),
            buy_volume=buy_vol.tolist(),
            bid_sizes=np.array([bid_size]), ask_sizes=np.array([ask_size]),
        )

    def _evaluate_and_execute(self, ticker: str) -> None:
        """Evaluate one ticker and execute if verdict is ENTER."""
        thought = self.evaluate_candidate(ticker)
        if not thought or thought.verdict != "ENTER":
            return
        data = self._get_ohlcv_arrays(ticker, lookback=1)
        if data:
            self.exec_mgr.execute_entry(ticker, thought, float(data[0][-1]))

    def _run_tickers(self, tickers: list[str]) -> None:
        """Evaluate all tickers in one pass."""
        for ticker in tickers:
            if not self._running:
                return
            try:
                self._evaluate_and_execute(ticker)
            except Exception as exc:
                self.journal.append({"event": "evaluation_error",
                                     "ticker": ticker, "error": str(exc)})

    def run_forever(self, tickers: list[str], poll_interval: float = 1.0) -> None:
        """Main loop: poll IB, evaluate candidates, execute verdicts."""
        self._running = True
        print(f"JULI Prime — monitoring {tickers}")

        while self._running:
            try:
                self.brain.check_safety_nets()
            except RuntimeError as e:
                print(f"SAFETY NET: {e}")
                self.journal.append({"event": "safety_net_triggered", "reason": str(e)})
                break

            self._run_tickers(tickers)
            self.exec_mgr.check_exits(self._get_ohlcv_arrays)
            time.sleep(poll_interval)

        print("JULI Prime stopped.")


__all__ = ["IBPaperTradingBot", "SafetyNetStopped"]
