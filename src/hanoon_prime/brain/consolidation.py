"""hanoon_prime.brain.consolidation — System 2: Background consolidation.

Runs asynchronously in a background thread every 10-30s.
Handles network I/O (HALIM, regime), heavy compute (thinker), disk
persistence, and sleep replay. IB feeds trade data; ConsolidationEngine
provides the slow path for state updates and offline memory consolidation.
"""
from __future__ import annotations
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional
from ..reflection.buffer import Fill, Trade, TradeBuffer
from ..reflection.supervisor import LearningSupervisor
from .halim_adapter import HalimAdapter
from .memory import JuliMemory
from .neurons.sleep import SleepReplayEngine, SleepResult
from .shared_state import BrainState
from .thinker import Thinker

log = logging.getLogger(__name__)
HALIM_URL: str = "http://127.0.0.1:8765"
CYCLE_INTERVAL: float = 30.0


class ConsolidationEngine:
    """System 2: Background cognitive engine — runs every 10-30s."""

    def __init__(
        self, brain_state: BrainState, halim_url: str = HALIM_URL,
        interval: float = CYCLE_INTERVAL, sleep_engine: Optional[SleepReplayEngine] = None,
    ) -> None:
        self.state = brain_state
        self.interval = interval
        self.memory = JuliMemory()
        self.halim = HalimAdapter(base_url=halim_url)
        self.thinker = Thinker()
        from ..reflection.buffer import BUY, SELL
        self.buffer = TradeBuffer(on_trade_closed=self._on_trade_closed)
        self.supervisor = LearningSupervisor(self.buffer, self.memory)
        self._sleep_engine = sleep_engine
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start System 2 background loop."""
        self._running = True
        self._sync_initial_state()
        self.supervisor.start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="consolidation")
        self._thread.start()
        log.info("System 2 started (interval=%.0fs)", self.interval)

    def stop(self) -> None:
        """Stop System 2 background loop."""
        self._running = False
        if self._thread: self._thread.join(timeout=5.0)
        self._persist_state()
        log.info("System 2 stopped")

    def _sync_initial_state(self) -> None:
        """Push initial state to shared dict on startup."""
        weights = self.memory.get_weights()
        self.state.update(threshold=self.memory.threshold, indicator_weights=weights)

    def _loop(self) -> None:
        """Main background loop — runs every interval seconds."""
        while self._running:
            try: self._cycle()
            except Exception as e: log.error("System 2 cycle error: %s", e)
            time.sleep(self.interval)

    def _cycle(self) -> None:
        """One full System 2 cognitive cycle."""
        self._update_regime()
        self._update_halim()
        self._run_thinker()
        self._persist_state()
        log.info("S2 | regime=%.2f halim=%.3f thinker=%.4f",
                 self.state.get("regime_multiplier", 1.0),
                 self.state.get("halim_modifier", 0.0),
                 self.state.get("thinker_modifier", 0.0))

    def _update_regime(self) -> None:
        """Get regime classification from HALIM."""
        alpha = self._get_latest_alpha()
        if not alpha: return
        try:
            regime = self.halim.get_regime(alpha, self.state.get_latest_prices())
        except Exception as e:
            log.warning("Regime query failed: %s", e)
            return
        if not isinstance(regime, dict): return
        self._apply_regime(regime)

    def _apply_regime(self, regime: dict[str, Any]) -> None:
        """Apply regime data to shared state."""
        self.state.update(
            regime_multiplier=regime.get("multiplier", 1.0),
            regime_label=regime.get("regime", "normal"),
            regime_confidence=regime.get("confidence", 0.5),
            regime_risk=regime.get("risk_adjustment", "normal"),
            regime_drivers=regime.get("key_drivers", []),
            regime_description=regime.get("description", ""),
        )

    def _update_halim(self) -> None:
        """Poll HALIM external AI advisor (network I/O)."""
        alpha = self._get_latest_alpha()
        if not alpha: return
        ticker = max(alpha, key=lambda k: abs(alpha.get(k, 0)))
        mod = self.halim.get_modifier(ticker, alpha, 0.0, "SCAN")
        self.state.update(halim_modifier=mod)

    def _run_thinker(self) -> None:
        """Run thinker deliberation and write to shared state."""
        alpha = self._get_latest_alpha()
        if not alpha: return
        prices = self.state.get_latest_prices()
        regime = self.state.get("regime_label", "unknown")
        threshold = self.state.get("threshold", 0.58)
        r = self.thinker.think(alpha, 0.0, 1, regime, prices, prices, prices, threshold)
        self.state.update(thinker_modifier=r.modifier, thinker_confidence_mod=r.confidence_mod,
                          thinker_risk_scalar=r.risk_scalar)

    def _persist_state(self) -> None:
        """Atomic write to state.json (disk I/O)."""
        try:
            state_dir = Path(__file__).resolve().parents[2] / "runtime"
            state_dir.mkdir(parents=True, exist_ok=True)
            tmp = state_dir / "state.json.tmp"
            tmp.write_text(json.dumps(self._persist_data(), default=str))
            tmp.replace(state_dir / "state.json")
        except Exception as e:
            log.warning("State persist failed: %s", e)

    def _persist_data(self) -> dict[str, Any]:
        """Build state data for persistence."""
        ep = getattr(getattr(self.thinker, "episodic", None), "size", 0)
        return {"weights": self.memory.get_weights(), "threshold": self.memory.threshold,
                "pred_error": self.memory.pred_error, "episodic_size": ep,
                "brain_state": self.state.snapshot(), "timestamp": time.time()}

    def _get_latest_alpha(self) -> dict[str, float] | None:
        """Read latest alpha from System 1 via shared state."""
        return self.state.get_latest_alpha()

    def on_trade_close(self, ticker: str, won: bool, pnl_pct: float, direction: int = 1,
                       qty: float = 1.0, avg_price: float = 0.0, fees: float = 0.0) -> None:
        """Route trade close to thinker + buffer (slow path learning)."""
        from ..reflection.buffer import BUY, SELL
        alpha = self._get_latest_alpha() or {}
        self.thinker.episodic.add(alpha, pnl_pct)
        self.thinker.emotion.update(won, pnl_pct)
        self.state.set_refractory(2.0)
        side = BUY if direction > 0 else SELL
        self.buffer.on_fill(Fill(ticker=ticker, side=side, qty=qty, price=avg_price,
                                 time=time.time(), commission=fees))

    def _on_trade_closed(self, trade: Trade) -> None:
        """Callback from TradeBuffer when a round-trip closes."""
        self.supervisor.on_trade_close(trade)
        log.info("Buffer trade closed: %s pnl=%.2f", trade.ticker, trade.pnl)

    def run_sleep_replay(self) -> Optional[SleepResult]:
        """Run sleep consolidation cycle for offline learning."""
        if self._sleep_engine is None: return None
        return self._sleep_engine.run_cycle()

    @property
    def sleep_engine(self) -> Optional[SleepReplayEngine]:
        """Access sleep replay engine."""
        return self._sleep_engine


__all__ = ["ConsolidationEngine"]