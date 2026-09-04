"""hanoon_prime.brain.orchestrator — Neuromorphic Brain Coordinator.

IB feeds trade data; this brain makes ALL decisions.
Slow path: ConsolidationEngine for background work.
"""

from __future__ import annotations
import logging
from typing import Any, Optional
from ..cortex import Cortex, Thought
from ..hippocampus import Hippocampus
from .affective import Affective
from .consolidation import ConsolidationEngine
from .deliberation import Deliberator
from .dynamics import Dynamics
from .episodic import EpisodicMemory
from .exits import ExitPolicy, ExitSignal
from .memory import JuliMemory
from .neurons.bridge import NeuromorphicBridge
from .neurons.sleep import SleepReplayEngine, SleepResult
from .risk import RiskEngine, SizingResult
from .shared_state import BrainState

log = logging.getLogger(__name__)


class NeuromorphicBrain:
    """Neuromorphic brain — LOCAL SOURCE OF TRUTH for all decisions."""

    NEURO_BLEND: float = 0.3

    def __init__(
        self, brain_state: BrainState | None = None, enable_neuromorphic: bool = True
    ) -> None:
        self.state = brain_state or BrainState()
        self.memory = JuliMemory()
        self.episodic = EpisodicMemory()
        weights = self.memory.get_weights()
        self.cortex = Cortex(weights=weights)
        self.hippocampus = Hippocampus(cortex=self.cortex, safety_enabled=False)
        self.affective = Affective()
        self.deliberator = Deliberator(threshold=self.memory.threshold)
        self.dynamics = Dynamics(base_threshold=self.memory.threshold)
        self.risk = RiskEngine()
        self.exits = ExitPolicy()
        self._neuromorphic: Optional[NeuromorphicBridge] = None
        self._sleep_engine: Optional[SleepReplayEngine] = None
        self._consolidation: Optional[ConsolidationEngine] = None
        self._last_alpha: dict[str, dict[str, float]] = {}
        self._decision_count: int = 0
        if enable_neuromorphic:
            self._init_neuromorphic()

    def _init_neuromorphic(self) -> None:
        """Initialize neuromorphic bridge, sleep engine, and consolidation."""
        self._neuromorphic = NeuromorphicBridge()
        self._sleep_engine = SleepReplayEngine(
            network=self._neuromorphic._network,
            stdp=self._neuromorphic._stdp,
            memory=self._neuromorphic._memory,
        )
        self._consolidation = ConsolidationEngine(
            brain_state=self.state, sleep_engine=self._sleep_engine
        )

    def start(self) -> None:
        """Start the neuromorphic brain (includes slow path)."""
        if self._consolidation is not None:
            self._consolidation.start()
        log.info("NeuromorphicBrain started (neuro=%s)", self._neuromorphic is not None)

    def stop(self) -> None:
        """Stop background consolidation."""
        if self._consolidation is not None:
            self._consolidation.stop()

    def tick(
        self,
        alpha: dict[str, float],
        ticker: str,
        entry_price: float = 0.0,
        atr: float = 1.0,
        open_positions: int = 0,
    ) -> dict[str, Any]:
        """FAST PATH: All decisions from neuromorphic brain."""
        if self.state.is_refractory():
            return self._refractory_response(ticker)
        return self._evaluate_fast(ticker, alpha, entry_price, atr, open_positions)

    def _refractory_response(self, ticker: str) -> dict[str, Any]:
        """No-trade response during refractory period."""
        return {
            "ticker": ticker,
            "direction": 0,
            "score": 0.0,
            "verdict": "REFRACTORY",
            "confidence": 0.0,
            "sizing": SizingResult(),
            "regime": "refractory",
            "trace": {},
        }

    def _get_regime_data(self) -> tuple[float, str, str, float, float]:
        """Get regime modifiers from shared state (updated by slow path)."""
        return (
            self.state.get("regime_multiplier", 1.0),
            self.state.get("regime_label", "unknown"),
            self.state.get("regime_risk", "normal"),
            self.state.get("halim_modifier", 0.0),
            self.state.get("episodic_bias", 0.0),
        )

    def _process_alpha(self, alpha: dict[str, float], ticker: str) -> dict[str, Any]:
        """Process alpha through neuromorphic network."""
        if self._neuromorphic is None:
            return {"score": 0.0, "trace": {}}
        r = self._neuromorphic.process_alpha(alpha, ticker)
        t = r.get("trace", {})
        return {
            "score": r.get("score", 0.0),
            "spikes": t.get("spikes", 0),
            "evidence": t.get("evidence", {}),
        }

    def _evaluate_fast(
        self,
        ticker: str,
        alpha: dict[str, float],
        entry_price: float,
        atr: float,
        open_positions: int,
    ) -> dict[str, Any]:
        """Core fast evaluation — all decisions via neuromorphic brain."""
        r, rl, rr, hm, eb = self._get_regime_data()
        base = self.cortex.evaluate(alpha)
        neuro_result = self._process_alpha(alpha, ticker)
        neuro_score = neuro_result.get("score", 0.0)
        blended = (1 - self.NEURO_BLEND) * base.score + self.NEURO_BLEND * neuro_score
        score = (
            blended * r + hm + eb
        )  # EOD awareness: kill entries in last N minutes of RTH
        from ..config import TRADING_CONFIG
        from ..monitor.sleep_manager import SleepManager

        remaining = SleepManager().minutes_to_close()
        if 0 < remaining <= TRADING_CONFIG.eod_flatten_minutes:
            score = 0.0  # No new entries near close
            log.debug("EOD penalty: score zeroed (%.1f min to close)", remaining)
        direction = 1 if score > 0 else (-1 if score < 0 else 0)
        stabilized, dyn_reason = self.dynamics.process(score, direction)
        final_dir = 1 if stabilized > 0 else (-1 if stabilized < 0 else 0)
        sizing = self._maybe_size(
            stabilized, base.confidence, entry_price, atr, open_positions
        )
        self._store_decision(ticker, alpha, stabilized)
        return {
            "ticker": ticker,
            "verdict": base.verdict,
            "score": stabilized,
            "direction": final_dir,
            "confidence": base.confidence,
            "sizing": sizing,
            "regime": rl,
            "risk": rr,
            "trace": {
                "base": base.score,
                "regime": r,
                "halim": hm,
                "neuro": neuro_result,
            },
            "dyn_reason": dyn_reason,
        }

    def _store_decision(
        self, ticker: str, alpha: dict[str, float], score: float
    ) -> None:
        """Store decision data for learning + episodic memory."""
        self._last_alpha[ticker] = alpha
        self.memory.record_score(ticker, score)
        self.state.set_latest_alpha(alpha)
        self._decision_count += 1

    def on_trade_close(
        self, ticker: str, won: bool, pnl_pct: float, direction: int = 1
    ) -> None:
        """All learning routes through neuromorphic brain."""
        self.dynamics.adapt_threshold(self.memory.pred_error)
        self.exits.deregister(ticker)
        log.info("LEARN %s %s pnl=%.4f", ticker, "WIN" if won else "LOSS", pnl_pct)
        self.episodic.add(self._last_alpha.get(ticker, {}), pnl_pct)
        if self._neuromorphic is not None:
            self._neuromorphic.learn_from_outcome(ticker, won, pnl_pct)

    def on_ib_fill(self, fill: dict[str, Any]) -> None:
        """Route IB fill data to consolidation engine."""
        if self._consolidation is not None:
            self._consolidation.on_trade_close(
                fill["ticker"],
                fill.get("won", False),
                fill.get("pnl_pct", 0.0),
                fill.get("direction", 1),
                fill.get("qty", 1.0),
                fill.get("price", 0.0),
                fill.get("fees", 0.0),
            )

    def sleep_replay(self, is_market_open: bool = False) -> SleepResult:
        """OFFLINE: Sleep replay for memory consolidation."""
        if is_market_open or self._sleep_engine is None:
            return SleepResult()
        result = self._sleep_engine.run_cycle()
        log.info(
            "SLEEP: patterns=%d spikes=%d",
            result.patterns_replayed,
            result.spikes_generated,
        )
        return result

    def register_position(self, ticker: str, entry_price: float) -> None:
        """Register position for exit monitoring."""
        self.exits.register(ticker, entry_price, self._last_alpha.get(ticker, {}))

    def check_exit(
        self, ticker: str, current_price: float, ib_pnl: float = 0.0, direction: int = 1
    ) -> ExitSignal:
        """Check if position should be exited."""
        return self.exits.evaluate(ticker, current_price, ib_pnl, direction)

    def _maybe_size(
        self,
        score: float,
        confidence: float,
        entry_price: float,
        atr: float,
        open_positions: int,
    ) -> SizingResult:
        """Size position if score clears the dynamic threshold."""
        if abs(score) <= self.dynamics.threshold:
            return SizingResult()
        return self.risk.evaluate(score, confidence, entry_price, atr, open_positions)

    def snapshot(self) -> dict[str, Any]:
        """Full brain snapshot for telemetry."""
        result = {
            "memory": self.memory.snapshot(),
            "episodic_size": self.episodic.size,
            "threshold": self.dynamics.threshold,
            "brain_state": self.state.snapshot(),
            "decision_count": self._decision_count,
            "neuromorphic": self._neuromorphic.snapshot() if self._neuromorphic else {},
        }
        if self._sleep_engine is not None:
            result["sleep_engine"] = {
                "initialized": True,
                "cycle_count": self._sleep_engine._cycle_count,
            }
        return result


JuliBrain = NeuromorphicBrain  # Backwards compat alias
__all__ = ["NeuromorphicBrain", "JuliBrain"]
