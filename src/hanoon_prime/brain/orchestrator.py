"""hanoon_prime.brain.orchestrator — System 1: Fast Reflexive Engine.

Sub-millisecond tick loop. Reads pre-computed parameters from
System 2's shared BrainState. Pure NumPy operations only.

Biological parallel: The brainstem — fast reflexive decisions
using pre-calibrated neural pathways from the cortex.
"""

from __future__ import annotations

import logging
from typing import Any

from ..cortex import Cortex, Thought
from ..hippocampus import Hippocampus
from .affective import Affective
from .deliberation import DeliberationResult, Deliberator
from .dynamics import Dynamics
from .episodic import EpisodicMemory
from .exits import ExitPolicy, ExitSignal
from .memory import JuliMemory
from .regime import RegimeState
from .risk import RiskEngine, SizingResult
from .salience import Salience
from .shared_state import BrainState

log = logging.getLogger(__name__)


class JuliBrain:
    """System 1: Fast reflexive brain — reads pre-computed state only."""

    def __init__(self, brain_state: BrainState | None = None) -> None:
        self.state = brain_state or BrainState()
        self.memory = JuliMemory()
        self.episodic = EpisodicMemory()
        weights = self.memory.get_weights()
        self.cortex = Cortex(weights=weights)
        self.hippocampus = Hippocampus(cortex=self.cortex, safety_enabled=False)
        self.affective = Affective()
        self.salience = Salience()
        self.deliberator = Deliberator(threshold=self.memory.threshold)
        self.dynamics = Dynamics(base_threshold=self.memory.threshold)
        self.risk = RiskEngine()
        self.exits = ExitPolicy()
        self._last_alpha: dict[str, dict[str, float]] = {}
        self._last_score: dict[str, float] = {}

    def tick(
        self,
        alpha: dict[str, float],
        ticker: str,
        entry_price: float = 0.0,
        atr: float = 1.0,
        open_positions: int = 0,
    ) -> dict[str, Any]:
        """Fast tick — reads System 2 state, pure NumPy ops."""
        if self.state.is_refractory():
            return self._refractory_response(ticker)
        return self._fast_evaluate(ticker, alpha, entry_price, atr, open_positions)

    def _refractory_response(self, ticker: str) -> dict[str, Any]:
        """Return no-trade response during refractory period."""
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

    def _fast_evaluate(
        self,
        ticker: str,
        alpha: dict[str, float],
        entry_price: float,
        atr: float,
        open_positions: int,
    ) -> dict[str, Any]:
        """Core fast evaluation — no I/O, pure computation."""
        regime_mult = self.state.get("regime_multiplier", 1.0)
        regime_label = self.state.get("regime_label", "unknown")
        halim_mod = self.state.get("halim_modifier", 0.0)
        episodic_bias = self.state.get("episodic_bias", 0.0)
        base = self.cortex.evaluate(alpha)
        score = base.score * regime_mult + halim_mod + episodic_bias
        direction = 1 if score > 0 else (-1 if score < 0 else 0)
        stabilized, dyn_reason = self.dynamics.process(score, direction)
        final_dir = 1 if stabilized > 0 else (-1 if stabilized < 0 else 0)
        sizing = self._maybe_size(
            stabilized, base.confidence, entry_price, atr, open_positions
        )
        self._last_alpha[ticker] = alpha
        self._last_score[ticker] = stabilized
        self.memory.record_score(ticker, stabilized)
        self.state.set_latest_alpha(alpha)
        return {
            "ticker": ticker,
            "verdict": base.verdict,
            "score": stabilized,
            "direction": final_dir,
            "confidence": base.confidence,
            "sizing": sizing,
            "regime": regime_label,
            "trace": {"base": base.score, "regime": regime_mult, "halim": halim_mod},
            "dyn_reason": dyn_reason,
        }

    def on_trade_close(
        self, ticker: str, won: bool, pnl_pct: float, direction: int = 1
    ) -> None:
        """Called when a trade closes — triggers reflection."""
        self.dynamics.adapt_threshold(self.memory.pred_error)
        self.exits.deregister(ticker)
        log.info("REFLECT %s %s pnl=%.4f", ticker, "WIN" if won else "LOSS", pnl_pct)

    def register_position(self, ticker: str, entry_price: float) -> None:
        """Register a new position for exit monitoring."""
        alpha = self._last_alpha.get(ticker, {})
        self.exits.register(ticker, entry_price, alpha)

    def check_exit(
        self, ticker: str, current_price: float, ib_pnl: float = 0.0, direction: int = 1
    ) -> ExitSignal:
        """Check if a position should be exited."""
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
        return {
            "memory": self.memory.snapshot(),
            "episodic_size": self.episodic.size,
            "threshold": self.dynamics.threshold,
            "brain_state": self.state.snapshot(),
        }
