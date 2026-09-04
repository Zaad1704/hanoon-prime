"""hanoon_prime.brain.orchestrator — Juli's brain core pipeline.

The main entry point for all cognitive processing. On each tick:
  indicators → cortex → regime/episodic/affective → deliberation
  → dynamics → risk → exit signal

This is the biological brain. IB is the eyes, this is the mind.
"""

from __future__ import annotations

import logging
from typing import Any

from ..cortex import Cortex, Thought
from ..hippocampus import Hippocampus
from .affective import Affective
from .config import DEFAULT_WEIGHTS, SIGNAL_THRESHOLD
from .deliberation import DeliberationResult, Deliberator
from .dynamics import Dynamics
from .episodic import EpisodicMemory
from .exits import ExitPolicy, ExitSignal
from .guardians import Guardians
from .halim_adapter import HalimAdapter
from .memory import JuliMemory
from .reflection import Reflector
from .regime import RegimeDetector, RegimeState
from .risk import RiskEngine, SizingResult
from .salience import Salience

log = logging.getLogger(__name__)


class JuliBrain:
    """The full biological brain — orchestrates all cognitive subsystems."""

    def __init__(self, halim_url: str = "http://127.0.0.1:8765") -> None:
        self.memory = JuliMemory()
        self.episodic = EpisodicMemory()
        weights = self.memory.get_weights()
        self.cortex = Cortex(weights=weights)
        self.hippocampus = Hippocampus(cortex=self.cortex, safety_enabled=False)
        self.regime = RegimeDetector()
        self.affective = Affective()
        self.salience = Salience()
        self.deliberator = Deliberator(threshold=self.memory.threshold)
        self.dynamics = Dynamics(base_threshold=self.memory.threshold)
        self.risk = RiskEngine()
        self.exits = ExitPolicy()
        self.guardians = Guardians()
        self.halim = HalimAdapter(base_url=halim_url)
        self.reflector = Reflector(self.memory, self.episodic)
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
        """Full brain cycle for one ticker. Returns decision dict."""
        self._check_guardians()
        regime = self._detect_regime(alpha)
        base = self.cortex.evaluate(alpha)
        result = self._deliberate(ticker, base, alpha, regime)
        stabilized, dyn_reason = self.dynamics.process(result.score, result.direction)
        final_dir = 1 if stabilized > 0 else (-1 if stabilized < 0 else 0)
        sizing = self._maybe_size(
            stabilized, result.confidence, entry_price, atr, open_positions
        )
        self._last_alpha[ticker] = alpha
        self._last_score[ticker] = stabilized
        self.memory.record_score(ticker, stabilized)
        return {
            "ticker": ticker,
            "verdict": result.verdict,
            "score": stabilized,
            "direction": final_dir,
            "confidence": result.confidence,
            "sizing": sizing,
            "regime": regime.regime,
            "trace": result.trace,
            "dyn_reason": dyn_reason,
        }

    def on_trade_close(
        self,
        ticker: str,
        won: bool,
        pnl_pct: float,
        direction: int = 1,
    ) -> None:
        """Called when a trade closes — triggers full reflection."""
        alpha = self._last_alpha.get(ticker, {})
        score = self._last_score.get(ticker, 0.0)
        self.reflector.on_trade_close(
            ticker,
            won,
            pnl_pct,
            direction,
            alpha,
            score,
        )
        self.affective.update(won, pnl_pct)
        self.exits.deregister(ticker)
        self.dynamics.adapt_threshold(self.memory.pred_error)

    def register_position(self, ticker: str, entry_price: float) -> None:
        """Register a new position for exit monitoring."""
        alpha = self._last_alpha.get(ticker, {})
        self.exits.register(ticker, entry_price, alpha)

    def check_exit(
        self,
        ticker: str,
        current_price: float,
        ib_pnl: float = 0.0,
        direction: int = 1,
    ) -> ExitSignal:
        """Check if a position should be exited."""
        return self.exits.evaluate(ticker, current_price, ib_pnl, direction)

    def _deliberate(
        self, ticker: str, base: Thought, alpha: dict[str, float], regime: RegimeState
    ) -> DeliberationResult:
        """Run full deliberation: episodic + affective + salience + halim."""
        episodic_mod = self.episodic.modifier(alpha)
        self.affective.update(won=True, pnl_pct=0.0)
        aff = self.affective.evaluate()
        self.salience.update(base.score, regime.vol_percentile)
        sal = self.salience.evaluate(base.score, regime.multiplier)
        halim_mod = self.halim.get_modifier(ticker, alpha, base.score, base.verdict)
        return self.deliberator.deliberate(
            base_score=base.score,
            base_confidence=base.confidence,
            episodic_mod=episodic_mod,
            affective_mod=aff.modifier,
            salience_atten=sal.confidence_atten,
            regime_mult=regime.multiplier,
            halim_mod=halim_mod,
        )

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

    def _detect_regime(self, alpha: dict[str, float]) -> RegimeState:
        """Detect market regime from alpha indicators."""
        prices = [alpha.get(k, 0.5) for k in ["vpin", "momentum", "vwap_deviation"]]
        return self.regime.detect(prices)

    def _check_guardians(self) -> None:
        """Run safety checks."""
        weights = self.memory.get_weights()
        verdict = self.guardians.check_weights(weights)
        if not verdict.safe:
            log.warning("Guardian: %s", verdict.reason)
        stab = self.guardians.check_learning_stability(self.memory.pred_error)
        if not stab.safe:
            log.warning("Learning instability: %s", stab.reason)

    def snapshot(self) -> dict[str, Any]:
        """Full brain snapshot for telemetry."""
        return {
            "memory": self.memory.snapshot(),
            "episodic_size": self.episodic.size,
            "threshold": self.dynamics.threshold,
            "affective": self.affective.evaluate().__dict__,
        }
