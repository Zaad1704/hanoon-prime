"""hanoon_prime.brain.orchestrator — Neuromorphic Brain Coordinator.

IB feeds trade data; this brain makes ALL decisions.
Slow path: ConsolidationEngine for background work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from ..cortex import Cortex
from ..hippocampus import Hippocampus
from ..types import FillInfo
from . import horizons
from .affective import Affective
from .cognitive.nash import MOD_BOUND as NASH_MOD_BOUND
from .cognitive.nash import NashBrain, NashPrediction
from .config import (
    _IRONYCLADE,
    DEFAULT_WEIGHTS,
    GATE_CLOSED_SIZE_SCALAR,
    NASH_PENALTY_MAX,
    NASH_VETO_HIGH,
    NASH_VETO_LOW,
)
from .consolidation import ConsolidationEngine
from .deliberation import Deliberator
from .dynamics import Dynamics
from .episodic import EpisodicMemory
from .exit_checks import ExitSignal
from .exits import ExitPolicy
from .gate_advisor import GateAdvisor
from .memory import JuliMemory
from .neurons.bridge import NeuromorphicBridge
from .neurons.sleep import SleepReplayEngine, SleepResult
from .realized_ev import RealizedStats
from .reflection import Reflector, TradeClose
from .risk import RiskEngine, SizingResult
from .shared_state import BrainState

log = logging.getLogger(__name__)
NEURO_BLEND: float = 0.3


@dataclass
class VerdictLabels:
    """Context labels bundled into a tick result."""

    regime_label: str
    risk_label: str
    halim: float


class NeuromorphicBrain:
    """Neuromorphic brain — LOCAL SOURCE OF TRUTH for all decisions."""

    def __init__(
        self, brain_state: BrainState | None = None, enable_neuromorphic: bool = True
    ) -> None:
        self.state = brain_state or BrainState()
        self.memory = JuliMemory()
        self._last_alpha: dict[str, dict[str, float]] = {}
        self._last_score: dict[str, float] = {}
        self._last_conf: dict[str, float] = {}
        self._realized: RealizedStats = RealizedStats()
        self._decision_count: int = 0
        self.episodic = EpisodicMemory()
        # The cortex scores ALL 27 weighted indicators; learned weights are
        # overlaid on the structural defaults (the memory may hold a subset).
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(self.memory.get_weights())
        self.cortex = Cortex(weights=weights)
        self.hippocampus = Hippocampus(cortex=self.cortex, safety_enabled=False)
        self.affective = Affective()
        self.deliberator = Deliberator(threshold=self.memory.threshold)
        self.dynamics = Dynamics(base_threshold=self.memory.threshold)
        self.nash = NashBrain()
        self.risk = RiskEngine(realized=self._realized)
        self.exits = ExitPolicy()
        self._reflector = Reflector(self.memory, self.episodic)
        self._advisor = GateAdvisor(realized=self._realized)
        self._neuromorphic: Optional[NeuromorphicBridge] = None
        self._sleep_engine: Optional[SleepReplayEngine] = None
        self._consolidation: Optional[ConsolidationEngine] = None
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
        bars: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """FAST PATH: All decisions from neuromorphic brain.

        ``bars`` (optional close/high/low arrays + regime intel) lets the
        brain classify the candidate's trading horizon itself — the
        horizon is part of the brain's world model, not an external tag.
        """
        if self.state.is_refractory():
            return self._refractory_response(ticker)
        return self._evaluate_fast(
            ticker, alpha, entry_price, atr, open_positions, bars=bars
        )

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

    def _get_regime_data(self) -> tuple[float, str, str, float, float, float]:
        """Get regime modifiers from shared state."""
        return (
            self.state.get("regime_multiplier", 1.0),
            self.state.get("regime_label", "unknown"),
            self.state.get("regime_risk", "normal"),
            self.state.get("halim_modifier", 0.0),
            self.state.get("episodic_bias", 0.0),
            self.state.get("nash_modifier", 0.0),
        )

    def _compute_neuro_score(self, alpha: dict[str, float], ticker: str) -> float:
        """Compute neuromorphic score."""
        if not self._neuromorphic:
            return 0.0
        result: dict[str, Any] = self._neuromorphic.process_alpha(alpha, ticker)
        return float(result.get("score", 0.0))

    def _compute_nash_mod(self, nash_pred: NashPrediction) -> float:
        """Compute Nash modifier from prediction."""
        return float(
            NASH_MOD_BOUND * 4 * (nash_pred.confidence * (nash_pred.win_prob - 0.5))
        )

    def _apply_nash_gate(
        self, score: float, direction: int, nash_pred: NashPrediction
    ) -> float:
        """Apply Nash's pattern memory as a BOUNDED PENALTY (brain-first).

        The old behavior zeroed the score (hard veto) — a one-way lock that
        could deadlock the brain on its own learned history (the exact
        failure mode rebuild engineered around in its Nash/EV gates). Now
        the pattern memory LEANS: a strong losing pattern subtracts a
        bounded penalty scaled by pattern confidence and win-prob deficit,
        but an overwhelming brain signal can still clear the threshold.
        Pattern memory advises; the brain decides.
        """
        if not nash_pred.gate_authority:
            return score
        conf = max(0.0, min(1.0, nash_pred.confidence))
        if direction > 0 and nash_pred.win_prob < NASH_VETO_LOW:
            deficit = (NASH_VETO_LOW - nash_pred.win_prob) / NASH_VETO_LOW
            return score - NASH_PENALTY_MAX * conf * deficit
        if direction < 0 and nash_pred.win_prob > NASH_VETO_HIGH:
            # Short side: score is NEGATIVE — leaning away means pulling
            # it toward zero (ADD the penalty), not amplifying the short.
            deficit = (nash_pred.win_prob - NASH_VETO_HIGH) / (1.0 - NASH_VETO_HIGH)
            return score + NASH_PENALTY_MAX * conf * deficit
        return score

    def _check_eod_penalty(self) -> float:
        """Return score penalty multiplier (0 if EOD, else 1)."""
        from ..config import TRADING_CONFIG
        from ..monitor.sleep_manager import SleepManager

        remaining = SleepManager().minutes_to_close()
        if 0 < remaining <= TRADING_CONFIG.eod_flatten_minutes:
            return 0.0
        return 1.0

    def _evaluate_fast(
        self,
        ticker: str,
        alpha: dict[str, float],
        entry_price: float,
        atr: float,
        open_positions: int,
        bars: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Core fast evaluation — all decisions via neuromorphic brain."""
        _r, rl, rr, hm, eb, _ = self._get_regime_data()
        # Horizon classification lives IN the brain (bars → horizon tag).
        horizon = self._classify_horizon(bars)
        ctx = self._score_pipeline(ticker, alpha, _r, hm, eb)
        ctx["horizon"] = horizon
        sizing = self._maybe_size(ctx, entry_price, atr, open_positions)
        # Gate advisor: realized-WR threshold raise + gate-closed size scalar.
        # Advisory by construction — it tunes the entry bar and sizing, it
        # never produces or flips a verdict.
        if sizing.risk_pass:
            if self._advisor.is_tightening():
                sizing.shares = max(1, int(sizing.shares * GATE_CLOSED_SIZE_SCALAR))
        self._store_decision(ticker, alpha, ctx["stabilized"], ctx["confidence"])
        self.state.update(
            nash_modifier=ctx["nash_op"], nash_win_prob=ctx["nash_win_prob"]
        )
        return self._build_tick_result(ticker, VerdictLabels(rl, rr, hm), ctx, sizing)

    def _news_bias(self, ticker: str) -> float:
        """Bounded live-news sentiment bias (System 2 evidence, ±0.03).

        Reads the news organ's published sentiment from shared state —
        the fast path never touches the network.
        """
        try:
            sent = self.state.get("news_sentiment", {}) or {}
            pol = float(sent.get(ticker, 0.0))
        except Exception:
            return 0.0
        return max(-0.03, min(0.03, pol * 0.03))

    @staticmethod
    def _classify_horizon(bars: dict[str, Any] | None) -> str:
        """Classify the trading horizon from bar arrays (never blocks)."""
        if not bars:
            return "scalp"
        close = bars.get("close")
        if close is None or len(close) < 10:
            return "scalp"
        classified = horizons.classify(
            close,
            bars.get("high"),
            bars.get("low"),
            regime=str(bars.get("regime", "unknown")),
            halim_verdict=bars.get("halim_verdict"),
        )
        # A disabled classification snaps to the closest ACTIVE rung.
        return horizons.get_horizon_manager().active(classified)

    def _score_pipeline(
        self,
        ticker: str,
        alpha: dict[str, float],
        regime_mul: float,
        halim: float,
        episodic: float,
    ) -> dict[str, Any]:
        """Compute the blended stabilized score and decision intermediates."""
        base = self.cortex.evaluate(alpha)
        nash_pred = self.nash.predict(alpha, base.score, base.direction)
        nash_op = self._compute_nash_mod(nash_pred)
        neuro_score = self._compute_neuro_score(alpha, ticker)
        blended = (1 - NEURO_BLEND) * base.score + NEURO_BLEND * neuro_score
        # Gate advisor: learned threshold delta (tighten after losing streaks,
        # relieve after winning ones). Advisory — folds into the score before
        # the dynamics stabilize it.
        advisor_delta = self._advisor.threshold_delta()
        news_bias = self._news_bias(ticker)
        raw = blended * regime_mul + halim + episodic + nash_op
        raw += news_bias - advisor_delta
        stabilized, dyn_reason, final_dir = self._stabilize(raw, nash_pred)
        return {
            "base": base,
            "nash_pred": nash_pred,
            "nash_op": nash_op,
            "neuro_score": neuro_score,
            "confidence": base.confidence,
            "raw_score": raw,
            "stabilized": stabilized,
            "final_dir": final_dir,
            "dyn_reason": dyn_reason,
            "nash_win_prob": nash_pred.win_prob,
            "advisor_delta": advisor_delta,
        }

    def _stabilize(
        self, raw: float, nash_pred: NashPrediction
    ) -> tuple[float, str, int]:
        """Nash lean + EOD zero + dynamics stabilization on the raw score."""
        direction = 1 if raw > 0 else (-1 if raw < 0 else 0)
        score = self._apply_nash_gate(raw, direction, nash_pred)
        if self._check_eod_penalty() == 0.0:
            score = 0.0
        stabilized, dyn_reason = self.dynamics.process(score, direction)
        final_dir = 1 if stabilized > 0 else (-1 if stabilized < 0 else 0)
        return stabilized, dyn_reason, final_dir

    def _maybe_size(
        self,
        ctx: dict[str, Any],
        entry_price: float,
        atr: float,
        open_positions: int,
    ) -> SizingResult:
        """Size position if score clears the dynamic threshold.

        Per-horizon patience scales the bar (longer horizons accept a
        slightly lower score; scalp keeps the full bar). Mechanical, not
        a gate — it shapes the sizing bar only.
        """
        score = float(ctx["stabilized"])
        horizon = str(ctx.get("horizon", "scalp"))
        patience = horizons.params_for(horizon).patience
        if abs(score) <= self.dynamics.threshold * patience:
            return SizingResult()
        return self.risk.evaluate(
            score,
            float(ctx["confidence"]),
            entry_price,
            atr,
            open_positions,
            horizon=horizon,
        )

    def _build_tick_result(
        self,
        ticker: str,
        labels: VerdictLabels,
        ctx: dict[str, Any],
        sizing: SizingResult,
    ) -> dict[str, Any]:
        """Assemble the verdict dict returned by tick()."""
        return {
            "ticker": ticker,
            "verdict": ctx["base"].verdict,
            "score": ctx["stabilized"],
            "direction": ctx["final_dir"],
            "confidence": ctx["confidence"],
            "sizing": sizing,
            "horizon": ctx.get("horizon", "scalp"),
            "regime": labels.regime_label,
            "risk": labels.risk_label,
            "trace": {
                "base": ctx["base"].score,
                "neuro": {"score": ctx["neuro_score"]},
                "nash": ctx["nash_op"],
                "halim": labels.halim,
            },
            "dyn_reason": ctx["dyn_reason"],
            "nash_win_prob": ctx["nash_win_prob"],
        }

    def _store_decision(
        self, ticker: str, alpha: dict[str, float], score: float, confidence: float
    ) -> None:
        """Store decision data for learning + episodic memory."""
        self._last_alpha[ticker] = alpha
        self._last_score[ticker] = score
        self._last_conf[ticker] = confidence
        self.memory.record_score(ticker, score)
        self.state.set_latest_alpha(alpha)
        self._decision_count += 1

    def on_trade_close(
        self,
        ticker: str,
        won: bool,
        pnl_pct: float,
        direction: int = 1,
        source: str = "real_trade",
    ) -> None:
        """All learning routes through neuromorphic brain — evolved per trade.

        Single-writer guarantee: this method owns every learning write.
        The consolidation buffer path (System 2) is telemetry/consolidation
        only — it must NOT also write memory.
        """
        self.dynamics.adapt_threshold(self.memory.pred_error)
        self.exits.deregister(ticker)
        log.info("LEARN %s %s pnl=%.4f", ticker, "WIN" if won else "LOSS", pnl_pct)
        self.episodic.add(self._last_alpha.get(ticker, {}), pnl_pct)
        if self._last_alpha.get(ticker):
            self.nash.record_outcome(self._last_alpha[ticker], 0.0, won)
        if self._neuromorphic is not None:
            self._neuromorphic.learn_from_outcome(ticker, won, pnl_pct)
        # IRONYCLADE: the realized-EV feedback loop only learns from real
        # executions (real_trade/ib_fill/ib_paper). Paper & synthetic fills
        # are excluded so the live gate isn't trained on backtest noise.
        if source in _IRONYCLADE:
            self._learn_from_real(ticker, won, pnl_pct, direction)

    def _learn_from_real(
        self, ticker: str, won: bool, pnl_pct: float, direction: int
    ) -> None:
        """The closed learning loop — runs once per REAL trade close.

        (a) Weight gradient over ALL 27 indicators (loss-aversion 1.2x via
        Reflector — the single writer for weights/episodes/calibration).
        (b) Hot-swap learned weights into the cortex so the very next tick
        scores with the updated brain.
        (c) Realized band/RR/conf-bins + calibration (the realized-EV gate).
        (d) Learned exits + gate advisor retune from the new sample.
        """
        conf = self._last_conf.get(ticker, 0.5)
        score = self._last_score.get(ticker, 0.0)
        self._reflector.on_trade_close(
            TradeClose(
                ticker=ticker,
                won=won,
                pnl_pct=pnl_pct,
                direction=direction,
                alpha=self._last_alpha.get(ticker, {}),
                predicted_score=conf,
            )
        )
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(self.memory.get_weights())
        self.cortex.set_weights(weights)
        self.memory.record_outcome(won)
        self.memory.update_pred_error(conf, 1.0 if won else 0.0)
        self._realized.add_outcome(score, won, pnl_pct, direction)
        self._realized.add_confidence_outcome(conf, won)
        self.exits.adapt_from_realized(self._realized)
        self._advisor.record_outcome(won)

    def on_ib_fill(self, fill: dict[str, Any]) -> None:
        """Route IB fill data to consolidation engine."""
        if self._consolidation is not None:
            self._consolidation.on_trade_close(
                fill["ticker"],
                fill.get("won", False),
                fill.get("pnl_pct", 0.0),
                fill.get("direction", 1),
                FillInfo(
                    qty=fill.get("qty", 1.0),
                    avg_price=fill.get("price", 0.0),
                    fees=fill.get("fees", 0.0),
                ),
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

    def register_position(
        self, ticker: str, entry_price: float, horizon: str = "scalp"
    ) -> None:
        """Register position for exit monitoring (per-horizon exit windows)."""
        self.exits.register(
            ticker, entry_price, self._last_alpha.get(ticker, {}), horizon=horizon
        )

    def check_exit(
        self, ticker: str, current_price: float, ib_pnl: float = 0.0, direction: int = 1
    ) -> ExitSignal:
        """Check if position should be exited."""
        return self.exits.evaluate(ticker, current_price, ib_pnl, direction)

    def snapshot(self) -> dict[str, Any]:
        """Full brain snapshot for telemetry."""
        result = {
            "memory": self.memory.snapshot(),
            "realized": self._realized.snapshot(),
            "episodic_size": self.episodic.size,
            "threshold": self.dynamics.threshold,
            "brain_state": self.state.snapshot(),
            "decision_count": self._decision_count,
            "neuromorphic": self._neuromorphic.snapshot() if self._neuromorphic else {},
            "nash": self.nash.get_telemetry(),
            "advisor": self._advisor.snapshot(),
            "exits_adaptive": self.exits.telemetry(),
        }
        if self._sleep_engine is not None:
            result["sleep_engine"] = {
                "initialized": True,
                "cycle_count": self._sleep_engine._cycle_count,
            }
        return result


JuliBrain = NeuromorphicBrain  # Backwards compat alias
__all__ = ["NeuromorphicBrain", "JuliBrain"]
