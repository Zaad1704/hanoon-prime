"""hanoon_prime.brain.orchestrator — Neuromorphic Brain Coordinator.

IB feeds trade data; this brain makes ALL decisions.
Slow path: ConsolidationEngine for background work.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

from ..cortex import Cortex
from ..edge import score_to_win_prob
from ..hippocampus import Hippocampus
from ..juli_feed import check_tick_latency, compute_alpha_from_snap, entry_bars
from ..types import FillInfo
from . import horizons
from .cognitive.emotion import CONF_BOUND, RISK_CEIL, RISK_FLOOR
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
from .cross_asset import CrossAssetEngine
from .dynamics import Dynamics
from .episodic import EpisodicMemory
from .exit_checks import ExitSignal
from .exit_ladder import ExitLadder
from .exits import ExitPolicy
from .gate_advisor import GateAdvisor
from .horizon_bandit import HorizonBandit
from .learned_exit import LearnedExitPolicy
from .learning_config import CROSS_ASSET_MOD_BOUND, REGIME_MIN_TRADES
from .memory import JuliMemory
from .meta_label import MetaLabelModel
from .meta_label import feature_vector as meta_features
from .neurons.bridge import NeuromorphicBridge
from .neurons.sleep import SleepReplayEngine, SleepResult
from .policy.governor import Governor
from .policy.portfolio_gate import portfolio_gate, scale_shares
from .policy.trading_policy import TRADING_CONFIG
from .policy.verdict import ENTER, HOLD, VETOED, Verdict
from .probe_recovery import ProbeRecovery
from .realized_ev import RealizedStats
from .reflection import Reflector, TradeClose
from .regime import RegimeDetector
from .regime_weights import RegimeWeights
from .risk import RiskEngine, SizingResult
from .shared_state import DEFAULT_POLICY_STATE, BrainState
from .strategy_genome import StrategyGenome
from .thinker import TOTAL_MOD_BOUND

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
        self._eval_fail_count: int = 0
        self.episodic = EpisodicMemory()
        # The cortex scores ALL 27 weighted indicators; learned weights are
        # overlaid on the structural defaults (the memory may hold a subset).
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(self.memory.get_weights())
        # Unified threshold: cortex admission and dynamics sizing share ONE
        # adaptive number (memory threshold). _update_admission_threshold()
        # pushes the dynamics value into the cortex each cycle, so the
        # prediction-error calibration loop actually moves who enters.
        self.cortex = Cortex(weights=weights, threshold=self.memory.threshold)
        self.hippocampus = Hippocampus(cortex=self.cortex, safety_enabled=False)
        self.dynamics = Dynamics(base_threshold=self.memory.threshold)
        self.nash = NashBrain()
        self.risk = RiskEngine(realized=self._realized)
        self.exits = ExitPolicy()
        self._exit_ladder = ExitLadder(
            self.exits, win_rate_provider=self._realized.recent_win_rate
        )
        self._reflector = Reflector(self.memory, self.episodic)
        self._advisor = GateAdvisor(realized=self._realized)
        self.governor = Governor()
        self.probe = ProbeRecovery()
        self.trading_policy = TRADING_CONFIG
        self._init_strategy_organs()
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

    def _init_strategy_organs(self) -> None:
        """Wire the strategy-learning organs (all advisory, bounded).

        Regime fallback, cross-asset lead-lag, meta-label sizing, horizon
        bandit, per-regime weights, learned-exit attribution — plus the
        per-ticker decision context the close path learns from. Cortex
        stays the sole verdict source (R1).
        """
        self._regime_detector = RegimeDetector()
        self._cross_asset = CrossAssetEngine()
        self._meta = MetaLabelModel()
        self._bandit = HorizonBandit()
        self._regime_weights = RegimeWeights()
        self._learned_exit = LearnedExitPolicy()
        self.genome = StrategyGenome(self)
        self._last_regime: dict[str, str] = {}
        self._last_vol_pct: dict[str, float] = {}
        self._last_horizon: dict[str, str] = {}
        self._wkey: tuple[str, int] = ("", -1)
        self._wversion: int = 0

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

    def begin_entry_cycle(self) -> None:
        """Start one entry cycle: fresh governor budget (called by juli)."""
        self.governor.begin_cycle()

    def decide_entry(
        self,
        ticker: str,
        snap: dict[str, Any] | None,
        open_positions: set[str] | dict[str, Any],
        session: str = "rth",
    ) -> Verdict:
        """THE single entry decision point (fast path, policy from snapshot)."""
        veto = self._check_snapshot_valid(snap)
        if veto:
            return Verdict(ticker=ticker, action=VETOED, reason=veto, stage="validity")
        assert snap is not None
        prices = snap.get("prices")
        if prices is None:
            prices = []
        self.state.set_latest_prices(prices)
        outcome = self._score_candidate(ticker, snap, len(open_positions))
        if isinstance(outcome, Verdict):
            return outcome
        result, bars, thought = outcome
        policy = self.state.get("policy_state", DEFAULT_POLICY_STATE)
        if not isinstance(policy, dict):
            policy = dict(DEFAULT_POLICY_STATE)
        admitted = self._apply_fast_gates(ticker, snap, thought, policy, session)
        if admitted is not None:
            return admitted
        ok, reason = self.governor.may_enter(ticker)
        if not ok:
            return Verdict(
                ticker=ticker, action=VETOED, reason=reason, stage="governor"
            )
        return self._admit_verdict(
            ticker, snap, result, bars, thought, policy, len(open_positions)
        )

    def _check_snapshot_valid(self, snap: dict[str, Any] | None) -> str:
        """Return a veto reason when the snapshot is unusable, else ''."""
        if snap is None or not isinstance(snap, dict):
            return "no_data"
        prices = snap.get("prices") or []  # array-safe (plain list)
        if len(prices) < 20:
            return "no_data"
        for key in ("last", "bid", "ask", "mid"):
            value = snap.get(key)
            if value is not None and isinstance(value, (int, float)):
                if math.isnan(float(value)):
                    return "no_data"
        return ""

    def _score_candidate(
        self, ticker: str, snap: dict[str, Any], open_count: int
    ) -> tuple[dict[str, Any], dict[str, Any], SimpleNamespace] | Verdict:
        """Score through the brain; Verdict short-circuit on failure."""
        prices = snap.get("prices") or []  # array-safe (plain list)
        t0 = time.perf_counter_ns()
        try:
            alpha = compute_alpha_from_snap(snap)
            bars = entry_bars(snap, prices, self.state.get("regime_label", "unknown"))
            result = self.tick(
                alpha,
                ticker,
                entry_price=float(prices[-1]),
                atr=float(snap.get("atr", 1.0)),
                open_positions=open_count,
                bars=bars,
            )
            check_tick_latency(t0, ticker)
        except Exception as e:
            self.note_eval_failure(ticker, e)
            log.warning("Entry eval failed for %s: %s", ticker, e)
            return Verdict(
                ticker=ticker,
                action=VETOED,
                reason="eval_error",
                stage="pipeline",
            )
        thought = self._extract_thought(result, ticker)
        if isinstance(thought, Verdict):
            return thought
        return result, bars, thought

    def _extract_thought(
        self, result: dict[str, Any], ticker: str
    ) -> SimpleNamespace | Verdict:
        """Normalize direction/score/confidence or short-circuit HOLD."""
        thought_raw = result.get("thought")
        if isinstance(thought_raw, SimpleNamespace):
            direction = int(thought_raw.direction)
            score = float(getattr(thought_raw, "score", 0.0))
            confidence = float(getattr(thought_raw, "confidence", 0.5))
        else:
            direction = int(result.get("direction", 0))
            score = float(result.get("score", 0.0))
            confidence = float(result.get("confidence", 0.5))
        if direction == 0:
            return Verdict(
                ticker=ticker, action=HOLD, reason="no_signal", stage="pipeline"
            )
        return SimpleNamespace(direction=direction, score=score, confidence=confidence)

    def _apply_fast_gates(
        self,
        ticker: str,
        snap: dict[str, Any],
        thought: SimpleNamespace,
        policy: dict[str, Any],
        session: str,
    ) -> Verdict | None:
        """Session/direction/penny/safety gates; None = admitted through."""
        if not self.trading_policy.is_session_active(session):
            return Verdict(
                ticker=ticker,
                action=VETOED,
                reason="session_disabled",
                stage="trading_policy",
            )
        side = "BUY" if thought.direction > 0 else "SELL"
        if not self.trading_policy.is_direction_allowed(side):
            return Verdict(
                ticker=ticker,
                action=VETOED,
                reason="direction_rejected",
                stage="trading_policy",
                direction=thought.direction,
            )
        price = float(snap.get("last") or 0.0)
        cleared, penalty = self.trading_policy.is_penny_bar_cleared(
            ticker, price, float(thought.score)
        )
        if not cleared:
            return Verdict(
                ticker=ticker,
                action=VETOED,
                reason=penalty,
                stage="trading_policy",
            )
        if policy.get("authorized", True) is False:
            return self._halted_verdict(ticker, thought, snap, policy)
        return None

    def _halted_verdict(
        self,
        ticker: str,
        thought: SimpleNamespace,
        snap: dict[str, Any],
        policy: dict[str, Any],
    ) -> Verdict:
        """Safety halt: probe recovery admits; else VETOED at the reason."""
        probe = self.probe.maybe_probe(
            float(thought.score),
            float(snap.get("bid", 0.0)),
            float(snap.get("ask", 0.0)),
            int(policy.get("consecutive_losses", 0)),
        )
        if probe:
            log.info("PROBE %s: recovery entry admitted", ticker)
            return Verdict(
                ticker=ticker,
                action=ENTER,
                reason="probe_recovery",
                stage="probe_recovery",
                direction=thought.direction,
                score=float(thought.score),
            )
        return Verdict(
            ticker=ticker,
            action=VETOED,
            reason=str(policy.get("pause_reason") or "halted"),
            stage="safety",
        )

    def _admit_verdict(
        self,
        ticker: str,
        snap: dict[str, Any],
        result: dict[str, Any],
        bars: dict[str, Any],
        thought: SimpleNamespace,
        policy: dict[str, Any],
        open_count: int,
    ) -> Verdict:
        """Governor admitted: size, portfolio-gate, scale, build ENTER."""
        sizing = result.get("sizing")
        if not isinstance(sizing, SizingResult):
            sizing = self._size_entry(ticker, snap, bars, thought, open_count)
        if sizing is None or not sizing.risk_pass or int(sizing.shares) <= 0:
            return Verdict(
                ticker=ticker, action=HOLD, reason="not_sized", stage="pipeline"
            )
        return self._portfolio_admit(ticker, snap, sizing, policy, thought)

    def _portfolio_admit(
        self,
        ticker: str,
        snap: dict[str, Any],
        sizing: SizingResult,
        policy: dict[str, Any],
        thought: SimpleNamespace,
    ) -> Verdict:
        """Portfolio risk gate + scaling; Verdict short-circuit on veto."""
        prices = snap.get("prices") or []  # array-safe (plain list)
        price = float(snap.get("last") or snap.get("mid") or prices[-1])
        notional = float(sizing.shares) * price
        ok, reason = portfolio_gate(ticker, notional, policy)
        if not ok:
            return Verdict(
                ticker=ticker,
                action=VETOED,
                reason=reason,
                stage="portfolio_risk",
            )
        shares = scale_shares(
            int(sizing.shares),
            price,
            float(policy.get("risk_scalar", 1.0)),
            float(policy.get("exposure", 0.0)),
        )
        if shares <= 0:
            return Verdict(
                ticker=ticker,
                action=VETOED,
                reason="sized_to_zero",
                stage="portfolio_risk",
            )
        return self._build_enter(ticker, thought, sizing, shares)

    def _size_entry(
        self,
        _ticker: str,
        snap: dict[str, Any],
        bars: dict[str, Any],
        thought: SimpleNamespace,
        open_count: int,
    ) -> SizingResult:
        """Size an admitted candidate from thought (mirrors _maybe_size)."""
        prices = snap.get("prices") or []  # array-safe (plain list)
        horizon = self._classify_horizon(bars)
        patience = horizons.params_for(horizon).patience
        if abs(float(thought.score)) <= self.dynamics.threshold * patience:
            return SizingResult()
        return self.risk.evaluate(
            float(thought.score),
            float(thought.confidence),
            float(prices[-1]),
            float(snap.get("atr", 1.0)),
            open_count,
            horizon=horizon,
        )

    @staticmethod
    def _build_enter(
        ticker: str,
        thought: SimpleNamespace,
        sizing: SizingResult,
        shares: int,
    ) -> Verdict:
        """Log the admission and build the ENTER Verdict."""
        log.info(
            "THINK %s %s score=%.3f shares=%d",
            ticker,
            "BUY" if thought.direction > 0 else "SELL",
            float(thought.score),
            shares,
        )
        return Verdict(
            ticker=ticker,
            action=ENTER,
            reason="admitted",
            stage="entry",
            sizing=sizing,
            stop=sizing.stop_price,
            target=sizing.target_price,
            horizon=getattr(sizing, "horizon", "scalp"),
            score=float(thought.score),
            direction=int(thought.direction),
            thought=thought,
        )

    def note_entry(self, ticker: str) -> None:
        """Register an executed entry with the governor (post-fill)."""
        self.governor.note_entry(ticker)

    def resume(self) -> None:
        """Clear the brain-side halt state (webapp resume command)."""
        if self._consolidation is not None:
            self._consolidation.safety.resume()
        policy = self.state.get("policy_state", DEFAULT_POLICY_STATE)
        if not isinstance(policy, dict):
            policy = dict(DEFAULT_POLICY_STATE)
        self.state.update(
            policy_state={
                **policy,
                "authorized": True,
                "halted": False,
                "pause_reason": "",
            }
        )

    def set_safety_enabled(self, enabled: bool) -> None:
        """Toggle safety via the producer so the slow cortex honors it."""
        if self._consolidation is not None:
            self._consolidation.safety.set_enabled(bool(enabled))
        policy = self.state.get("policy_state", DEFAULT_POLICY_STATE)
        if not isinstance(policy, dict):
            policy = dict(DEFAULT_POLICY_STATE)
        self.state.update(policy_state={**policy, "enabled": bool(enabled)})

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

    def _canonical_regime(self, label: str) -> str:
        """Map any regime label to a canonical meta/bandit cell."""
        if label.startswith("trending_bull"):
            return "trend_up"
        if label.startswith("trending_bear"):
            return "trend_down"
        if label.startswith("trending"):
            return "trend_up"
        if label in ("volatile", "crisis"):
            return "vol"
        if label in ("ranging", "range"):
            return "range"
        if label in ("normal", "unknown", "refractory"):
            return "unknown"
        return "unknown"

    def _local_regime_fallback(self, label: str) -> tuple[float, str]:
        """Local RegimeDetector fallback when HALIM's label is stale.

        The slow path publishes regime from the HALIM service; when that
        label is stuck at "unknown" (service down) the local numpy
        detector classifies from the shared close prices instead so the
        strategy organs always see a real regime.
        """
        if label != "unknown":
            return 1.0, label
        prices = self.state.get_latest_prices() or []
        rs = self._regime_detector.detect(prices)
        if rs.regime == "unknown":
            return 1.0, label
        return rs.multiplier, rs.regime

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
        _r, rl, rr, hm, _, _ = self._get_regime_data()
        _r, rl = self._local_regime_fallback(rl)
        canon = self._canonical_regime(rl)
        # Cross-asset lead-lag (SPY/QQQ/IWM/VXX refs via shared state).
        cross = self._cross_asset.update(
            ticker, entry_price, self.state.get("ref_prices")
        ).modifier
        cross = max(-CROSS_ASSET_MOD_BOUND, min(CROSS_ASSET_MOD_BOUND, cross))
        # Horizon: mechanical classifier, bandit may override from realized
        # data (bounded, advisory — shapes patience/sizing, never verdicts).
        horizon = self._classify_horizon(bars)
        horizon, hz_reason = self._bandit.select(canon, horizon)
        self._apply_regime_weights(canon)
        # Episodic k-NN modifier is queried LIVE against the current alpha
        # (bounded ±EPISODIC_MOD_BOUND, confidence-gated) and mirrored into
        # shared state so telemetry shows the applied bias.
        eb = self.episodic.modifier(alpha)
        self.state.update(episodic_bias=eb)
        ctx = self._score_pipeline(ticker, alpha, _r, hm, eb, cross=cross)
        ctx["horizon"] = horizon
        ctx["horizon_reason"] = hz_reason
        ctx["regime_canon"] = canon
        sizing = self._maybe_size(ctx, entry_price, atr, open_positions)
        self._scale_admitted_size(ctx, sizing, canon, horizon, bars)
        self._store_decision(ticker, alpha, ctx["stabilized"], ctx["confidence"])
        self._remember_decision(ticker, canon, horizon, self._vol_pct(bars))
        self.state.update(
            nash_modifier=ctx["nash_op"], nash_win_prob=ctx["nash_win_prob"]
        )
        return self._build_tick_result(ticker, VerdictLabels(rl, rr, hm), ctx, sizing)

    def _scale_admitted_size(
        self,
        ctx: dict[str, Any],
        sizing: SizingResult,
        canon: str,
        horizon: str,
        bars: dict[str, Any] | None,
    ) -> None:
        """Scale an ADMITTED entry's size by advisor + meta-label factors.

        Advisory by construction — it tunes sizing only, never produces or
        flips a verdict (R1).
        """
        if not sizing.risk_pass:
            return
        if self._advisor.is_tightening():
            sizing.shares = max(1, int(sizing.shares * GATE_CLOSED_SIZE_SCALAR))
        # Affective state (fear/greed from live streak) scales size within
        # [RISK_FLOOR, RISK_CEIL] — a losing streak shrinks, a hot streak
        # grows. Bounded, advisory (never refuses/forces an entry).
        sizing.shares = max(1, int(sizing.shares * self._bounded_thinker_risk_scalar()))
        vol_pct = self._vol_pct(bars)
        meta_scale = self._meta.size_scalar(
            ctx["confidence"], ctx["stabilized"], vol_pct, canon, horizon
        )
        sizing.shares = max(1, int(sizing.shares * meta_scale))

    def note_eval_failure(self, ticker: str, err: Exception) -> None:
        """Count entry-eval failures (FIXES.md Class D): the pipeline
        monitor alerts when these accumulate — a silent total outage
        once hid behind per-ticker warnings."""
        self._eval_fail_count += 1
        if self._eval_fail_count % 10 == 1:
            log.warning("Eval failure #%s (%s): %s", self._eval_fail_count, ticker, err)

    def _remember_decision(
        self, ticker: str, canon: str, horizon: str, vol_pct: float
    ) -> None:
        """Store the decision context the close path will learn from."""
        self._last_regime[ticker] = canon
        self._last_vol_pct[ticker] = vol_pct
        self._last_horizon[ticker] = horizon

    @staticmethod
    def _vol_pct(bars: dict[str, Any] | None) -> float:
        """Volatility percentile proxy from bar context (meta feature)."""
        if not bars:
            return 0.5
        close = bars.get("close")
        if close is None or len(close) == 0:
            return 0.5
        window = list(close)[-20:]
        if len(window) < 10:
            return 0.5
        rets = [(b - a) / a for a, b in zip(window, window[1:]) if a]
        if not rets:
            return 0.5
        mean_abs = sum(abs(c) for c in window) / len(window)
        var = sum(r * r for r in rets) / len(rets)
        spread = (var**0.5) / (mean_abs + 1e-12)
        return float(max(0.0, min(1.0, spread * 100.0)))

    def _weights_for(self, regime: str) -> dict[str, float]:
        """Global learned weights overlaid with the regime vector if trained."""
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(self.memory.get_weights())
        regime_vec = self._regime_weights.weights_for(regime)
        if regime_vec:
            weights.update(regime_vec)
        return weights

    def _apply_regime_weights(self, regime: str) -> None:
        """Hot-swap regime-blended weights (cached per regime+version)."""
        key = (regime, self._wversion)
        if key == self._wkey:
            return
        self._wkey = key
        self.cortex.set_weights(self._weights_for(regime))

    def _bounded_thinker_modifier(self) -> float:
        """Slow-path 5-pillar deliberation bias, clamped to TOTAL_MOD_BOUND.

        The Thinker (semantic/episodic/planning/metacognition/nash) runs in
        the S2 slow path and publishes ``thinker_modifier`` to shared state;
        this fast-path reader consumes it — bounded so no single pillar
        dominates the verdict (R1: cortex still emits the verdict).
        """
        try:
            mod = float(self.state.get("thinker_modifier", 0.0) or 0.0)
        except Exception:
            return 0.0
        return max(-TOTAL_MOD_BOUND, min(TOTAL_MOD_BOUND, mod))

    def _bounded_thinker_confidence(self) -> float:
        """Slow-path affective confidence nudge, clamped to CONF_BOUND."""
        try:
            mod = float(self.state.get("thinker_confidence_mod", 0.0) or 0.0)
        except Exception:
            return 0.0
        return max(-CONF_BOUND, min(CONF_BOUND, mod))

    def _bounded_thinker_risk_scalar(self) -> float:
        """Affective (fear/greed) position-size multiplier from the S2 path."""
        try:
            scalar = float(self.state.get("thinker_risk_scalar", 1.0) or 1.0)
        except Exception:
            return 1.0
        return max(RISK_FLOOR, min(RISK_CEIL, scalar))

    def _news_bias(self, ticker: str) -> float:
        """Bounded live-news sentiment bias (System 2 evidence, ±0.03).

        Reads the news organ's published sentiment from shared state —
        the fast path never touches the network.
        """
        try:
            sent = self.state.get("news_sentiment", {}) or {}  # array-safe: dict-typed
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
        cross: float = 0.0,
    ) -> dict[str, Any]:
        """Compute the blended stabilized score and decision intermediates."""
        thinker_mod = self._bounded_thinker_modifier()
        thinker_conf = self._bounded_thinker_confidence()
        self.cortex._threshold = self.dynamics.threshold
        base = self.cortex.evaluate(alpha, prior_top=self._realized.dynamic_prior_top())
        nash_pred = self.nash.predict(alpha, base.score, base.direction)
        nash_op = self._compute_nash_mod(nash_pred)
        neuro_score = self._compute_neuro_score(alpha, ticker)
        cal_adj = self._calibration_nudge(base.score)
        blended = (1 - NEURO_BLEND) * (base.score + cal_adj) + NEURO_BLEND * neuro_score
        advisor_delta = self._advisor.threshold_delta()
        news_bias = self._news_bias(ticker)
        raw = blended * regime_mul + halim + episodic + nash_op
        raw += news_bias + cross - advisor_delta + thinker_mod
        stabilized, dyn_reason, final_dir = self._stabilize(raw, nash_pred)
        confidence = max(0.05, min(0.95, base.confidence + thinker_conf))
        return {
            "base": base,
            "nash_pred": nash_pred,
            "nash_op": nash_op,
            "neuro_score": neuro_score,
            "confidence": confidence,
            "raw_score": raw,
            "stabilized": stabilized,
            "final_dir": final_dir,
            "dyn_reason": dyn_reason,
            "nash_win_prob": nash_pred.win_prob,
            "advisor_delta": advisor_delta,
            "thinker_mod": thinker_mod,
            "thinker_conf": thinker_conf,
        }

    def _calibration_nudge(self, score: float) -> float:
        """Prediction-error nudge (rebuild ``prediction_error_adjustment``).

        Realized band WR minus the predicted win prob, applied along the
        score's own direction so it shapes sizing + final_dir — never
        cortex verdicts (R1: cortex is the sole verdict emitter). When the
        opt-in flag is off, or data is thin, returns 0.0 → the live path is
        byte-identical until ``CALIBRATION_NUDGE_ENABLED`` is opted in.
        """
        adj = self._realized.calibration_adjustment(abs(score))
        return adj if score >= 0 else -adj

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
            "horizon_reason": ctx.get("horizon_reason", "classifier"),
            "regime_canon": ctx.get("regime_canon", "unknown"),
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
        self._last_horizon.setdefault(ticker, "scalp")
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
        regime: str | None = None,
        horizon: str | None = None,
        vol_pct: float = 0.5,
        exit_triggers: list[str] | None = None,
    ) -> None:
        """All learning routes through neuromorphic brain — evolved per trade.

        IRONYCLADE: ONLY real IB fills feed the learning system.
        """
        canon = regime or self._last_regime.get(ticker, "unknown")
        hz = horizon or self._last_horizon.get(ticker, "scalp")
        vp = self._last_vol_pct.get(ticker, vol_pct)
        if not self._ironclade_gate(ticker, source):
            return
        self.dynamics.adapt_threshold(self.memory.pred_error)
        self.memory.threshold = (
            self.dynamics.threshold
        )  # persist adapted threshold → survives restart
        self.exits.deregister(ticker)
        log.info("LEARN %s %s pnl=%.4f", ticker, "WIN" if won else "LOSS", pnl_pct)
        self.episodic.add(self._last_alpha.get(ticker, {}), pnl_pct)
        if self._last_alpha.get(ticker):
            self.nash.record_outcome(self._last_alpha[ticker], 0.0, won)
        if self._neuromorphic is not None:
            self._neuromorphic.learn_from_outcome(ticker, won, pnl_pct)
        self._learn_strategy_organs(ticker, won, pnl_pct, direction, canon, hz, vp)
        if exit_triggers is not None:
            self._learned_exit.record(0.0, exit_triggers, won)
        self._learn_from_real(ticker, won, pnl_pct, direction, canon)

    def _ironclade_gate(self, ticker: str, source: str) -> bool:
        """Block non-IB sources from feeding the learning system."""
        if source in _IRONYCLADE:
            return True
        log.info("LEARN BLOCKED %s source=%s", ticker, source)
        self.exits.deregister(ticker)
        return False

    def _learn_strategy_organs(
        self,
        ticker: str,
        won: bool,
        pnl_pct: float,
        direction: int,
        canon: str,
        hz: str,
        vp: float,
    ) -> None:
        """Feed every strategy organ from one close (bounded, advisory).

        Meta-label (sizing), horizon bandit (selection), per-regime
        weights (scoring) — the realized loop stays IRONYCLADE-gated;
        these organs learn from every close.
        """
        self._meta.record(
            meta_features(
                self._last_conf.get(ticker, 0.5),
                self._last_score.get(ticker, 0.0),
                vp,
                canon,
                hz,
            ),
            won,
        )
        self._bandit.update(canon, hz, pnl_pct)
        self._regime_weights.learn(
            canon, self._last_alpha.get(ticker, {}), won, direction
        )

    def _learn_from_real(
        self, ticker: str, won: bool, pnl_pct: float, direction: int, regime: str
    ) -> None:
        """The closed learning loop — runs once per REAL trade close.

        (a) Weight gradient over ALL 27 indicators (loss-aversion 1.2x via
        Reflector — the single writer for weights/episodes/calibration).
        (b) Hot-swap learned weights into the cortex so the very next tick
        scores with the updated brain; per-regime vector blended when its
        regime is trained (REGIME_MIN_TRADES real closes).
        (c) Realized band/RR/conf-bins + calibration (the realized-EV gate).
        (d) Learned exits + gate advisor retune from the new sample.
        """
        conf = self._last_conf.get(ticker, 0.5)
        score = self._last_score.get(ticker, 0.0)
        # predicted_score is score_to_win_prob (win-prob), not sizing confidence (was: conf).
        self._reflector.on_trade_close(
            TradeClose(
                ticker=ticker,
                won=won,
                pnl_pct=pnl_pct,
                direction=direction,
                alpha=self._last_alpha.get(ticker, {}),
                predicted_score=score_to_win_prob(score),
                regime=regime,
            )
        )
        weights = self._weights_for(regime)
        self._wversion += 1
        self._apply_regime_weights(regime)
        # memory.record_outcome + update_pred_error are owned by the
        # Reflector above (single-writer); _realized/advisor/exits write
        # their own stores below.
        self._realized.add_outcome(score, won, pnl_pct, direction)
        self._realized.add_confidence_outcome(conf, won)
        self.exits.adapt_from_realized(self._realized)
        self._advisor.record_outcome(won)
        # Apply any pending HALIM recommendations (fetched by consolidation)
        self._apply_halim_recommendations()

    def _apply_halim_recommendations(self) -> None:
        """Apply pending HALIM recommendations to Juli parameters."""
        from .halim_recommendations import apply_recommendation

        recs = self.state.get("halim_recommendations", [])
        if not recs:
            return
        self.state.update(halim_recommendations=[])  # consume
        for rec in recs:
            try:
                apply_recommendation(rec, self.dynamics, self.memory)
            except Exception as e:
                log.warning("HALIM rec apply failed: %s", e)

    def _learned_exit_trade_count(self) -> int:
        """Real exits recorded by the learned-exit attributor."""
        return self._learned_exit.count

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
        self,
        ticker: str,
        current_price: float,
        ib_pnl: float = 0.0,
        direction: int = 1,
        *,
        exit_likelihood: float = 0.0,
        win_rate: float = 0.5,
        stop_price: Optional[float] = None,
        force_exit: bool = False,
    ) -> ExitSignal:
        """Check if position should be exited via the 3-tier exit ladder.

        Exit-likelihood not passed is derived from the pillar signal;
        win_rate not passed falls back to the realized win rate. TIER1
        stays dormant until a stop_price/force_exit is supplied.
        """
        return self._exit_ladder.evaluate(
            ticker,
            current_price,
            ib_pnl,
            direction,
            exit_likelihood=exit_likelihood,
            win_rate=win_rate,
            stop_price=stop_price,
            force_exit=force_exit,
        )

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
            "meta_label": self._meta.snapshot(),
            "horizon_bandit": self._bandit.snapshot(),
            "regime_weights": self._regime_weights.snapshot(),
            "learned_exit": {"trades": self._learned_exit_trade_count()},
            "genome": self.genome.get_genome(),
        }
        if self._sleep_engine is not None:
            result["sleep_engine"] = {
                "initialized": True,
                "cycle_count": self._sleep_engine._cycle_count,
            }
        return result

    def reset_learning(self) -> None:
        """Reset all learning state to clean defaults.

        Called after detecting poisoned data from synthetic/reconciled trades
        that fed the learning system before ironclade protection was active.
        Clears: episodic memory, nash patterns, realized-EV stats, meta-label,
        horizon bandit, regime weights, learned exits, and indicator weights.
        """
        log.warning("IRONCLADE: resetting all learning state (poisoned data cleanup)")
        # Episodic k-NN memory
        self.episodic.clear()
        # Nash pattern brain
        self.nash = NashBrain()
        # Realized-EV stats (band/conf bins)
        self._realized.reset()
        # Meta-label model
        self._meta = MetaLabelModel()
        # Horizon bandit
        self._bandit = HorizonBandit()
        # Per-regime weights
        self._regime_weights = RegimeWeights()
        # Learned exit policy
        self._learned_exit = LearnedExitPolicy()
        # Gate advisor
        self._advisor = GateAdvisor()
        # Reset indicator weights to immune defaults
        from ..immune import INDICATOR_WEIGHTS

        self.memory.set_weights(INDICATOR_WEIGHTS)
        self.cortex.set_weights(INDICATOR_WEIGHTS)
        # Clear decision context
        self._last_alpha.clear()
        self._last_score.clear()
        self._last_conf.clear()
        self._last_regime.clear()
        self._last_vol_pct.clear()
        self._last_horizon.clear()
        self._decision_count = 0
        log.warning("IRONCLADE: learning state reset complete")


JuliBrain = NeuromorphicBrain  # Backwards compat alias
__all__ = ["NeuromorphicBrain", "JuliBrain"]
