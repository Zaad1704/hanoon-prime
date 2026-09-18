"""hanoon_prime.brain.neurons.bridge — Neuromorphic Bridge interface.

Wires spiking neural network computation into JULI's decision pipeline.
Provides process_alpha() for scoring and apply_outcome() for learning.

R1 COMPLIANT: Outputs scores only, never verdict strings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adaptive_thresholds_wiring import feed_market_env, snapshot_base_thresholds
from .attractor import AttractorMemory
from .bridge_scoring import InputEncoder, ScoreComputer
from .constants import ALPHA_KEYS
from .lif import LIFNeuron
from .moe_config import DECISION_NEURONS, HIDDEN_NEURONS
from .moe_gate import route_experts
from .network import LIFNetwork
from .stdp import STDPLearner
from .threshold_adapter import DynamicThresholdAdapter
from .weights_config import get_default_synaptic_weights

log = logging.getLogger(__name__)

DECISION_THRESHOLD: float = 0.65


class NeuromorphicBridge:
    """Bridge between alpha inputs and decision outputs via neuromorphic computation."""

    def __init__(self, memory_path: Optional[Path] = None) -> None:
        self._network = LIFNetwork()
        self._stdp = STDPLearner()
        self._memory = AttractorMemory(filepath=memory_path)
        self._threshold_adapter = DynamicThresholdAdapter()
        self._initialized = False
        self._last_scores: Dict[str, float] = {}
        self._base_neuron_thresholds: Dict[str, float] = {}
        self._build_network()

    @staticmethod
    def _memory_path() -> Path:
        """Default persist location for attractors (lazy import mirrors
        adaptive_thresholds._persist_path; avoids import cycles)."""
        from ..config import STATE_DIR

        return STATE_DIR / "attractor_memory.json"

    def update_market_env(
        self,
        ticker: str,
        prices: list[float],
        vix: float | None = None,
    ) -> None:
        """Feed market context for adaptive firing thresholds (gated)."""
        feed_market_env(self, ticker, prices, vix)

    def feed_context(
        self,
        ticker: str,
        prices: list[float],
        regime: str = "unknown",
        regime_mul: float = 1.0,
        vix: float | None = None,
    ) -> None:
        """One per-tick market feed: adaptive thresholds + MoE routing."""
        feed_market_env(self, ticker, prices, vix)
        route_experts(self, regime, regime_mul)

    def _build_network(self) -> None:
        """Build the neuron network topology with input, hidden, decision neurons."""
        for key in ALPHA_KEYS:
            suffix = "bull" if "bull" in key else "bear"
            neuron = LIFNeuron(id=f"{suffix}_{key}", tau=0.05, threshold=0.7)
            self._network.add_neuron(neuron)

        for hid in HIDDEN_NEURONS:
            neuron = LIFNeuron(id=hid, tau=0.1, threshold=0.7)
            self._network.add_neuron(neuron)
            self._stdp.create_synapse(neuron.id, "decision_hold", 0.1)

        for dec in DECISION_NEURONS:
            neuron = LIFNeuron(id=dec, tau=0.05, threshold=0.6)
            self._network.add_neuron(neuron)

        self._wire_synapses()
        self._register_all_synapses_for_plasticity()
        self._network.reset_settle_counts()  # Initialize decision evidence tracking
        self._base_neuron_thresholds = snapshot_base_thresholds(self._network)
        self._initialized = True

    def _register_all_synapses_for_plasticity(self) -> None:
        """Mirror every wired synapse into the STDP store so reward and sleep
        replay can consolidate the full input→hidden→decision path (not just
        the hidden→hold synapses created in :meth:`_build_network`)."""
        for src_id, targets in self._network._weights.items():
            for dst_id, w in targets.items():
                self._stdp.create_synapse(src_id, dst_id, w)

    def _wire_synapses(self) -> None:
        """Wire default synaptic connections from config."""
        for key, w in get_default_synaptic_weights().items():
            parts = key.split(",")
            if len(parts) == 2:
                self._network.connect(parts[0], parts[1], w)

    def process_alpha(
        self, alpha: Dict[str, float], ticker: str = ""
    ) -> Dict[str, Any]:
        """Process alpha through the network. Returns decision score."""
        if not self._initialized:
            return {"score": 0.0, "confidence": 0.5, "trace": {}}

        InputEncoder.encode(self._network, alpha)
        spikes = self._network.step_all(dt=0.05)
        evidence = self._network._decision_evidence

        score = ScoreComputer.compute(evidence)
        confidence = ScoreComputer.confidence(evidence, score)
        self._last_scores[ticker] = score

        return {
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "trace": {"spikes": len(spikes), "evidence": dict(evidence)},
        }

    def learn_from_outcome(self, _ticker: str, won: bool, pnl: float) -> None:
        """Apply STDP learning from trade outcome, scaled by realized P&L.

        Gated by NEURO_LEARN_ENABLED: plasticity collects in the STDP store
        as usual (inert on scoring), but only syncs into the fast-path
        ``LIFNetwork._weights`` dict — the store the scorer actually reads —
        once the flag and its backtest gate clear. The eligibility trace is
        each Synapse.trace: accumulated per-spike Δw, decayed by
        :meth:`STDPLearner.update_traces`, and gated by
        :meth:`STDPLearner.apply_reward` (distal reward via trace > 0.1).
        """
        from ...immune import NEURO_LEARN_ENABLED

        reward = (1.0 if won else -1.0) * (1.0 + min(1.0, abs(pnl)))
        self._stdp.apply_reward(reward)
        self._stdp.update_traces(dt=0.01)
        if NEURO_LEARN_ENABLED:
            self._sync_plasticity_to_network()

    def _sync_plasticity_to_network(self) -> None:
        """Write STDP-learned strengths back into the fast-path _weights."""
        for (src_id, dst_id), syn in self._stdp._synapses.items():
            targets = self._network._weights.setdefault(src_id, {})
            targets[dst_id] = syn.strength

    def snapshot(self) -> Dict[str, Any]:
        """Full state snapshot for telemetry."""
        return {
            "network": self._network.snapshot(),
            "memory_size": len(self._memory),
            "initialized": self._initialized,
        }

    @property
    def threshold(self) -> float:
        """Current decision threshold."""
        return DECISION_THRESHOLD

    @property
    def memory(self) -> AttractorMemory:
        """Access the attractor memory."""
        return self._memory

    def store_outcome(
        self, ticker: str, pattern: List[float], won: bool, pnl_pct: float
    ) -> None:
        """Store trade outcome for memory."""
        self._memory.store(ticker, pattern, won, pnl_pct)


def create_bridge() -> NeuromorphicBridge:
    """Factory function for bridge creation."""
    return NeuromorphicBridge()


__all__ = [
    "NeuromorphicBridge",
    "create_bridge",
    "DECISION_THRESHOLD",
]
