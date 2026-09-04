"""hanoon_prime.brain.neurons.bridge — Neuromorphic Bridge interface.

Wires spiking neural network computation into JULI's decision pipeline.
Provides process_alpha() for scoring and apply_outcome() for learning.

R1 COMPLIANT: Outputs scores only, never verdict strings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .attractor import AttractorMemory
from .bridge_scoring import InputEncoder, ScoreComputer
from .constants import ALPHA_KEYS
from .lif import LIFNeuron
from .moe_config import DECISION_NEURONS, HIDDEN_NEURONS
from .network import LIFNetwork
from .stdp import STDPLearner
from .threshold_adapter import DynamicThresholdAdapter
from .weights_config import get_default_synaptic_weights

log = logging.getLogger(__name__)

DECISION_THRESHOLD: float = 0.65


class NeuromorphicBridge:
    """Bridge between alpha inputs and decision outputs via neuromorphic computation."""

    def __init__(self) -> None:
        self._network = LIFNetwork()
        self._stdp = STDPLearner()
        self._memory = AttractorMemory()
        self._threshold_adapter = DynamicThresholdAdapter()
        self._initialized = False
        self._last_scores: Dict[str, float] = {}
        self._build_network()

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
        self._network.reset_settle_counts()  # Initialize decision evidence tracking
        self._initialized = True

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

    def learn_from_outcome(self, ticker: str, won: bool, pnl: float) -> None:
        """Apply STDP learning from trade outcome."""
        reward = 1.0 if won else -1.0
        self._stdp.apply_reward(reward)
        self._stdp.update_traces(dt=0.01)

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
