"""hanoon_prime.brain.neurons — Neuromorphic computing core.

Leaky Integrate-and-Fire (LIF) spiking neural network with:
  - Volumetric encoding (ticker → membrane potential)
  - Spike-timing-dependent plasticity (STDP) for continuous learning
  - Attractor memory for pattern recall
  - Sleep replay for offline consolidation

R1 COMPLIANT: Only Cortex produces verdicts — this module outputs scores only.
"""

from __future__ import annotations

from .attractor import Attractor, AttractorMemory
from .bridge import NeuromorphicBridge, create_bridge
from .lif import LIFNeuron
from .moe_config import (
    DECISION_NEURONS,
    HIDDEN_NEURONS,
    LIQUIDITY_HIDDEN,
    MEANREV_HIDDEN,
    MOMENTUM_HIDDEN,
    NETWORK_ARCHITECTURE,
)
from .network import LIFNetwork
from .sleep import SleepReplayEngine, SleepResult
from .spike import Spike
from .stdp import STDPLearner, Synapse
from .threshold_adapter import DynamicThresholdAdapter
from .weights_config import get_default_synaptic_weights, get_network_topology

__all__ = [
    "Spike",
    "LIFNeuron",
    "LIFNetwork",
    "STDPLearner",
    "Synapse",
    "AttractorMemory",
    "Attractor",
    "SleepReplayEngine",
    "SleepResult",
    "DynamicThresholdAdapter",
    "get_default_synaptic_weights",
    "get_network_topology",
    "MOMENTUM_HIDDEN",
    "MEANREV_HIDDEN",
    "LIQUIDITY_HIDDEN",
    "HIDDEN_NEURONS",
    "DECISION_NEURONS",
    "NETWORK_ARCHITECTURE",
    "NeuromorphicBridge",
    "create_bridge",
]
