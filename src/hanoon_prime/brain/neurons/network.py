"""hanoon_prime.brain.neurons.network — LIF Network structure and connections.

Manages the spiking neural network topology: neurons, synapses, and
coordination with NetworkStepper for vectorized stepping.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional

from .lif import LIFNeuron
from .spike import Spike


class LIFNetwork:
    """Network of LIF neurons with synaptic connections."""

    def __init__(self) -> None:
        self._neurons: Dict[str, LIFNeuron] = {}
        self._synapses: Dict[str, List[str]] = {}
        self._weights: Dict[str, Dict[str, float]] = {}
        self._spike_log: deque = deque(maxlen=1000)
        self._settle_spike_counts: Dict[str, int] = {}
        self._decision_evidence: Dict[str, float] = {}

    def add_neuron(self, neuron: LIFNeuron) -> None:
        """Add a neuron to the network."""
        self._neurons[neuron.id] = neuron
        if neuron.id not in self._synapses:
            self._synapses[neuron.id] = []
        if neuron.id not in self._weights:
            self._weights[neuron.id] = {}

    def connect(self, src_id: str, dst_id: str, strength: float = 0.5) -> None:
        """Create a synaptic connection."""
        if src_id not in self._synapses:
            self._synapses[src_id] = []
        if dst_id not in self._synapses[src_id]:
            self._synapses[src_id].append(dst_id)
        if src_id not in self._weights:
            self._weights[src_id] = {}
        self._weights[src_id][dst_id] = strength

    def get_neuron(self, neuron_id: str) -> Optional[LIFNeuron]:
        """Get neuron by ID."""
        return self._neurons.get(neuron_id)

    def set_neuron_input(self, neuron_id: str, current: float) -> None:
        """Set input current for a neuron."""
        neuron = self._neurons.get(neuron_id)
        if neuron is not None:
            neuron.set_input(current)

    def reset_settle_counts(self) -> None:
        """Reset settlement counters for new cycle."""
        self._settle_spike_counts = {nid: 0 for nid in self._neurons}
        decision_ids = [nid for nid in self._neurons if nid.startswith("decision_")]
        self._decision_evidence = {nid: 0.0 for nid in decision_ids}

    def step_all(self, dt: float = 0.05) -> List[Spike]:
        """Advance all neurons one step. Returns all spikes produced."""
        now = time.time()
        spikes: List[Spike] = []
        neurons = self._neurons

        for nid, neuron in neurons.items():
            if neuron.is_refractory:
                continue
            result = neuron.step(dt)
            if result is not None:
                self._process_spike(nid, result, spikes)

        return spikes

    def _process_spike(self, nid: str, spike: Spike, spikes: List[Spike]) -> None:
        """Process a single spike: record, propagate, update evidence."""
        spikes.append(spike)
        self._spike_log.append(spike)
        self._settle_spike_counts[nid] = self._settle_spike_counts.get(nid, 0) + 1

        # Propagate to targets
        for target in self._synapses.get(nid, []):
            self._propagate_to_target(nid, target, spike)

    def _propagate_to_target(self, src_id: str, target_id: str, spike: Spike) -> None:
        """Propagate spike to a target neuron."""
        target_neuron = self._neurons.get(target_id)
        if target_neuron is None:
            return
        weight = self._weights.get(src_id, {}).get(target_id, 0.5)
        target_neuron.inject_charge(weight * spike.amplitude)
        if target_id in self._decision_evidence:
            self._decision_evidence[target_id] += weight * spike.amplitude

    def snapshot(self) -> dict:
        """Full network state for telemetry."""
        return {
            "neuron_count": len(self._neurons),
            "synapse_count": sum(len(v) for v in self._synapses.values()),
        }


__all__ = ["LIFNetwork", "LIFNeuron", "Spike"]
