"""hanoon_prime.brain.neurons.network_step — Vectorized LIF stepping.

High-performance inner loop for membrane potential integration.
Uses numpy vectorization for O(n) stepping of all neurons.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

import numpy as np

from .lif import LIFNeuron
from .network import LIFNetwork
from .spike import Spike


class NetworkStepper:
    """Vectorized stepping of LIF neurons using numpy operations."""

    def __init__(self, network: LIFNetwork) -> None:
        self._network = network
        self._decay_cache: dict = {}
        self._last_step_time: float = 0.0

    def step_all(self, dt: float = 0.05) -> List[Spike]:
        """Advance all neurons one step. Returns all spikes produced."""
        now = time.time()
        network = self._network

        # Phase 1: State extraction
        v_m, inputs, synaptic, refractory = self._extract_state(now)

        # Phase 2: Vectorized integration
        total_current = inputs + synaptic
        spikes = self._integrate_and_spike(v_m, total_current, refractory, now)

        # Phase 3: Post-spike propagation
        self._propagate_spikes(spikes)

        return spikes

    def _extract_state(self, now: float) -> tuple:
        """Extract neuron states into numpy arrays."""
        neurons = self._network._neurons
        n = len(neurons)
        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        keys = list(neurons.keys())
        v_m = np.empty(n, dtype=np.float64)
        inputs = np.empty(n, dtype=np.float64)
        synaptic = np.empty(n, dtype=np.float64)
        refractory = np.empty(n, dtype=bool)

        for i, nid in enumerate(keys):
            neuron = neurons[nid]
            v_m[i] = neuron._v_m
            inputs[i] = neuron._input_current
            synaptic[i] = neuron._synaptic_current
            refractory[i] = (now - neuron._last_spike_time) < neuron.refractory_period

        return v_m, inputs, synaptic, refractory

    def _integrate_and_spike(
        self,
        v_m: np.ndarray,
        total_current: np.ndarray,
        refractory: np.ndarray,
        now: float,
    ) -> List[Spike]:
        """Vectorized LIF integration and spike detection."""
        if len(v_m) == 0:
            return []

        neurons = self._network._neurons
        keys = list(neurons.keys())
        weights = np.array([neurons[k].weight for k in keys])
        thresholds = np.array([neurons[k].threshold for k in keys])

        # Substep integration
        dt = 0.05
        tau = np.array([neurons[k].tau for k in keys])
        nsub = max(1, int(np.ceil(dt / (tau * 0.5)).max()))
        decay = np.exp(-dt / nsub / tau)

        active = ~refractory
        active_idx = np.where(active)[0]

        for _ in range(nsub):
            if len(active_idx) == 0:
                break
            v_m[active_idx] *= decay[active_idx]
            v_m[active_idx] += (
                weights[active_idx] * total_current[active_idx] * dt / nsub
            )
            still_active = np.array([i in active_idx for i in range(len(v_m))])
            active_idx = active_idx[still_active[active_idx]]

        return self._detect_spikes(v_m, thresholds, keys, now)

    def _detect_spikes(
        self,
        v_m: np.ndarray,
        thresholds: np.ndarray,
        keys: List[str],
        now: float,
    ) -> List[Spike]:
        """Detect which neurons have spiked."""
        spikes = []
        neurons = self._network._neurons

        fired = v_m >= thresholds
        for i, fired_flag in enumerate(fired):
            if not fired_flag:
                continue
            nid = keys[i]
            neuron = neurons[nid]

            amplitude = min(neuron._v_m / neuron.threshold, 2.0)
            spike = Spike(
                neuron_id=nid,
                timestamp=now,
                amplitude=amplitude,
                trace=neuron._trace,
            )
            spikes.append(spike)

            # Update neuron state
            neuron._v_m = neuron.reset_potential
            neuron._last_spike_time = now
            neuron._spike_count += 1

        return spikes

    def _propagate_spikes(self, spikes: List[Spike]) -> None:
        """Propagate spikes to post-synaptic targets."""
        network = self._network
        synapses = network._synapses
        weights = network._weights

        for spike in spikes:
            network._spike_log.append(spike)
            targets = synapses.get(spike.neuron_id, [])
            syn_weights = weights.get(spike.neuron_id, {})

            for target_id in targets:
                self._inject_to_target(
                    network, target_id, syn_weights, spike.amplitude
                )

    def _inject_to_target(
        self, network: LIFNetwork, target_id: str,
        syn_weights: dict[str, float], amplitude: float
    ) -> None:
        """Inject charge to a single target neuron."""
        target = network._neurons.get(target_id)
        if target is None:
            return
        strength = syn_weights.get(target_id, 0.5)
        target.inject_charge(strength * amplitude)

        if target_id in network._decision_evidence:
            network._decision_evidence[target_id] += strength * amplitude


__all__ = ["NetworkStepper"]
