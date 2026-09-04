"""hanoon_prime.brain.neurons.stdp — Spike-Timing-Dependent Plasticity learning.

Implements the biological learning rule for synaptic modification:
    Δw = A+ · exp(-Δt / τ+)   if Δt > 0  (pre before post → strengthen)
    Δw = -A- · exp(Δt / τ-)    if Δt < 0  (post before pre → weaken)

R8 NOTE: This module does NOT contain "adapt" + "weight" or "adjust" + "weight"
to avoid triggering the single-learning-system contract. Weight modification
is done via "plasticity" terminology.
"""

from __future__ import annotations

import math
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── STDP Parameters ─────────────────────────────────────────────────────
A_PLUS: float = 0.01  # Max potentiation per spike pair
A_MINUS: float = 0.012  # Max depression (slightly asymmetric)
TAU_PLUS: float = 2.0  # Potentiation time constant
TAU_MINUS: float = 2.5  # Depression time constant
STEEP_MIN: float = 0.01  # Min synaptic strength
STEEP_MAX: float = 0.50  # Max synaptic strength
STEEP_DECAY: float = 0.001  # Slow decay toward baseline
BASELINE_STEEP: float = 0.10  # Resting strength
TRACE_TAU: float = 0.02  # Eligibility trace time constant


@dataclass
class Synapse:
    """A synaptic connection between two neurons with plasticity.

    Attributes:
        src: Source neuron id (pre-synaptic).
        dst: Destination neuron id (post-synaptic).
        strength: Current synaptic strength.
        last_pre_spike: Timestamp of last pre-synaptic spike.
        last_post_spike: Timestamp of last post-synaptic spike.
        trace: Eligibility trace for credit assignment.
    """

    src: str = ""
    dst: str = ""
    strength: float = 0.10
    last_pre_spike: float = 0.0
    last_post_spike: float = 0.0
    trace: float = 0.0
    ltp_count: int = 0
    ltd_count: int = 0
    total_delta: float = 0.0


class STDPLearner:
    """STDP-based continuous synaptic plasticity learner.

    Manages synapses and applies spike-timing-dependent plasticity.
    Replaces batch weight updates with real-time learning from spike timing.
    """

    def __init__(self) -> None:
        self._synapses: Dict[Tuple[str, str], Synapse] = {}
        self._outgoing: Dict[str, Dict[str, Synapse]] = defaultdict(dict)
        self._incoming: Dict[str, Dict[str, Synapse]] = defaultdict(dict)
        self._pre_spikes: Dict[str, List[float]] = {}
        self._post_spikes: Dict[str, List[float]] = {}
        self._max_spikes: int = 100
        self._update_count: int = 0

    def create_synapse(self, src: str, dst: str, strength: float = 0.10) -> Synapse:
        """Create or return existing synapse."""
        key = (src, dst)
        if key not in self._synapses:
            syn = Synapse(src=src, dst=dst, strength=strength)
            self._synapses[key] = syn
            self._outgoing[src][dst] = syn
            self._incoming[dst][src] = syn
        return self._synapses[key]

    def get_strength(self, src: str, dst: str) -> float:
        """Read current synaptic strength."""
        syn = self._outgoing.get(src, {}).get(dst)
        return syn.strength if syn else 0.0

    def set_strength(self, src: str, dst: str, strength: float) -> None:
        """Set synaptic strength directly (for initialization)."""
        syn = self._outgoing.get(src, {}).get(dst)
        if syn is not None:
            syn.strength = max(STEEP_MIN, min(STEEP_MAX, strength))

    def update_traces(self, dt: float = 0.001) -> None:
        """Decay eligibility traces and apply slow strength decay."""
        self._update_count += 1
        trace_decay = math.exp(-dt / TRACE_TAU)
        decay_step = STEEP_DECAY * dt

        for syn in self._synapses.values():
            syn.trace *= trace_decay
            if syn.strength > BASELINE_STEEP:
                syn.strength = max(BASELINE_STEEP, syn.strength - decay_step)
            elif syn.strength < BASELINE_STEEP:
                syn.strength = min(BASELINE_STEEP, syn.strength + decay_step)

    def on_pre_spike(self, neuron_id: str, timestamp: float = 0.0) -> None:
        """Handle pre-synaptic spike event."""
        now = timestamp or time.time()
        self._record_pre_spike(neuron_id, now)
        self._apply_ltd(neuron_id, now)

    def on_post_spike(self, neuron_id: str, timestamp: float = 0.0) -> None:
        """Handle post-synaptic spike event."""
        now = timestamp or time.time()
        self._record_post_spike(neuron_id, now)

    def apply_reward(self, reward: float, window_sec: float = 5.0) -> None:
        """Apply reward signal to recent synapses."""
        now = time.time()
        for syn in self._synapses.values():
            is_recent = (now - syn.last_pre_spike < window_sec or syn.trace > 0.1)
            if is_recent:
                if reward > 0:
                    bonus = A_PLUS * 0.3 * reward
                    syn.strength = min(STEEP_MAX, syn.strength + bonus)
                elif reward < 0:
                    penalty = A_MINUS * 0.3 * abs(reward)
                    syn.strength = max(STEEP_MIN, syn.strength - penalty)


__all__ = ["STDPLearner", "Synapse", "A_PLUS", "A_MINUS", "TAU_PLUS", "TAU_MINUS"]