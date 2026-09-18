"""hanoon_prime.brain.neurons.replay — Network/STDP driving shared by sleep.

Holds the mechanics of pushing stored patterns back through the SNN so the
SleepReplayEngine can stay policy-and-bookkeeping. Mirrors the live
InputEncoder id convention: a plain indicator name drives ``bull_<name>_bull``
with +value and ``bear_<name>_bear`` with -value; legacy ``alpha_<i>`` keys
resolve through EPISODIC_KEYS order.
"""

from __future__ import annotations

import random
from typing import Dict, Tuple

from ..config import EPISODIC_KEYS
from .network import LIFNetwork
from .stdp import STDPLearner

SynapseKey = Tuple[str, str]
Snapshot = Dict[SynapseKey, float]


def encode_pattern(network: LIFNetwork, center: list[float]) -> Dict[str, float]:
    """Map a pattern center (EPISODIC_KEYS order) onto real input neurons.

    The two-sided convention (bear side = -value) matches the live encoder, so
    replay reinforces the exact wiring the network would have used in-session.
    """
    pattern: Dict[str, float] = {}
    for i, value in enumerate(center):
        if i >= len(EPISODIC_KEYS):
            break
        name = EPISODIC_KEYS[i]
        bull_id, bear_id = f"bull_{name}_bull", f"bear_{name}_bear"
        if bull_id in network._neurons:
            pattern[bull_id] = float(value)
        if bear_id in network._neurons:
            pattern[bear_id] = -float(value)
    return pattern


def apply_pattern_input(
    network: LIFNetwork,
    pattern: Dict[str, float],
    weight: float,
) -> None:
    """Set input currents for a recorded pattern (indexed ids still work)."""
    for key, value in pattern.items():
        if key.startswith("alpha_"):
            _apply_indexed_input(network, key, value, weight)
        else:
            network.set_neuron_input(key, value * weight)


def _apply_indexed_input(
    network: LIFNetwork,
    key: str,
    value: float,
    weight: float,
) -> None:
    """Back-compat: resolve a legacy ``alpha_<i>`` key to real neurons."""
    try:
        idx = int(key.split("_", 1)[1])
    except (ValueError, IndexError):
        return
    if idx >= len(EPISODIC_KEYS):
        return
    name = EPISODIC_KEYS[idx]
    bull_id, bear_id = f"bull_{name}_bull", f"bear_{name}_bear"
    if bull_id in network._neurons:
        network.set_neuron_input(bull_id, value * weight)
    if bear_id in network._neurons:
        network.set_neuron_input(bear_id, -value * weight)


def inject_poisson(network: LIFNetwork, rate: float) -> None:
    """Spontaneous background current (replay literature needs it to drive
    the very synapses being consolidated). Only quiet input neurons, tiny
    currents, one shot per call."""
    prob = min(1.0, rate * 0.05)
    for nid, neuron in network._neurons.items():
        if not nid.startswith(("bull_", "bear_")):
            continue
        if getattr(neuron, "_input_current", 0.0) == 0.0 and random.random() < prob:
            network.set_neuron_input(nid, random.uniform(0.05, 0.3))


def drive_steps(
    network: LIFNetwork,
    stdp: STDPLearner,
    steps: int,
    poisson_rate: float,
    dt: float = 0.05,
) -> int:
    """Run ``steps`` simulation ticks, feeding each spike to STDP.

    Returns the total spike count. Trace decay runs every tick so pre/post
    windows close as they would live.
    """
    spikes = 0
    for _ in range(steps):
        inject_poisson(network, poisson_rate)
        new_spikes = network.step_all(dt=dt)
        for sp in new_spikes:
            spikes += 1
            stdp.on_pre_spike(sp.neuron_id, sp.timestamp)
            stdp.on_post_spike(sp.neuron_id, sp.timestamp)
        stdp.update_traces(dt=dt)
    return spikes


def snapshot_synapses(stdp: STDPLearner) -> Snapshot:
    """Current (src, dst) -> strength across the STDP store."""
    return {(s.src, s.dst): s.strength for s in stdp._synapses.values()}


def synapse_delta_stats(before: Snapshot, stdp: STDPLearner) -> Tuple[int, float]:
    """(synapses changed, total abs change) versus a prior snapshot."""
    updated = 0
    change = 0.0
    after = snapshot_synapses(stdp)
    for syn in before:
        delta = after.get(syn, 0.0) - before.get(syn, 0.0)
        if abs(delta) > 1e-9:
            updated += 1
            change += abs(delta)
    return updated, change


__all__ = [
    "apply_pattern_input",
    "drive_steps",
    "encode_pattern",
    "inject_poisson",
    "snapshot_synapses",
    "synapse_delta_stats",
]
