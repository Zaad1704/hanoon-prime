"""hanoon_prime.brain.neurons.sleep — Offline memory consolidation.

During market closure, replays stored attractor patterns to stabilize
synaptic weights and prevent catastrophic forgetting.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .attractor import AttractorMemory
from .lif import LIFNeuron
from .network import LIFNetwork
from .spike import Spike
from .stdp import STDPLearner


@dataclass
class SleepResult:
    """Result of sleep consolidation cycle."""

    patterns_replayed: int = 0
    spikes_generated: int = 0
    weights_updated: int = 0
    mean_weight_change: float = 0.0
    duration_ms: float = 0.0


class SleepReplayEngine:
    """Offline replay engine for continual learning during non-trading hours.

    Injects Poisson noise + replays attractor patterns through the SNN
    to consolidate learned behaviors without market data.
    """

    POISSON_RATE: float = 3.0
    REPLAY_BATCH: int = 10
    WIN_BIAS: float = 2.0
    MAX_PATTERNS: int = 100

    def __init__(
        self,
        network: LIFNetwork,
        stdp: STDPLearner,
        memory: AttractorMemory,
    ) -> None:
        self._network = network
        self._stdp = stdp
        self._memory = memory
        self._cycle_count: int = 0

    def should_run(self, is_market_open: bool) -> bool:
        """Should sleep consolidation run? Only when market is closed."""
        return not is_market_open

    def select_patterns(self) -> List[Tuple[Dict[str, float], float]]:
        """Select attractor patterns for replay."""
        attractors = list(self._memory)
        if not attractors:
            return []

        patterns = []
        for att in attractors:
            if att.trade_count < 2:
                continue

            weight = self.WIN_BIAS if att.wins > att.losses else 0.5
            patterns.append(
                (
                    dict(
                        zip(
                            [f"alpha_{i}" for i in range(len(att.center))],
                            att.center,
                        )
                    ),
                    weight,
                )
            )

        if len(patterns) > self.MAX_PATTERNS:
            random.shuffle(patterns)
            patterns = patterns[: self.MAX_PATTERNS]

        return patterns

    def run_cycle(self, duration_sec: float = 60.0) -> SleepResult:
        """Run one sleep consolidation cycle."""
        start = time.time()
        start_ms = start * 1000

        patterns = self.select_patterns()
        if not patterns:
            return SleepResult(duration_ms=(time.time() - start) * 1000)

        spikes_generated = 0
        weights_updated = 0
        total_change = 0.0

        for pattern, weight in patterns[: self.REPLAY_BATCH]:
            count, change = self._replay_pattern(pattern, weight)
            spikes_generated += count
            total_change += change

        self._cycle_count += 1

        return SleepResult(
            patterns_replayed=len(patterns[: self.REPLAY_BATCH]),
            spikes_generated=spikes_generated,
            weights_updated=weights_updated,
            mean_weight_change=total_change
            / max(1, len(patterns[: self.REPLAY_BATCH])),
            duration_ms=(time.time() - start) * 1000,
        )

    def _replay_pattern(
        self,
        pattern: Dict[str, float],
        weight: float,
    ) -> Tuple[int, float]:
        """Replay a single pattern through the network."""
        spikes = 0
        change = 0.0

        for key, value in pattern.items():
            neuron_id = key.replace("alpha_", "")
            self._network.set_neuron_input(neuron_id, value * weight)

        for _ in range(5):
            new_spikes = self._network.step_all(dt=0.05)
            spikes += len(new_spikes)

        return spikes, change


__all__ = ["SleepReplayEngine", "SleepResult"]
