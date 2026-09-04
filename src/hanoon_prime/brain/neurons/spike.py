"""hanoon_prime.brain.neurons.spike — Spike event dataclass.

A spike event represents a neuron firing, containing the neuron ID,
timestamp, amplitude (for rate coding), and eligibility trace for STDP.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Spike:
    """A spike event produced by a neuron when it fires.

    Attributes:
        neuron_id: Unique identifier of the firing neuron.
        timestamp: Unix timestamp when the spike occurred.
        amplitude: Spike strength in [0, 2], computed as V_m / threshold.
        trace: Eligibility trace for STDP weight updates.
    """

    neuron_id: str
    timestamp: float = field(default_factory=time.time)
    amplitude: float = 1.0  # Spike strength for rate coding
    trace: float = 0.0  # Eligibility trace for STDP

    def __post_init__(self) -> None:
        """Ensure timestamp is properly set."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


__all__ = ["Spike"]
