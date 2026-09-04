"""hanoon_prime.brain.neurons.lif — Leaky Integrate-and-Fire neuron model.

Implements the biological neuron model:
    V_j(t + Δt) = V_j(t) · e^(-Δt / τ) + Σ_i w_{i,j} · s_i(t)

Neurons only fire when V_m >= threshold, then enter a refractory period.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .spike import Spike


@dataclass
class LIFNeuron:
    """A single Leaky Integrate-and-Fire neuron.

    Attributes:
        id: Unique neuron identifier.
        tau: Membrane time constant (decay rate) in seconds.
        threshold: Firing threshold θ. When V_m >= θ, neuron spikes.
        rest_potential: Resting membrane potential (typically 0.0).
        reset_potential: Potential after a spike (typically 0.0).
        refractory_period: Minimum seconds between spikes.
        weight: Input synaptic weight.
    """

    id: str = ""
    tau: float = 0.05  # 50ms default membrane time constant
    threshold: float = 0.7  # Fire when V_m >= 0.7
    rest_potential: float = 0.0
    reset_potential: float = 0.0
    refractory_period: float = 0.001  # 1ms refractory
    weight: float = 1.0

    # Runtime state
    _v_m: float = field(default=0.0, init=False, repr=False)
    _last_spike_time: float = field(default=0.0, init=False, repr=False)
    _spike_count: int = field(default=0, init=False, repr=False)
    _input_current: float = field(default=0.0, init=False, repr=False)
    _trace: float = field(default=0.0, init=False, repr=False)
    _last_update: float = field(default=0.0, init=False, repr=False)
    _synaptic_current: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize runtime state."""
        self._v_m = self.rest_potential
        self._last_update = time.time()

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def v_m(self) -> float:
        """Current membrane potential."""
        return self._v_m

    @property
    def is_refractory(self) -> bool:
        """Check if neuron is in refractory period."""
        return (time.time() - self._last_spike_time) < self.refractory_period

    @property
    def spike_rate(self) -> float:
        """Spike rate (Hz) over the last second."""
        if self._spike_count == 0:
            return 0.0
        return min(self._spike_count / 1.0, 100.0)

    # ── Input Methods ───────────────────────────────────────────────────

    def set_input(self, current: float) -> None:
        """Set input current for next step (scaled for threshold/tau)."""
        scaled = current * 2.0 * (self.threshold / self.tau)
        self._input_current = scaled

    def inject_charge(self, charge: float) -> None:
        """Accumulate synaptic current from a post-synaptic spike."""
        if self.is_refractory:
            return
        scaled = charge * 2.0 * (self.threshold / self.tau)
        self._synaptic_current += scaled
        self._trace += charge

    # ── Stepping ────────────────────────────────────────────────────────

    def step(self, dt: float = 0.05) -> Optional[Spike]:
        """Advance one time step. Returns Spike if neuron fires.

        Args:
            dt: Simulation timestep in seconds (default 50ms).

        Returns:
            Spike object if neuron fired, None otherwise.
        """
        if self.is_refractory:
            return None

        now = time.time()
        total_current = self._input_current + self._synaptic_current
        self._synaptic_current = 0.0

        # Substep integration for accuracy with small tau
        nsub = max(1, int(math.ceil(dt / (self.tau * 0.5))))
        sub_dt = dt / nsub
        decay_factor = math.exp(-sub_dt / self.tau)

        for _ in range(nsub):
            self._v_m = self._v_m * decay_factor + self.weight * total_current * sub_dt

        # Trace decay
        self._trace *= math.exp(-dt / (self.tau * 5))

        # Spike-count leak
        if self._spike_count > 0:
            self._spike_count *= math.exp(-dt / 1.0)
            if self._spike_count < 0.01:
                self._spike_count = 0

        return self._check_fire(now)

    def _check_fire(self, now: float) -> Optional[Spike]:
        """Check if threshold reached and spike if so."""
        if self._v_m >= self.threshold:
            amplitude = min(self._v_m / self.threshold, 2.0)
            spike = Spike(
                neuron_id=self.id,
                timestamp=now,
                amplitude=amplitude,
                trace=self._trace,
            )
            self._reset()
            self._spike_count += 1
            return spike
        return None

    def _reset(self) -> None:
        """Reset after a spike."""
        self._v_m = self.reset_potential
        self._input_current = 0.0
        self._trace = 0.0
        self._last_spike_time = time.time()

    # ── Telemetry ──────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Current state for telemetry."""
        return {
            "id": self.id,
            "v_m": round(self._v_m, 4),
            "threshold": self.threshold,
            "tau": self.tau,
            "refractory": self.is_refractory,
            "trace": round(self._trace, 4),
            "spike_count": self._spike_count,
        }


__all__ = ["LIFNeuron", "Spike"]
