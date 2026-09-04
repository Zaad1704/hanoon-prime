"""hanoon_prime.brain — Juli's biological brain subsystem.

All cognitive modules live here. The orchestrator (juli.py) coordinates
the pipeline: indicators → regime → episodic → affective → salience →
deliberation → dynamics → risk → execution.

Enhanced with neuromorphic neurons for spiking computation.
"""

from __future__ import annotations

from .neurons import (
    Attractor,
    AttractorMemory,
    LIFNetwork,
    LIFNeuron,
    NeuromorphicBridge,
    SleepReplayEngine,
    SleepResult,
    STDPLearner,
    Synapse,
    create_bridge,
)
from .orchestrator import JuliBrain, NeuromorphicBrain

__all__ = [
    "LIFNeuron",
    "LIFNetwork",
    "STDPLearner",
    "Synapse",
    "AttractorMemory",
    "Attractor",
    "SleepReplayEngine",
    "SleepResult",
    "NeuromorphicBridge",
    "create_bridge",
    "NeuromorphicBrain",
    "JuliBrain",
]
