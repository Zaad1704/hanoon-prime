"""hanoon_prime.brain.neurons.bridge_scoring — Neuromorphic input encoding.

Maps market indicators (alpha) to spike-encoded neuron inputs.
"""

from __future__ import annotations


def _set_input_for_alpha(network: dict, key: str, alpha: dict[str, float]) -> None:
    """Set input for a single alpha key."""
    if key == "market_regime":
        return
    if key.endswith("_bull"):
        _set_neuron_input(network, f"bull_{key}", alpha[key])
    elif key.endswith("_bear"):
        _set_neuron_input(network, f"bear_{key}", alpha[key])
    else:
        # Legacy non-directional keys
        _set_neuron_input(network, f"bull_{key}", alpha.get(key, 0.5))
        _set_neuron_input(network, f"bear_{key}", -alpha.get(key, 0.5))


def _set_neuron_input(network: dict, neuron_id: str, value: float) -> None:
    """Helper to set neuron input if it exists."""
    if neuron_id in network._neurons:
        network.set_neuron_input(neuron_id, value)


class InputEncoder:
    """Maps alpha dictionaries to neuron input currents."""

    @classmethod
    def encode(cls, network, alpha: dict[str, float]) -> None:
        """Set input currents on neurons based on alpha values."""
        for key in alpha.keys():
            _set_input_for_alpha(network, key, alpha)


class ScoreComputer:
    """Computes decision scores from neuron evidence."""

    @staticmethod
    def compute(evidence: dict[str, float]) -> float:
        """Compute normalized score [-1, 1] from decision evidence."""
        long_ev = evidence.get("decision_long", 0.0)
        short_ev = evidence.get("decision_short", 0.0)
        denom = max(abs(long_ev), abs(short_ev), 1.0)
        score = (long_ev - short_ev) / denom
        return max(-1.0, min(1.0, score))

    @staticmethod
    def confidence(evidence: dict[str, float], score: float) -> float:
        """Compute confidence from evidence magnitude."""
        total_ev = sum(abs(e) for e in evidence.values())
        raw_conf = min(1.0, total_ev / 10.0)
        return 0.5 + raw_conf * 0.45


__all__ = ["InputEncoder", "ScoreComputer"]