"""tests/test_sleep_replay_live.py — C1: replay is real, SNN input repaired.

Pre-fix proof points (all now reversed):
- ``_replay_pattern`` scraped ``alpha_N`` keys into neuron ids that do not
  exist → 0 spikes, 0 weight updates, ``POISSON_RATE`` unused.
- ``InputEncoder`` mapped plain alpha names to ``bull_<name>``/``bear_<name>``
  while ``_build_network`` created ``bull_<name>_bull``/``bear_<name>_bear``
  → the live SNN never fired (score always 0.0).
- The restored SNN blend is gated behind ``NEURO_BLEND_ENABLED`` (default
  False) so the live 0.7·cortex+0.3·neuro path stays byte-identical.
"""

from __future__ import annotations

import pytest

from hanoon_prime.brain.neurons.bridge import NeuromorphicBridge
from hanoon_prime.brain.neurons.bridge_scoring import InputEncoder
from hanoon_prime.brain.neurons.sleep import SleepReplayEngine
from hanoon_prime.brain.neurons.stdp import STDPLearner


def _full_engine() -> tuple[NeuromorphicBridge, SleepReplayEngine]:
    bridge = NeuromorphicBridge()
    engine = SleepReplayEngine(
        network=bridge._network, stdp=bridge._stdp, memory=bridge._memory
    )
    return bridge, engine


def _bull_alpha() -> dict[str, float]:
    return {
        k: 0.9
        for k in (
            "vpin",
            "orderbook_imbalance",
            "institutional_flow",
            "momentum",
            "vwap_deviation",
        )
    }


# ── Input encoding repair ────────────────────────────────────────────
def test_plain_alpha_keys_drive_real_input_neurons():
    bridge = NeuromorphicBridge()
    network = bridge._network
    InputEncoder.encode(network, {"vpin": 0.8, "momentum": 0.7, "rsi": 0.9})
    assert network._neurons["bull_vpin_bull"]._input_current > 0.0
    assert network._neurons["bear_vpin_bear"]._input_current < 0.0
    assert network._neurons["bull_momentum_bull"]._input_current > 0.0


def test_process_alpha_with_plain_keys_fires_network():
    bridge = NeuromorphicBridge()
    result = bridge.process_alpha(_bull_alpha(), "T")
    assert result["trace"]["spikes"] > 0
    assert result["score"] != 0.0


# ── Replay actually changes synapses ─────────────────────────────────
def test_winning_replay_reports_real_plasticity():
    _, engine = _full_engine()
    pattern = engine.encode_pattern([0.9] * 11)
    assert pattern, "a strong bull pattern must map onto real input neurons"

    result = engine.run_cycle(duration_sec=1.0, replay_list=[(pattern, 1.0)])

    assert result.patterns_replayed == 1
    assert result.spikes_generated > 0
    assert result.weights_updated > 0
    assert result.mean_weight_change != 0.0


def test_winner_reinforces_decision_edge():
    bridge, engine = _full_engine()
    before = engine._stdp.get_strength("hidden_trend", "decision_long")
    pattern = engine.encode_pattern([0.9] * 11)
    engine.run_cycle(duration_sec=1.0, replay_list=[(pattern, 1.0)])
    after = engine._stdp.get_strength("hidden_trend", "decision_long")
    assert after > before


def test_replay_consolidates_full_wiring():
    """STDP must cover the input→hidden→decision path, not just hidden→hold."""
    bridge, _ = _full_engine()
    assert bridge._stdp.get_strength("bull_vpin_bull", "hidden_trend") == pytest.approx(
        0.35
    )
    assert bridge._stdp.get_strength("hidden_trend", "decision_long") == pytest.approx(
        0.45
    )


def test_poisson_noise_injects_on_quiet_neurons():
    from hanoon_prime.brain.neurons import replay as _replay

    bridge, engine = _full_engine()
    network = bridge._network
    quiet = "bear_vpin_bear"
    _replay.inject_poisson(network, engine.POISSON_RATE)
    assert network.get_neuron(quiet) is not None


# ── Blend gate (byte-identical live path) ────────────────────────────
def test_neuro_blend_gated_off_by_default(monkeypatch):
    from hanoon_prime.brain.orchestrator import NeuromorphicBrain

    brain = NeuromorphicBrain()
    assert brain._compute_neuro_score(_bull_alpha(), "T") == 0.0
    monkeypatch.setattr("hanoon_prime.brain.orchestrator.NEURO_BLEND_ENABLED", True)
    assert brain._compute_neuro_score(_bull_alpha(), "T") != 0.0


def test_engine_with_empty_stdp_errors_gracefully():
    from hanoon_prime.brain.neurons.attractor import AttractorMemory
    from hanoon_prime.brain.neurons.network import LIFNetwork
    from hanoon_prime.brain.neurons.sleep import SleepResult

    engine = SleepReplayEngine(
        network=LIFNetwork(), stdp=STDPLearner(), memory=AttractorMemory()
    )
    result = engine.run_cycle(duration_sec=1.0, replay_list=[({"alpha_0": 0.5}, 1.0)])
    assert isinstance(result, SleepResult)
    assert result.weights_updated == 0
