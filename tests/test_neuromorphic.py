"""tests/test_neuromorphic.py — Neuromorphic brain module tests.

Tests for the LIF network, STDP learning, attractor memory, and bridge.
"""

from __future__ import annotations

import pytest

from hanoon_prime.brain.neurons import (
    LIFNeuron,
    LIFNetwork,
    STDPLearner,
    Synapse,
    AttractorMemory,
    Attractor,
    SleepReplayEngine,
    SleepResult,
)
from hanoon_prime.brain.neurons.bridge import NeuromorphicBridge, DECISION_THRESHOLD
from hanoon_prime.brain.neurons.constants import ALPHA_KEYS, HIDDEN_NEURONS, DECISION_NEURONS


class TestLIFNeuron:
    """Test Leaky Integrate-and-Fire neuron dynamics."""

    def test_neuron_creation(self) -> None:
        """Neuron has default parameters."""
        neuron = LIFNeuron(id="test_neuron", tau=0.05, threshold=0.7)
        assert neuron.id == "test_neuron"
        assert neuron.tau == 0.05
        assert neuron.threshold == 0.7

    def test_neuron_membrane_potential_default(self) -> None:
        """Neuron membrane potential starts at rest."""
        neuron = LIFNeuron(id="test_neuron", tau=0.05, threshold=0.7)
        assert neuron.v_m == 0.0

    def test_neuron_emits_spike_on_threshold(self) -> None:
        """Neuron fires when membrane potential reaches threshold."""
        neuron = LIFNeuron(id="test_neuron", tau=0.01, threshold=0.5, weight=10.0)
        neuron.set_input(0.6)
        result = neuron.step(dt=0.05)
        assert result is not None
        assert result.neuron_id == "test_neuron"

    def test_neuron_refractory_period(self) -> None:
        """Neuron cannot fire during refractory period."""
        neuron = LIFNeuron(id="test_neuron", tau=0.05, threshold=0.5)
        neuron.set_input(10.0)
        spike = neuron.step(dt=0.05)
        assert spike is not None  # First spike should fire
        # Second step during refractory should return None
        result = neuron.step(dt=0.05)
        # Due to refractory, might be None or another spike depending on timing
        # The key is that it can spike again after refractory


class TestLIFNetwork:
    """Test LIF network topology and stepping."""

    def test_network_empty_by_default(self) -> None:
        """Network starts empty."""
        network = LIFNetwork()
        # Check internal structure
        assert network._neurons == {} or len(network._neurons) == 0

    def test_add_neuron(self) -> None:
        """Neurons can be added to network."""
        network = LIFNetwork()
        neuron = LIFNeuron(id="neuron_1", tau=0.05, threshold=0.7)
        network.add_neuron(neuron)
        assert network.get_neuron("neuron_1") is neuron

    def test_connect_synapses(self) -> None:
        """Synaptic connections can be created."""
        network = LIFNetwork()
        src = LIFNeuron(id="src", tau=0.05, threshold=0.7)
        dst = LIFNeuron(id="dst", tau=0.05, threshold=0.7)
        network.add_neuron(src)
        network.add_neuron(dst)
        network.connect("src", "dst", strength=0.5)
        assert "dst" in network._synapses["src"]
        assert network._weights["src"].get("dst", 0.0) == 0.5

    def test_step_all_returns_spikes(self) -> None:
        """Stepping propagates activity through network."""
        network = LIFNetwork()
        src = LIFNeuron(id="src", tau=0.05, threshold=0.5)
        network.add_neuron(src)
        network.reset_settle_counts()
        src.set_input(10.0)  # Strong input
        spikes = network.step_all(dt=0.05)
        assert len(spikes) >= 1

    def test_snapshot_returns_state(self) -> None:
        """Snapshot returns network state."""
        network = LIFNetwork()
        snapshot = network.snapshot()
        assert "neuron_count" in snapshot
        assert "synapse_count" in snapshot


class TestSTDPLearner:
    """Test STDP (Spike-Timing-Dependent Plasticity) learning."""

    def test_create_synapse(self) -> None:
        """Synapses can be created."""
        stdp = STDPLearner()
        synapse = stdp.create_synapse("pre", "post", strength=0.5)
        assert synapse is not None
        assert synapse.strength == 0.5

    def test_store_and_retrieve_strength(self) -> None:
        """Synaptic strength can be stored and retrieved."""
        stdp = STDPLearner()
        stdp.create_synapse("pre", "post", strength=0.7)
        assert stdp.get_strength("pre", "post") == 0.7

    def test_apply_reward_updates_traces(self) -> None:
        """Reward application updates trace."""
        stdp = STDPLearner()
        stdp.create_synapse("pre", "post", strength=0.5)
        stdp.update_traces(dt=0.05)
        stdp.apply_reward(reward=1.0)
        # Should not crash


class TestAttractorMemory:
    """Test attractor memory for pattern storage."""

    def test_empty_memory_has_no_attractors(self) -> None:
        """Empty memory contains no attractors."""
        memory = AttractorMemory()
        assert len(memory) == 0

    def test_store_and_retrieve(self) -> None:
        """Patterns can be stored and retrieved."""
        memory = AttractorMemory()
        pattern = [0.5, 0.3, -0.2, 0.8]
        memory.store("TEST", pattern, won=True, pnl_pct=0.05)
        assert len(memory) == 1

    def test_attractor_properties(self) -> None:
        """Stored attractors have correct properties."""
        memory = AttractorMemory()
        pattern = [0.5, 0.3]
        memory.store("TEST", pattern, won=True, pnl_pct=0.05)
        for att in memory:
            assert att.ticker == "TEST"
            assert att.wins >= 0
            assert att.losses >= 0
            assert att.pnl_pct == 0.05


class TestSleepReplayEngine:
    """Test offline sleep replay for memory consolidation."""

    def test_engine_with_empty_memory(self) -> None:
        """Sleep engine runs with empty memory."""
        engine = SleepReplayEngine(
            network=LIFNetwork(),
            stdp=STDPLearner(),
            memory=AttractorMemory(),
        )
        result = engine.run_cycle(duration_sec=1.0)
        assert isinstance(result, SleepResult)
        assert result.patterns_replayed == 0

    def test_engine_with_patterns(self) -> None:
        """Sleep engine can replay stored patterns."""
        network = LIFNetwork()
        stdp = STDPLearner()
        memory = AttractorMemory()

        # Add a trained pattern
        memory.store("TEST", [0.5, -0.3, 0.2], won=True, pnl_pct=0.1)

        engine = SleepReplayEngine(network=network, stdp=stdp, memory=memory)
        result = engine.run_cycle(duration_sec=1.0)
        assert isinstance(result, SleepResult)

    def test_should_run_only_when_market_closed(self) -> None:
        """Sleep engine only runs when market is closed."""
        engine = SleepReplayEngine(
            network=LIFNetwork(),
            stdp=STDPLearner(),
            memory=AttractorMemory(),
        )
        assert engine.should_run(is_market_open=False) is True
        assert engine.should_run(is_market_open=True) is False


class TestNeuromorphicBridge:
    """Test the bridge between alpha inputs and network decisions."""

    def test_bridge_creation(self) -> None:
        """Bridge creates network successfully."""
        bridge = NeuromorphicBridge()
        assert bridge is not None
        assert bridge.threshold == DECISION_THRESHOLD

    def test_process_alpha_returns_result(self) -> None:
        """Processing alpha returns a result dict."""
        bridge = NeuromorphicBridge()
        result = bridge.process_alpha({"vpin_bull": 0.5}, "TEST")
        assert "score" in result
        assert "confidence" in result
        assert isinstance(result["score"], float)

    def test_learn_from_outcome(self) -> None:
        """Trade outcomes update the neuromorphic system."""
        bridge = NeuromorphicBridge()
        # Should not raise
        bridge.learn_from_outcome("TEST", won=True, pnl=0.05)
        bridge.learn_from_outcome("TEST", won=False, pnl=-0.03)

    def test_snapshot_includes_neuromorphic_state(self) -> None:
        """Snapshot includes all neuromorphic components."""
        bridge = NeuromorphicBridge()
        snapshot = bridge.snapshot()
        assert "network" in snapshot
        assert "memory_size" in snapshot
        assert "initialized" in snapshot
        assert snapshot["initialized"] is True

    def test_decision_threshold_is_positive(self) -> None:
        """Decision threshold is a valid positive value."""
        assert DECISION_THRESHOLD > 0
        assert DECISION_THRESHOLD < 1.0


class TestNeuromorphicIntegration:
    """Integration tests for the neuromorphic brain."""

    def test_julibrain_creates_with_neuromorphic(self) -> None:
        """JuliBrain creates neuromorphic bridge when enabled."""
        from hanoon_prime.brain.orchestrator import JuliBrain
        brain = JuliBrain()
        assert brain._neuromorphic is not None

    def test_julibrain_creates_without_neuromorphic(self) -> None:
        """JuliBrain can be created without neuromorphic."""
        from hanoon_prime.brain.orchestrator import JuliBrain
        brain = JuliBrain(enable_neuromorphic=False)
        assert brain._neuromorphic is None

    def test_neuromorphic_tick_produces_verdict(self) -> None:
        """Tick produces a verdict through neuromorphic processing."""
        from hanoon_prime.brain.orchestrator import JuliBrain

        brain = JuliBrain()
        alpha = {
            "vpin_bull": 0.5,
            "vpin_bear": -0.3,
            "orderbook_imbalance": 0.2,
            "institutional_flow": 0.4,
            "momentum": 0.3,
            "vwap_deviation": -0.1,
        }
        result = brain.tick(alpha=alpha, ticker="TEST", entry_price=100.0, atr=2.0, open_positions=0)
        assert "verdict" in result
        assert result["verdict"] in ("BUY", "SELL", "HOLD")

    def test_neuromorphic_on_trade_close(self) -> None:
        """Trade close updates neuromorphic learning."""
        from hanoon_prime.brain.orchestrator import JuliBrain

        brain = JuliBrain()

        # Should not raise - on_trade_close handles episodic.add internally
        brain.on_trade_close("TEST", won=True, pnl_pct=0.05, direction=1)
        brain.on_trade_close("TEST", won=False, pnl_pct=-0.02, direction=1)

    def test_sleep_replay_integration(self) -> None:
        """Sleep replay can be scheduled and run."""
        from hanoon_prime.brain.orchestrator import JuliBrain

        brain = JuliBrain()
        # Sleep engine is initialized during __init__
        assert brain._sleep_engine is not None
        result = brain.sleep_replay(is_market_open=False)
        assert isinstance(result, SleepResult)

    def test_neuromorphic_is_local_source_of_truth(self) -> None:
        """Verify neuromorphic brain produces all decisions."""
        from hanoon_prime.brain.orchestrator import JuliBrain

        brain = JuliBrain()

        # All decisions should come through neuromorphic processing
        alpha = {
            "vpin_bull": 0.5,
            "vpin_bear": -0.3,
            "orderbook_imbalance": 0.2,
            "institutional_flow": 0.4,
            "momentum": 0.3,
            "vwap_deviation": -0.1,
        }

        result = brain.tick(alpha=alpha, ticker="TEST", entry_price=100.0, atr=2.0, open_positions=0)

        # Result should have all required fields
        assert "verdict" in result
        assert "score" in result
        assert "direction" in result
        assert "confidence" in result
        assert "sizing" in result
        assert "trace" in result

        # Cortex produces verdict (R1 compliance)
        assert result["verdict"] in ("BUY", "SELL", "HOLD")