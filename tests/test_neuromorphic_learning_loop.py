"""tests/test_neuromorphic_learning_loop.py — A1: STDP writes the fast path.

Pre-fix: STDPLearner mutated its own store, but the scorer (and propagation)
read ``LIFNetwork._weights`` — two disjoint copies, so "learning" never moved
a score. A1 syncs the STDP store back into ``_weights`` behind
``NEURO_LEARN_ENABLED`` (default OFF → byte-identical live path).
"""

from __future__ import annotations

import pytest

from hanoon_prime.brain.neurons.bridge import NeuromorphicBridge
from hanoon_prime.brain.neurons.replay import (
    apply_pattern_input,
    drive_steps,
    encode_pattern,
)


def _bridge_with_history() -> NeuromorphicBridge:
    """Bridge with a full-descriptor replay so decision-path synapses spike."""
    bridge = NeuromorphicBridge()
    pattern = encode_pattern(bridge._network, [0.8] * 11)
    apply_pattern_input(bridge._network, pattern, weight=1.0)
    drive_steps(bridge._network, bridge._stdp, steps=8, poisson_rate=0.0)
    return bridge


def test_learning_writeback_gated_off_by_default():
    bridge = _bridge_with_history()
    baseline = bridge._network._weights["hidden_trend"]["decision_long"]

    bridge.learn_from_outcome("T", won=True, pnl=0.5)

    assert bridge._network._weights["hidden_trend"]["decision_long"] == baseline


def test_learned_decision_edge_lands_in_fast_path(monkeypatch):
    monkeypatch.setattr("hanoon_prime.immune.NEURO_LEARN_ENABLED", True)

    win = _bridge_with_history()
    win.learn_from_outcome("T", won=True, pnl=0.5)

    loss = _bridge_with_history()
    loss.learn_from_outcome("T", won=False, pnl=-0.3)

    winning = win._network._weights["hidden_trend"]["decision_long"]
    losing = loss._network._weights["hidden_trend"]["decision_long"]
    assert winning > losing
    assert winning > 0.45  # above the static default


def test_sync_covers_all_stdp_synapses(monkeypatch):
    monkeypatch.setattr("hanoon_prime.immune.NEURO_LEARN_ENABLED", True)
    bridge = _bridge_with_history()

    bridge.learn_from_outcome("T", won=False, pnl=-0.3)

    for (src, dst), syn in bridge._stdp._synapses.items():
        assert bridge._network._weights[src][dst] == pytest.approx(syn.strength)


def test_losing_outcome_pulls_fast_path_down(monkeypatch):
    monkeypatch.setattr("hanoon_prime.immune.NEURO_LEARN_ENABLED", True)
    bridge = _bridge_with_history()
    pre = bridge._stdp.get_strength("bull_vpin_bull", "hidden_trend")

    bridge.learn_from_outcome("T", won=False, pnl=-0.3)
    after = bridge._network._weights["bull_vpin_bull"]["hidden_trend"]

    assert after < pre
    assert after >= 0.01  # STEEP_MIN floor respected
