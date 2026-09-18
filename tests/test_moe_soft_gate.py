"""tests/test_moe_soft_gate.py — G1: regime soft-gates expert routing.

Pre-fix: MoE expert neurons existed (expert_momentum_*, expert_meanrev_*,
expert_liquidity_*) but had zero input/output synapses — dormant. G1 wires
them behind NEURO_MOE_GATE_ENABLED and routes expert→decision weights by a
normalized softmax over regime affinities (regime modulates routing, never a
hard switch; Cortex keeps verdict authority — R1).
"""

from __future__ import annotations

import pytest

from hanoon_prime.brain.neurons.bridge import NeuromorphicBridge
from hanoon_prime.brain.neurons.moe_gate import soft_route


def test_routing_weights_are_normalized():
    route = soft_route("trending", multiplier=1.0)
    assert set(route) == {"momentum", "meanrev", "liquidity"}
    assert all(0.0 < w < 1.0 for w in route.values())
    assert sum(route.values()) == pytest.approx(1.0)


def test_trending_tilts_toward_momentum():
    route = soft_route("trending", multiplier=1.0)
    assert route["momentum"] > route["liquidity"] > route["meanrev"]


def test_mean_reverting_tilts_toward_meanrev():
    route = soft_route("mean_reverting", multiplier=1.0)
    assert route["meanrev"] > route["liquidity"] > route["momentum"]


def test_unknown_regime_balanced():
    route = soft_route("unknown", multiplier=1.0)
    assert route["momentum"] == pytest.approx(route["meanrev"], abs=1e-6)
    assert route["meanrev"] == pytest.approx(route["liquidity"], abs=1e-6)


def test_multiplier_strengthens_momentum_family():
    base = soft_route("trending", multiplier=1.0)
    boosted = soft_route("trending", multiplier=1.5)
    assert boosted["momentum"] > base["momentum"]
    assert boosted["meanrev"] < base["meanrev"]


def test_gate_off_by_default_experts_stay_dormant():
    bridge = NeuromorphicBridge()
    bridge.feed_context("T", [100.0, 101.0], "trending", 1.5)

    for fam in ("momentum", "meanrev", "liquidity"):
        for i in (1, 2, 3):
            targets = bridge._network._weights.get(f"expert_{fam}_{i}", {})
            assert targets == {}, f"expert_{fam}_{i} unexpectedly routed: {targets}"


def test_gate_on_wires_and_routes_experts(monkeypatch):
    from hanoon_prime.brain.neurons.moe_config import MOE_EXPERT_DECISION_WEIGHTS

    monkeypatch.setattr("hanoon_prime.immune.NEURO_MOE_GATE_ENABLED", True)
    bridge = NeuromorphicBridge()

    bridge.feed_context("T", [100.0, 101.0, 102.0], "trending", 1.5)
    route = soft_route("trending", 1.5)

    for fam, targets in MOE_EXPERT_DECISION_WEIGHTS.items():
        for i in (1, 2, 3):
            expert = f"expert_{fam}_{i}"
            assert f"expert_{fam}_{i}" in bridge._network._weights
            for direction, base in targets.items():
                assert bridge._network._weights[expert][
                    f"decision_{direction}"
                ] == pytest.approx(base * route[fam])


def test_experts_receive_inputs_when_gated_on(monkeypatch):
    monkeypatch.setattr("hanoon_prime.immune.NEURO_MOE_GATE_ENABLED", True)
    from hanoon_prime.brain.neurons.moe_gate import EXPERT_INPUT_SOURCES

    bridge = NeuromorphicBridge()
    bridge.feed_context("T", [100.0], "trending", 1.0)

    for fam, sources in EXPERT_INPUT_SOURCES.items():
        for src in sources:
            for expert in (
                f"expert_{fam}_1",
                f"expert_{fam}_2",
                f"expert_{fam}_3",
            ):
                assert bridge._network._weights[src][expert] == pytest.approx(0.2)


def test_process_alpha_unchanged_when_gate_off():
    bridge = NeuromorphicBridge()
    result = bridge.process_alpha({"vpin": 0.9, "momentum": 0.8}, "T")
    assert set(result) == {"score", "confidence", "trace"}
