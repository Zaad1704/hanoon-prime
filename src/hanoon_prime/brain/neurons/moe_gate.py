"""hanoon_prime.brain.neurons.moe_gate — Soft regime-gated expert routing (G1).

Literature principle (section 5): regime modulates expert *routing*, not the
base forecast. :func:`soft_route` turns a regime label + regime multiplier
into normalized routing weights over the moment/meanrev/liquidity families.
The gate is a deterministic, tuned prior over regimes today — true offline
expert training is future work (A2-style); the plumbing lets those weights be
swapped for learned ones without touching the wiring.

Routing is applied in :func:`route_experts` which also wires the expert
subnets on first use. Everything is gated by NEURO_MOE_GATE_ENABLED (default
OFF → expert neurons stay dormant, byte-identical live path).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from .moe_config import MOE_EXPERT_DECISION_WEIGHTS

EXPERT_FAMILIES: Tuple[str, ...] = ("momentum", "meanrev", "liquidity")

# Real input-neuron ids per expert family (guard against id drift by
# presence-checking against network._neurons before connecting).
EXPERT_INPUT_SOURCES: Dict[str, Tuple[str, ...]] = {
    "momentum": (
        "bull_momentum_bull",
        "bear_momentum_bear",
        "bull_vwap_deviation_bull",
        "bear_vwap_deviation_bear",
        "bull_price_momentum_bull",
        "bear_price_momentum_bear",
        "bull_trend_strength_bull",
        "bear_trend_strength_bear",
    ),
    "meanrev": (
        "bull_mean_reversion_bull",
        "bear_mean_reversion_bear",
        "bull_vwap_deviation_bull",
        "bear_vwap_deviation_bear",
    ),
    "liquidity": (
        "bull_liquidity_bull",
        "bear_liquidity_bear",
        "bull_order_imbalance_bull",
        "bear_order_imbalance_bear",
        "bull_flow_confidence_bull",
        "bear_flow_confidence_bear",
    ),
}

REGIME_AFFINITY: Dict[str, Dict[str, float]] = {
    "trending": {"momentum": 1.0, "meanrev": 0.2, "liquidity": 0.5},
    "mean_reverting": {"momentum": 0.2, "meanrev": 1.0, "liquidity": 0.4},
    "ranging": {"momentum": 0.2, "meanrev": 0.8, "liquidity": 0.6},
    "unknown": {"momentum": 0.5, "meanrev": 0.5, "liquidity": 0.5},
}
SOFTMAX_TEMPERATURE: float = 2.0
EXPERT_INPUT_WEIGHT: float = 0.2


def soft_route(regime: str, multiplier: float = 1.0) -> Dict[str, float]:
    """Normalized routing weights ∈ (0,1) summing to 1.

    ``multiplier`` (regime_mul) scales the momentum family affinity before
    softmax, so a strong-trend read tilts routing further toward momentum —
    without ever zeroing a family (soft, not hard routing).
    """
    affinity = REGIME_AFFINITY.get(regime, REGIME_AFFINITY["unknown"])
    current = dict(affinity)
    current["momentum"] *= max(0.0, multiplier)
    expo = {
        fam: math.exp(SOFTMAX_TEMPERATURE * current[fam]) for fam in EXPERT_FAMILIES
    }
    total = sum(expo.values())
    return {fam: expo[fam] / total for fam in EXPERT_FAMILIES}


def _family_expert_ids(family: str) -> Tuple[str, ...]:
    return tuple(f"expert_{family}_{i}" for i in (1, 2, 3))


def _wire_family(bridge: object, family: str, sources: Tuple[str, ...]) -> None:
    """Wire one expert family's input synapses registered for STDP."""
    for expert_id in _family_expert_ids(family):
        for src in sources:
            if src in bridge._network._neurons:  # type: ignore[attr-defined]
                bridge._network.connect(  # type: ignore[attr-defined]
                    src, expert_id, EXPERT_INPUT_WEIGHT
                )
        if "decision_hold" in bridge._network._neurons:  # type: ignore[attr-defined]
            bridge._stdp.create_synapse(expert_id, "decision_hold", 0.1)  # type: ignore[attr-defined]


def _wire_expert_synapses(bridge: object) -> None:
    """Connect input neurons → experts and register experts in STDP."""
    for family, sources in EXPERT_INPUT_SOURCES.items():
        _wire_family(bridge, family, sources)


def route_experts(
    bridge: object,
    regime: str,
    multiplier: float = 1.0,
) -> None:
    """Apply the soft gate to expert→decision synapses (gated).

    ``bridge`` is the NeuromorphicBridge. With NEURO_MOE_GATE_ENABLED off this
    is a no-op: expert neurons stay synapse-less and dead.
    """
    from ...immune import NEURO_MOE_GATE_ENABLED

    if not NEURO_MOE_GATE_ENABLED:
        return
    route = soft_route(regime, multiplier)
    if not getattr(bridge, "_expert_wired", False):
        _wire_expert_synapses(bridge)
        bridge._expert_wired = True  # type: ignore[attr-defined]
    weights = bridge._network._weights  # type: ignore[attr-defined]
    for family, targets in MOE_EXPERT_DECISION_WEIGHTS.items():
        for direction, base_weight in targets.items():
            scaled = base_weight * route[family]
            for expert_id in _family_expert_ids(family):
                weights.setdefault(expert_id, {})[f"decision_{direction}"] = scaled


__all__ = [
    "EXPERT_FAMILIES",
    "EXPERT_INPUT_SOURCES",
    "REGIME_AFFINITY",
    "route_experts",
    "soft_route",
]
