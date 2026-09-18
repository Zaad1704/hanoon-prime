"""hanoon_prime.brain.neurons.moe_config — Mixture-of-Experts configuration.

Expert specialists for different market regimes:
- Momentum: Trending markets (long/short bias)
- Mean-reversion: Range-bound markets
- Liquidity: Order flow patterns

Each expert has 3 neurons for pattern formation and recall.
"""

from __future__ import annotations

# ── Expert Definitions ────────────────────────────────────────────────

MOMENTUM_HIDDEN: tuple[str, ...] = (
    "expert_momentum_1",
    "expert_momentum_2",
    "expert_momentum_3",
)

MEANREV_HIDDEN: tuple[str, ...] = (
    "expert_meanrev_1",
    "expert_meanrev_2",
    "expert_meanrev_3",
)

LIQUIDITY_HIDDEN: tuple[str, ...] = (
    "expert_liquidity_1",
    "expert_liquidity_2",
    "expert_liquidity_3",
)

CROSSASSET_HIDDEN: tuple[str, ...] = (
    "cross_regime",
    "cross_correlation",
    "cross_momentum",
)

# ── Combined Lists ───────────────────────────────────────────────────

MOE_EXPERT_INPUTS: tuple[str, ...] = (
    "momentum_bull",
    "momentum_bear",
    "vwap_deviation_bull",
    "vwap_deviation_bear",
    "orderbook_imbalance_bull",
    "orderbook_imbalance_bear",
)

CROSSASSET_EXPERT_INPUTS: tuple[str, ...] = (
    "market_regime",
    "correlation_strength",
    "momentum_shift",
    "volatility_state",
)

MOE_HIDDEN_NEURONS: tuple[str, ...] = (
    MOMENTUM_HIDDEN + MEANREV_HIDDEN + LIQUIDITY_HIDDEN
)

HIDDEN_NEURONS: tuple[str, ...] = (
    "hidden_trend",
    "hidden_momentum",
    "hidden_flow",
    "hidden_conviction",
    "hidden_volatility",
    "hidden_risk",
    "hidden_conflict",
    "hidden_structure",
) + MOE_HIDDEN_NEURONS

DECISION_NEURONS: tuple[str, ...] = (
    "decision_long",
    "decision_short",
    "decision_hold",
)

# ── Expert Weights ───────────────────────────────────────────────────

MOE_EXPERT_DECISION_WEIGHTS: dict[str, dict[str, float]] = {
    "momentum": {
        "long": 0.35,
        "short": 0.30,
        "hold": 0.25,
    },
    "meanrev": {
        "long": 0.25,
        "short": 0.35,
        "hold": 0.25,
    },
    "liquidity": {
        "long": 0.30,
        "short": 0.25,
        "hold": 0.30,
    },
}

# Audited 2026-09-18 (was: claimed 66/20/89/256 — never matched the built
# network). REAL build via NeuromorphicBridge._build_network:
#   inputs 32 (16 bull + 16 bear from ALPHA_KEYS; cross-asset neurons are not
#   built), hidden 17 (8 generalist + 9 MoE experts; cross_regime/cross_* ids
#   are NOT in HIDDEN_NEURONS), decisions 3, total 52, wired pairs 37
#   (input→hidden 29 + hidden→decision 8). Expert neurons build exists but
#   carries zero input/output synapses while NEURO_MOE_GATE_ENABLED is off;
#   wiring through moe_gate adds ~64 pairs (input→expert 48 + expert→decision 36).
NETWORK_ARCHITECTURE: dict[str, int] = {
    "input_neurons": 32,
    "hidden_neurons": 17,
    "decision_neurons": 3,
    "total_neurons": 52,
    "approximate_synapses": 37,
}


__all__ = [
    "MOMENTUM_HIDDEN",
    "MEANREV_HIDDEN",
    "LIQUIDITY_HIDDEN",
    "CROSSASSET_HIDDEN",
    "MOE_EXPERT_INPUTS",
    "CROSSASSET_EXPERT_INPUTS",
    "MOE_HIDDEN_NEURONS",
    "HIDDEN_NEURONS",
    "DECISION_NEURONS",
    "MOE_EXPERT_DECISION_WEIGHTS",
    "NETWORK_ARCHITECTURE",
]
