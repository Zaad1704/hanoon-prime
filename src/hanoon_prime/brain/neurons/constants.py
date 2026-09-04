"""hanoon_prime.brain.neurons.constants — Network architecture constants.

Defines the neuromorphic brain topology: neuron counts, layer sizes,
and alpha key mappings for volumetric encoding of trading signals.
"""

from __future__ import annotations

# ── Alpha Keys (31 total → 62 input neurons) ────────────────────────────
# Base 27 indicators mapped to bull/bear neuron pairs
ALPHA_KEYS: tuple[str, ...] = (
    "vpin_bull", "vpin_bear",
    "orderbook_imbalance_bull", "orderbook_imbalance_bear",
    "institutional_flow_bull", "institutional_flow_bear",
    "momentum_bull", "momentum_bear",
    "vwap_deviation_bull", "vwap_deviation_bear",
    "buy_volume_bull", "buy_volume_bear",
    "bid_strength_bull", "bid_strength_bear",
    "ask_strength_bull", "ask_strength_bear",
    "volume_zscore_bull", "volume_zscore_bear",
    "price_momentum_bull", "price_momentum_bear",
    "trend_strength_bull", "trend_strength_bear",
    "volatility_bull", "volatility_bear",
    "liquidity_bull", "liquidity_bear",
    "order_imbalance_bull", "order_imbalance_bear",
    "flow_confidence_bull", "flow_confidence_bear",
    "mean_reversion_bull", "mean_reversion_bear",
)

# Cross-asset signals (4 keys → 4 input neurons)
CROSS_ASSET_KEYS: tuple[str, ...] = (
    "market_regime",
    "correlation_strength",
    "momentum_shift",
    "volatility_state",
)

# All input neuron keys
ALL_ALPHA_KEYS: tuple[str, ...] = ALPHA_KEYS + CROSS_ASSET_KEYS

# ── Neuron Lists by Layer ──────────────────────────────────────────────
# Input layer: 62 neurons (31 alpha keys × bull/bear + 4 cross-asset)
INPUT_NEURONS: dict[str, tuple[str, ...]] = {
    "bull": tuple(k.replace("_bear", "_bull") for k in ALPHA_KEYS if k.endswith("_bull")),
    "bear": tuple(k.replace("_bull", "_bear") for k in ALPHA_KEYS if k.endswith("_bear")),
    "cross_asset": CROSS_ASSET_KEYS,
}

# Hidden layer: feature integration
BULL_HIDDEN: tuple[str, ...] = (
    "hidden_trend",
    "hidden_momentum",
    "hidden_flow",
    "hidden_conviction",
)

BEAR_HIDDEN: tuple[str, ...] = (
    "hidden_volatility",
    "hidden_risk",
    "hidden_conflict",
    "hidden_structure",
)

HIDDEN_NEURONS: tuple[str, :] = BULL_HIDDEN + BEAR_HIDDEN

# MoE Expert layer: domain specialists
# Each specialist gets 3 neurons for pattern formation
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

MOE_HIDDEN_NEURONS: tuple[str, ...] = MOMENTUM_HIDDEN + MEANREV_HIDDEN + LIQUIDITY_HIDDEN

# Decision layer: 3 neurons (long, short, hold)
DECISION_NEURONS: tuple[str, ...] = (
    "decision_long",
    "decision_short",
    "decision_hold",
)

# ── Network Summary ─────────────────────────────────────────────────────
TOTAL_NEURONS: int = len(ALL_ALPHA_KEYS) * 2 + len(HIDDEN_NEURONS) + len(MOE_HIDDEN_NEURONS) + len(DECISION_NEURONS)
INPUT_COUNT: int = len(ALL_ALPHA_KEYS) * 2
HIDDEN_COUNT: int = len(HIDDEN_NEURONS) + len(MOE_HIDDEN_NEURONS)
DECISION_COUNT: int = len(DECISION_NEURONS)

NETWORK_ARCHITECTURE: dict[str, int] = {
    "input": INPUT_COUNT,
    "hidden": HIDDEN_COUNT,
    "decision": DECISION_COUNT,
    "total": TOTAL_NEURONS,
}

# ── LIF Neuron Parameters ───────────────────────────────────────────────
TAU_FAST: float = 0.01  # 10ms (fast adaptation, momentum)
TAU_SLOW: float = 0.10  # 100ms (trend detection)
TAU_BASE: float = 0.05  # 50ms (default membrane time constant)

THRESHOLD_DEFAULT: float = 0.7  # Spike threshold
REST_POTENTIAL: float = 0.0  # Resting membrane potential
REFRACTORY_MS: float = 0.001  # 1ms refractory period

# ── STDP Parameters (market-scaled) ───────────────────────────────────────
A_PLUS: float = 0.01  # Max LTP per spike pair
A_MINUS: float = 0.012  # Max LTD per spike pair  
TAU_PLUS: float = 2.0  # LTP time constant (seconds)
TAU_MINUS: float = 2.5  # LTD time constant (seconds)
WEIGHT_MIN: float = 0.01  # Min synaptic strength
WEIGHT_MAX: float = 0.50  # Max synaptic strength

# ── Decision Threshold ───────────────────────────────────────────────────
DECISION_THRESHOLD: float = 0.65  # Minimum evidence for decision