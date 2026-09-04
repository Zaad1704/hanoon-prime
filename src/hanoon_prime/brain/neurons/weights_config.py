"""hanoon_prime.brain.neurons.weights_config — Default synaptic weights.

Defines the baseline connection strengths for the neuromorphic brain.
All alpha indicators feed into appropriate hidden neurons for decision making.
"""
from __future__ import annotations
from typing import Dict, List, Tuple


def _get_input_to_hidden_weights() -> Dict[str, float]:
    """Compute input to hidden layer weights."""
    weights: Dict[str, float] = {}
    _add_bull_weights(weights)
    _add_bear_weights(weights)
    _add_hidden_to_decision_weights(weights)
    return weights


def _add_bull_weights(weights: Dict[str, float]) -> None:
    """Add bull indicator weights to hidden_trend."""
    bull_to_trend = [
        ("bull_vpin_bull", 0.35), ("bull_momentum_bull", 0.40),
        ("bull_vwap_deviation_bull", 0.30), ("bull_orderbook_imbalance_bull", 0.45),
        ("bull_institutional_flow_bull", 0.50), ("bull_buy_volume_bull", 0.40),
        ("bull_bid_strength_bull", 0.35), ("bull_ask_strength_bull", 0.30),
        ("bull_volume_zscore_bull", 0.35), ("bull_price_momentum_bull", 0.40),
        ("bull_trend_strength_bull", 0.45), ("bull_liquidity_bull", 0.25),
        ("bull_order_imbalance_bull", 0.35), ("bull_flow_confidence_bull", 0.40),
        ("bull_mean_reversion_bull", 0.30),
    ]
    for src, w in bull_to_trend: weights[f"{src},hidden_trend"] = w


def _add_bear_weights(weights: Dict[str, float]) -> None:
    """Add bear indicator weights to hidden_volatility."""
    bear_to_vol = [
        ("bear_vpin_bear", 0.35), ("bear_momentum_bear", 0.30),
        ("bear_vwap_deviation_bear", 0.40), ("bear_orderbook_imbalance_bear", 0.45),
        ("bear_institutional_flow_bear", 0.35), ("bear_bid_strength_bear", 0.35),
        ("bear_ask_strength_bear", 0.40), ("bear_volume_zscore_bear", 0.35),
        ("bear_price_momentum_bear", 0.30), ("bear_trend_strength_bear", 0.30),
        ("bear_liquidity_bear", 0.35), ("bear_order_imbalance_bear", 0.40),
        ("bear_flow_confidence_bear", 0.30), ("bear_mean_reversion_bear", 0.25),
    ]
    for src, w in bear_to_vol: weights[f"{src},hidden_volatility"] = w


def _add_hidden_to_decision_weights(weights: Dict[str, float]) -> None:
    """Add hidden neuron to decision neuron weights."""
    weights.update({
        "hidden_trend,decision_long": 0.45, "hidden_flow,decision_long": 0.35,
        "hidden_conviction,decision_long": 0.25, "hidden_volatility,decision_short": 0.40,
        "hidden_conflict,decision_short": 0.30, "hidden_structure,decision_short": 0.25,
        "decision_long,decision_hold": -0.30, "decision_short,decision_hold": -0.30,
    })


def get_default_synaptic_weights() -> Dict[str, float]:
    """Get default weights from input to decision neurons."""
    return _get_input_to_hidden_weights()


def get_network_topology() -> Dict[str, List[Tuple]]:
    """Get network connectivity structure."""
    return {
        "input_to_hidden": [
            ("bull_vpin_bull", "hidden_trend"), ("bull_momentum_bull", "hidden_trend"),
            ("bull_vwap_deviation_bull", "hidden_trend"),
            ("bull_orderbook_imbalance_bull", "hidden_flow"),
            ("bull_institutional_flow_bull", "hidden_flow"),
            ("bull_buy_volume_bull", "hidden_conviction"),
            ("bull_bid_strength_bull", "hidden_trend"), ("bull_ask_strength_bull", "hidden_trend"),
            ("bull_volume_zscore_bull", "hidden_trend"), ("bull_price_momentum_bull", "hidden_trend"),
            ("bull_trend_strength_bull", "hidden_trend"), ("bull_liquidity_bull", "hidden_trend"),
            ("bull_order_imbalance_bull", "hidden_flow"), ("bull_flow_confidence_bull", "hidden_flow"),
            ("bull_mean_reversion_bull", "hidden_trend"),
            ("bear_vpin_bear", "hidden_volatility"), ("bear_momentum_bear", "hidden_volatility"),
            ("bear_vwap_deviation_bear", "hidden_volatility"),
            ("bear_orderbook_imbalance_bear", "hidden_conflict"),
            ("bear_institutional_flow_bear", "hidden_structure"),
            ("bear_bid_strength_bear", "hidden_volatility"), ("bear_ask_strength_bear", "hidden_volatility"),
            ("bear_volume_zscore_bear", "hidden_volatility"),
            ("bear_price_momentum_bear", "hidden_volatility"),
            ("bear_trend_strength_bear", "hidden_volatility"),
            ("bear_liquidity_bear", "hidden_volatility"),
            ("bear_order_imbalance_bear", "hidden_conflict"),
            ("bear_flow_confidence_bear", "hidden_structure"),
            ("bear_mean_reversion_bear", "hidden_volatility"),
        ],
        "hidden_to_decision": [
            ("hidden_trend", "decision_long", 0.45), ("hidden_flow", "decision_long", 0.35),
            ("hidden_conviction", "decision_long", 0.25),
            ("hidden_volatility", "decision_short", 0.40),
            ("hidden_conflict", "decision_short", 0.30),
            ("hidden_structure", "decision_short", 0.25),
            ("decision_long", "decision_hold", -0.30), ("decision_short", "decision_hold", -0.30),
        ],
    }


__all__ = ["get_default_synaptic_weights", "get_network_topology"]