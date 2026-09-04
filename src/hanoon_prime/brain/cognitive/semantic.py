"""hanoon_prime.brain.cognitive.semantic — Knowledge base (conceptual patterns).

Holds conceptual knowledge about market regimes and indicator reliability.
Regime masks suppress unreliable indicators; boost rules amplify reliable ones.
Modifier bounded by SEMANTIC_MOD_BOUND (±0.03).
"""
from __future__ import annotations

from typing import Any

# Regime → indicators to suppress (set to 0)
_REGIME_SUPPRESS: dict[str, list[str]] = {
    "overnight": [
        "orderbook_imbalance",
        "trade_intensity",
        "spread_tightness",
        "institutional_flow",
        "vpin",
    ],
    "post": ["institutional_flow", "trade_intensity"],
}

# Regime → indicator → boost multiplier
_REGIME_BOOST: dict[str, dict[str, float]] = {
    "rth": {"institutional_flow": 1.5, "volume_profile_proximity": 1.3},
}

MOD_BOUND: float = 0.03


class SemanticMemory:
    """Regime-aware indicator modulation via boost/suppress rules."""

    def evaluate(self, alpha: dict[str, float], regime: str = "unknown") -> float:
        """Compute bounded modifier from regime knowledge.

        Args:
            alpha: current indicator values.
            regime: current market regime label.

        Returns:
            Modifier in [-MOD_BOUND, +MOD_BOUND].
        """
        mod = 0.0
        suppressed = _REGIME_SUPPRESS.get(regime, [])
        boosted = _REGIME_BOOST.get(regime, {})
        for key in suppressed:
            if key in alpha and abs(alpha[key]) > 0.1:
                mod -= 0.01
        for key, mult in boosted.items():
            if key in alpha and alpha[key] > 0.5:
                mod += 0.01 * (mult - 1.0)
        return max(-MOD_BOUND, min(MOD_BOUND, mod))
