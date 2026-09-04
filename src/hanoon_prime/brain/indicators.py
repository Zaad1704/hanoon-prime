"""hanoon_prime.brain.indicators — perception orchestrator.

Thin orchestrator: defines INDICATOR_NAMES and merges cerebellum's 5 core
with 22 higher-order indicators from indicators_core + indicators_core_tech.
"""

from __future__ import annotations

from typing import Any

from ..cerebellum import compute_alpha as compute_core_alpha
from .indicators_core import compute_osc_signals
from .indicators_core_tech import compute_flow_signals

CORE_NAMES: tuple[str, ...] = (
    "vpin",
    "orderbook_imbalance",
    "institutional_flow",
    "momentum",
    "vwap_deviation",
)
EXTRA_NAMES: tuple[str, ...] = (
    "rsi",
    "macd_hist",
    "bollinger_position",
    "stoch_k",
    "stoch_d",
    "mfi",
    "adx",
    "hurst_exponent",
    "kelly_fraction",
    "ad_signal",
    "obv_divergence",
    "spread_tightness",
    "volume_profile_proximity",
    "trade_intensity",
    "mean_reversion",
    "trend_strength",
    "sr_proximity",
    "elliott_wave",
    "institutional_wave",
    "keltner_position",
    "vw_macd_hist",
    "microstructure",
    "fib_proximity",
)
INDICATOR_NAMES: tuple[str, ...] = CORE_NAMES + EXTRA_NAMES


def compute_all_alpha(
    close: Any,
    high: Any,
    low: Any,
    volume: Any,
    buy_volume: Any | None = None,
    bid_sizes: Any | None = None,
    ask_sizes: Any | None = None,
) -> dict[str, float]:
    """Merge cerebellum's 5 core + 22 higher-order into one alpha dict."""
    core = compute_core_alpha(close, volume, buy_volume, bid_sizes, ask_sizes)
    alpha: dict[str, float] = {k: core.get(k, 0.0) for k in CORE_NAMES}
    alpha.update(compute_osc_signals(close, high, low, volume))
    alpha.update(compute_flow_signals(close, high, low, volume))
    alpha["volatility"] = core.get("volatility", 0.0)
    return alpha


__all__ = ["INDICATOR_NAMES", "CORE_NAMES", "EXTRA_NAMES", "compute_all_alpha"]
