"""hanoon_prime.brain.indicators — perception orchestrator.

Thin orchestrator: defines INDICATOR_NAMES and merges cerebellum's 5 core
with 22 higher-order indicators from indicators_core + indicators_core_tech.
"""

from __future__ import annotations

from ..cerebellum import compute_alpha as compute_core_alpha
from ..types import BarSeries
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


def compute_all_alpha(bars: BarSeries) -> dict[str, float]:
    """Merge cerebellum's 5 core + 22 higher-order into one alpha dict.

    ``BarSeries`` carries the OHLCV (+ depth) arrays; grouping them avoids
    a wide positional signature while keeping every caller's data identical.
    """
    core = compute_core_alpha(
        bars.close, bars.volume, bars.buy_volume, bars.bid_sizes, bars.ask_sizes
    )
    alpha: dict[str, float] = {k: core.get(k, 0.0) for k in CORE_NAMES}
    alpha.update(compute_osc_signals(bars.close, bars.high, bars.low, bars.volume))
    alpha.update(compute_flow_signals(bars.close, bars.high, bars.low, bars.volume))
    alpha["volatility"] = core.get("volatility", 0.0)
    return alpha


__all__ = ["INDICATOR_NAMES", "CORE_NAMES", "EXTRA_NAMES", "compute_all_alpha"]
