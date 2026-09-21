"""hanoon_prime.brain.indicators — perception orchestrator.

Thin orchestrator: defines INDICATOR_NAMES and merges cerebellum's 5 core
with 22 higher-order indicators from indicators_core + indicators_core_tech.
"""

from __future__ import annotations

import logging

from ..cerebellum import compute_alpha as compute_core_alpha
from ..fracdiff import fracdiff
from ..immune import FRACDIFF_D, FRACDIFF_ENABLED
from ..types import BarSeries
from .indicators_core import compute_osc_signals
from .indicators_core_tech import compute_flow_signals

log = logging.getLogger(__name__)

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
    When ``FRACDIFF_ENABLED`` is True, a fractionally-differentiated close
    series is computed and stored on ``bars.fracdiff_close`` for downstream
    stationarity-aware indicators.
    """
    if FRACDIFF_ENABLED and bars.fracdiff_close is None:
        try:
            import numpy as np

            close_arr = np.asarray(bars.close, dtype=float).ravel()
            fd = fracdiff(close_arr, FRACDIFF_D)
            if fd is not None and len(fd) > 0:
                bars.fracdiff_close = fd
        except Exception as exc:
            log.debug("FracDiff transform failed (falling back to raw): %s", exc)
    core = compute_core_alpha(
        bars.close, bars.volume, bars.buy_volume, bars.bid_sizes, bars.ask_sizes
    )
    alpha: dict[str, float] = {k: core.get(k, 0.0) for k in CORE_NAMES}
    alpha.update(compute_osc_signals(bars.close, bars.high, bars.low, bars.volume))
    alpha.update(compute_flow_signals(bars.close, bars.high, bars.low, bars.volume))
    alpha["volatility"] = core.get("volatility", 0.0)
    if bars.fracdiff_close is not None and len(bars.fracdiff_close) > 0:
        alpha["fracdiff_available"] = 1.0
    return alpha


__all__ = ["INDICATOR_NAMES", "CORE_NAMES", "EXTRA_NAMES", "compute_all_alpha"]
