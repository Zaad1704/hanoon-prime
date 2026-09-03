"""hanoon_prime — the minimal, profit-first trading system (v2.0).

Neuro-morphic architecture:
  eyes        → data ingestion (OHLCV, buy-volume, bid/ask, ATR, VWAP)
  cerebellum  → 5 indicators (vpin, orderbook_imbalance, inst_flow, mom, vwap_dev)
  cortex      → z-score normalization → tanh scoring → BUY/SELL/HOLD
  hands       → ATR-based brackets (2.0×ATR stop, 6.0×ATR target), dual LONG/SHORT
  hippocampus → asymmetric punishment learning, safety nets, sizing
  immune      → all hard-coded params (no env vars)
  memory      → immutable hash-chained journal
  edge        → score → win_prob → EV → Kelly
  metrics     → performance calculation
  validator   → permutation edge test + weight calibration
  calibrate   → auto-calibration CLI
  types       → shared dataclasses (Position, Trade)
"""

from .calibrate import Calibration, calibrate
from .cerebellum import (
    INDICATOR_NAMES,
    compute_alpha,
    compute_institutional_flow,
    compute_momentum,
    compute_orderbook_imbalance,
    compute_vpin,
    compute_vwap_deviation,
)
from .cortex import Cortex, Thought
from .edge import compute_ev, compute_fee_drag, kelly_fraction, score_to_win_prob
from .eyes import (
    compute_buy_volume,
    compute_vwap,
    estimate_bid_ask,
    load_ohlcv,
    rolling_atr,
)
from .hands import simulate_ticker
from .hippocampus import Hippocampus
from .immune import (
    ATR_STOP_MULT,
    ATR_TARGET_MULT,
    CONSECUTIVE_LOSSES_PAUSE,
    DAILY_LOSS_LIMIT,
    EDGE_LOOKBACK,
    ENTRY_THRESHOLD,
    INDICATOR_WEIGHTS,
    MAX_CONCURRENT_POSITIONS,
    MAX_LOSS_PER_TRADE,
    MAX_POSITION_NOTIONAL,
    PRIOR_BOTTOM,
    PRIOR_TOP,
    PRIOR_TOP_MAX,
    SCORE_INVERT,
)
from .memory import Journal
from .metrics import calculate_metrics
from .types import Position, Trade
from .validator import (
    calibrate_weights,
    evaluate_indicator_edge,
    evaluate_indicator_pooled,
)

# IB Gateway adapter — optional dependency (requires ib_insync)
try:
    from .ib_adapter import IBStreamingBot, SafetyNetStopped

    _ib_available = True
except ImportError:
    _ib_available = False

__version__ = "2.0.0"
__all__ = [
    # Cerebellum
    "INDICATOR_NAMES",
    "compute_alpha",
    "compute_vpin",
    "compute_orderbook_imbalance",
    "compute_institutional_flow",
    "compute_momentum",
    "compute_vwap_deviation",
    # Cortex
    "Cortex",
    "Thought",
    # Edge
    "compute_ev",
    "score_to_win_prob",
    "kelly_fraction",
    "compute_fee_drag",
    # Eyes
    "load_ohlcv",
    "compute_buy_volume",
    "estimate_bid_ask",
    "rolling_atr",
    "compute_vwap",
    # Hands
    "simulate_ticker",
    # Hippocampus
    "Hippocampus",
    # Immune constants
    "INDICATOR_WEIGHTS",
    "ATR_STOP_MULT",
    "ATR_TARGET_MULT",
    "EDGE_LOOKBACK",
    "ENTRY_THRESHOLD",
    "MAX_POSITION_NOTIONAL",
    "MAX_LOSS_PER_TRADE",
    "MAX_CONCURRENT_POSITIONS",
    "DAILY_LOSS_LIMIT",
    "CONSECUTIVE_LOSSES_PAUSE",
    "SCORE_INVERT",
    "PRIOR_BOTTOM",
    "PRIOR_TOP",
    "PRIOR_TOP_MAX",
    # Memory
    "Journal",
    # Metrics
    "calculate_metrics",
    # Validator
    "evaluate_indicator_pooled",
    "evaluate_indicator_edge",
    "calibrate_weights",
    # Calibrate
    "Calibration",
    "calibrate",
    # Types
    "Position",
    "Trade",
    # IB Gateway (live adapter — optional dependency)
    "IBStreamingBot",
    "SafetyNetStopped",
]
