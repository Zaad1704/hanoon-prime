"""hanoon_prime/constants — all tunable parameters, nothing else.

Single source of truth for scores, thresholds, and safety limits.
Every value here is validated by tests/test_contract.py.
"""
from __future__ import annotations

# ── Score range ──────────────────────────────────────────────────────────
THRESHOLD_MIN: float = 0.10   # floor: no candidate scores below this
THRESHOLD_MAX: float = 0.70  # ceiling: no candidate scores above this
SIGNAL_THRESHOLD: float = 0.62  # auto-calibrated via grid search
SCORE_INVERT: bool = False   # R5: NO score inversion — honest scoring

# ── Win probability mapping ────────────────────────────────────────────────
PRIOR_BOTTOM: float = 0.25   # worst producible candidate (~break-even)
PRIOR_TOP: float = 0.55      # best producible candidate (~marginal edge)
PRIOR_TOP_MAX: float = 0.65  # dynamic cap never exceeds this
CONFIDENCE_FLOOR: float = 0.50  # JULI never reports below 50-50

# ── R:R and EV ───────────────────────────────────────────────────────────
TARGET_R_R: float = 3.33   # 1:3.33 reward:risk (3% stop, 10% target)
FEE_RATE: float = 0.001   # 0.1% per leg (IBTiered)
FIXED_FEE: float = 0.50   # $0.50 per leg (IB tiered)

# ── Kelly ────────────────────────────────────────────────────────────────
KELLY_FRACTION: float = 0.25  # use 25% Kelly (conservative sizing)

# ── Safety Nets (R6: hard stops, NOT config-bypassable) ──────────────────
MAX_POSITION_NOTIONAL: float = 5000.0   # $5,000 max per trade
MAX_LOSS_PER_TRADE: float = 50.0         # $50 hard stop per trade
MAX_CONCURRENT_POSITIONS: int = 3        # max open positions
DAILY_LOSS_LIMIT: float = 200.0          # $200 daily loss → hard shutdown
CONSECUTIVE_LOSSES_PAUSE: int = 3        # 3 consecutive losses → pause 60 min
PAUSE_DURATION_MIN: int = 60             # pause duration in minutes

# ── Indicator weights (sum = 1.0) ────────────────────────────────────────
# Original working weights — institutional_flow at 0.25 (balanced).
# Auto-calibration confirms institutional_flow has strongest edge.
INDICATOR_WEIGHTS: dict[str, float] = {
    "vpin": 0.25,
    "orderbook_imbalance": 0.25,
    "institutional_flow": 0.25,
    "momentum": 0.15,
    "vwap_deviation": 0.10,
}

# ── Self-evaluation parameters ─────────────────────────────────────────────
EDGE_LOOKBACK: int = 30
EDGE_P_VALUE: float = 0.05
EDGE_MIN_TICKERS: int = 3  # min tickers (out of 5) where an indicator must show edge

# ── Learning ─────────────────────────────────────────────────────────────
LEARNING_RATE: float = 0.02   # per-trade, per-indicator weight update
WEIGHT_FLOOR: float = 0.01     # no weight may go below this
