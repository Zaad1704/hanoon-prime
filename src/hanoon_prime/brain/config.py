"""hanoon_prime.brain.config — central constants for JULI's brain.

Every bounded modifier, threshold, and learning parameter lives here.
No module should hardcode these values — import from here.

Source: distilled from rebuild's constants.py + immune.py.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
STATE_DIR = Path(__file__).resolve().parents[2] / "runtime"
JULI_STATE_FILE = STATE_DIR / "juli_state.json"

# ── Signal thresholds ────────────────────────────────────────────────
SIGNAL_THRESHOLD: float = 0.58
ENTRY_EV_THRESHOLD: float = 0.05
CONFIDENCE_FLOOR: float = 0.50
THRESHOLD_MIN: float = 0.10
THRESHOLD_MAX: float = 0.70

# ── Modifier bounds (no single module dominates) ─────────────────────
EPISODIC_MOD_BOUND: float = 0.10
HALIM_MOD_BOUND: float = 0.03
AFFECTIVE_MOD_BOUND: float = 0.05
CONSENSUS_BOOST_MAX: float = 0.04
CONSENSUS_PENALTY_MAX: float = 0.04
HYSTERESIS_DELTA: float = 0.03
SCORE_VELOCITY_WINDOW: int = 5

# ── Episodic memory ──────────────────────────────────────────────────
EPISODIC_K: int = 7
EPISODIC_CAPACITY: int = 2000
EPISODIC_MIN_SAMPLES: int = 10
EPISODIC_KEYS: tuple[str, ...] = (
    "vpin",
    "orderbook_imbalance",
    "institutional_flow",
    "momentum",
    "vwap_deviation",
    "adx",
    "bollinger_position",
    "rsi",
    "macd_hist",
    "mfi",
    "stoch_k",
)

# ── Learning ─────────────────────────────────────────────────────────
LEARNING_RATE: float = 0.02
REWARD_SCALE: float = 0.5
PENALTY_SCALE: float = 1.2
WEIGHT_DECAY: float = 0.999
WEIGHT_MIN: float = -2.0
WEIGHT_MAX: float = 2.0
PRED_ERR_EMA_ALPHA: float = 0.10
PRED_ERR_MIN_SAMPLES: int = 20

# ── Risk / sizing ────────────────────────────────────────────────────
MAX_POSITION_NOTIONAL: float = 5_000.0
MAX_LOSS_PER_TRADE: float = 50.0
MAX_CONCURRENT_POSITIONS: int = 3
DAILY_LOSS_LIMIT: float = 200.0
CONSECUTIVE_LOSSES_PAUSE: int = 3
PAUSE_DURATION_MIN: int = 60
KELLY_FRACTION: float = 0.25
TARGET_R_R: float = 3.0

# ── Exit policy ──────────────────────────────────────────────────────
PROFIT_LOCK_TIERS: list[tuple[float, float]] = [
    (0.10, 0.04),
    (0.07, 0.03),
    (0.05, 0.02),
    (0.03, 0.01),
]
GIVEBACK_KEEP_RATIO: float = 0.55
STALE_EXIT_MINUTES: float = 120.0
CONSOLIDATION_PULSES: int = 6

# ── Regime detection ─────────────────────────────────────────────────
REGIME_VOL_WINDOW: int = 20
REGIME_TREND_WINDOW: int = 20
REGIME_VOL_HIGH_PCT: float = 0.75
REGIME_VOL_LOW_PCT: float = 0.25

# ── Default adaptive weights (27 indicators) ─────────────────────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "vpin": 0.08,
    "orderbook_imbalance": 0.08,
    "institutional_flow": 0.10,
    "momentum": 0.08,
    "vwap_deviation": 0.06,
    "rsi": 0.05,
    "macd_hist": 0.05,
    "bollinger_position": 0.05,
    "adx": 0.05,
    "stoch_k": 0.04,
    "mfi": 0.04,
    "ad_signal": 0.03,
    "obv_divergence": 0.03,
    "volume_profile_proximity": 0.03,
    "spread_tightness": 0.03,
    "trade_intensity": 0.03,
    "hurst_exponent": 0.02,
    "mean_reversion": 0.02,
    "trend_strength": 0.02,
    "sr_proximity": 0.02,
    "elliott_wave": 0.02,
    "institutional_wave": 0.02,
    "keltner_position": 0.02,
    "vw_macd_hist": 0.02,
    "microstructure": 0.02,
    "fib_proximity": 0.02,
    "kelly_fraction": 0.01,
}

# ── Guardians ────────────────────────────────────────────────────────
MAX_WEIGHT_Drift: float = 0.50
MIN_ACTIVE_INDICATORS: int = 5
CIRCUIT_BREAKER_THRESHOLD: float = 0.15
