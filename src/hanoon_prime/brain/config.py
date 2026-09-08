"""hanoon_prime.brain.config — central constants for JULI's brain.

Every bounded modifier, threshold, and learning parameter lives here.
No module should hardcode these values — import from here.

Source: distilled from rebuild's constants.py + immune.py.
"""

from __future__ import annotations

from pathlib import Path

# ── Build identity ───────────────────────────────────────────────────
BUILD_NAME: str = "Aegis"
BUILD_VERSION: str = "2.3.0"
# Aegis = the shield. Name for this Prime line: disciplined hard-halt
# entry gate + bounded, self-auditing exits, ported cleanly off rebuild.
# Separate lineage from rebuild's "Snowflake" (v3.0.0) — no version clash.

# ── Paths ────────────────────────────────────────────────────────────
STATE_DIR = Path(__file__).resolve().parents[3] / "runtime"
JULI_STATE_FILE = STATE_DIR / "juli_state.json"

# ── Signal thresholds ────────────────────────────────────────────────
SIGNAL_THRESHOLD: float = 0.58
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
# Learned-exit adaptation (brain/exits.py): how far exits may self-tune
# from realized outcomes. Bounded so the exit policy can never drift
# unrecognizably — it adapts, it does not reinvent itself.
EXIT_ADAPT_GIVEBACK_MAX: float = 0.20  # |Δ| cap on giveback keep-ratio
EXIT_ADAPT_STALE_MAX: float = 60.0  # minutes cap on stale shift
EXIT_ADAPT_MIN_TRADES: int = 10  # realized exits before adapting
EXIT_ADAPT_STEP_SCALE: float = 0.10  # R:R gap → keep-ratio shift scale

# ── Gate advisor (Halim gate-advisor port: threshold + size scalar) ───
ADVISOR_TIGHTEN_STEP: float = 0.02  # threshold raise per losing window
ADVISOR_LOOSEN_STEP: float = 0.01  # threshold relief per winning window
ADVISOR_DELTA_MAX: float = 0.06  # max threshold raise
ADVISOR_LOOSEN_MAX: float = 0.02  # max threshold relief
ADVISOR_MIN_TRADES: int = 20  # realized trades before acting
ADVISOR_TIGHTEN_WR: float = 0.45  # below this WR → tighten
ADVISOR_LOOSEN_WR: float = 0.55  # above this WR → loosen
GATE_CLOSED_SIZE_SCALAR: float = 0.5  # size multiplier while gate tightening

# ── Confidence-band EV gate (realized_ev.py conf bins) ───────────────
CONF_BIN_SIZE: float = 0.05  # confidence bin width (0.50→bin 10)
CONF_MIN_SAMPLES: int = 20  # min trades in a conf bin before pull

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

# ── Realized EV gate (rebuild parity: realized-Band WR + realized-R:R) ──
# The structural win probability (edge.score_to_win_prob, PRIOR 0.25->0.60) is
# pulled toward the REALIZED win rate of the current score-band, and the
# structural 3:1 R:R is pulled toward the REALIZED R:R — but ONLY once a band
# has enough samples (reliability-weighting). With no samples the gate falls
# back to the structural math (thin-data safety — no deadlock, no death spiral).
ENTRY_EV_THRESHOLD: float = 0.05
CONSERVATIVE_EV_MIN: float = 0.07
BAND_MIN_SAMPLES: int = 20  # min trades in a score-band before gate can close
RR_MIN_TRADES: int = 10  # min trades before realized R:R is trusted
NASH_GATE_AUTHORITY_WR: float = 0.45  # band WR below this (with samples) => gate closes
RR_MAX_LOOKBACK: int = 200  # cap on realized-RR history ring
DIRECTION_EXP_BOUND: float = 0.10  # bounded short-side EV penalty (rebuild dir_adj)
RELIABILITY_FLOOR: float = 0.50  # min reliability weighting for a realized pull

# ── Penny-stock notional caps (rebuild brackets.py parity) ──────────────
# Caps the dollar notional of a single position for low-priced stocks to avoid
# catastrophic gap blowups (HUIX -$478, GAUZ -$340, MMA -$204 ... all pennies).
# For normal-price stocks the cap is well above KELLY sizing, so it never
# binds on healthy setups — only on sub-$20 penny rockets.
PENNY_NOTIONAL_CAPS: list[tuple[float, float]] = [
    (2.0, 500.0),
    (5.0, 600.0),
    (20.0, 750.0),
    (float("inf"), 1_000.0),
]

# ── Realized-state persistence ────────────────────────────────────────
JULI_REALIZED_FILE: Path = STATE_DIR / "juli_realized.json"

# ── Nash pattern memory (brain/cognitive/nash.py): bounded penalty ─────
# With >= _GATE_MIN samples, pattern memory whose realized win probability
# falls below 45% leans LONG entries away (bounded penalty, scaled by
# pattern confidence + deficit); above 55% leans SHORT entries away.
# BRAIN-FIRST (2026-09-07): pattern memory is advisory — it may never zero
# the score (the old hard veto could deadlock the brain on its own history).
NASH_VETO_LOW: float = 0.45
NASH_VETO_HIGH: float = 0.55
NASH_PENALTY_MAX: float = 0.15  # max bounded score penalty from pattern memory

# ── IRONYCLADE: trade sources that may update the realized learning loop ───
# Paper / synthetic / backtest fills are excluded so the live EV gate only
# learns from real execution (rebuild brain_engine.py:447).
_IRONYCLADE: frozenset[str] = frozenset({"real_trade", "ib_fill", "ib_paper"})
