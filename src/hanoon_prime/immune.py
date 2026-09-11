"""hanoon_prime.immune — all hard-coded parameters, no env vars.

R6: Every safety limit is a literal constant. No os.environ, no config
bypass. This module is the system's immune system — it cannot be
weakened at runtime.

Every value is validated by tests/test_contract.py.
"""

from __future__ import annotations

# ── Indicator set (R4: exactly 5, fixed forever) ─────────────────────
INDICATOR_NAMES: tuple[str, ...] = (
    "vpin",
    "orderbook_imbalance",
    "institutional_flow",
    "momentum",
    "vwap_deviation",
)

# ── Signal processing ─────────────────────────────────────────────────
EDGE_LOOKBACK: int = 50  # min bars for indicator + z-score warmup
Z_NORM_WINDOW: int = 50  # rolling z-score window
Z_CLIP: float = 3.0  # clip z-scores to [-Z_CLIP, +Z_CLIP]

# ── Indicator sub-parameters ──────────────────────────────────────────
MOMENTUM_LOOKBACK: int = 5  # price momentum lookback (bars)
EDGE_PERIOD: int = 14  # ATR + edge lookback period
ATR_PERIOD: int = 14  # ATR lookback for stop/target
VWAP_STD_WINDOW: int = 20  # VWAP std window (z-score normalization)

# ── Entry/exit ────────────────────────────────────────────────────────
ENTRY_THRESHOLD: float = 0.65  # |tanh score| threshold for entry
ATR_STOP_MULT: float = 2.0  # stop = 2.0 × ATR(14)
ATR_TARGET_MULT: float = 6.0  # target = 6.0 × ATR(14)  → R:R = 3:1
TIMEOUT_BARS: int = 999  # disabled — ATR barriers decide
MAX_SPREAD_BPS: float = 5.0  # max bid/ask spread in basis points

# ── Sub-dollar confidence bar (raise the bar, no hard block) ──────────
# Below PENNY_PRICE the brain must be EXTREMELY sure of a quick scalp,
# so the |score| must clear PENNY_SCORE_BAR (>> ENTRY_THRESHOLD). This
# dissuades the MOST_ACTIVE scanner from feeding junk micro-caps into
# scalp sizing while still leaving the door open for a genuinely strong
# setup. Not a hard price floor — just a much higher confidence bar.
PENNY_PRICE: float = 1.00  # price below this is "sub-dollar"
PENNY_SCORE_BAR: float = 0.85  # required |score| below PENNY_PRICE
PENNY_PREMIUM_BPS: float = 25.0  # extra spread allowance for micro-caps

# ── Full-parallel-throttle order pacing ───────────────────────────────
MAX_ENTRIES_PER_CYCLE: int = 2  # hard cap on new bracket entries per cycle
ENTRY_REUSE_COOLDOWN_SEC: float = 60.0  # min gap re-entering same ticker

# ── Direction ─────────────────────────────────────────────────────────
SHORT_ALLOWED: bool = True  # dual-screen: LONG + SHORT
DIRECTION_MIN_SCORE: float = 0.02  # |score| floor; below = no_signal, not a veto

# ── Position sizing (CONTRACT.md 194-196) ─────────────────────────────
MAX_POSITION_NOTIONAL: float = 5_000.0  # $5,000 max notional per trade
MAX_LOSS_PER_TRADE: float = 50.0  # $50 hard cap per individual trade
MAX_CONCURRENT_POSITIONS: int = 3  # max open positions simultaneously
DAILY_LOSS_LIMIT: float = 200.0  # $200 daily loss → halt
CONSECUTIVE_LOSSES_PAUSE: int = 3  # 3 consec losses → pause
PAUSE_DURATION_MIN: int = 60  # pause duration in minutes

# ── Win probability mapping (R5: no inversion) ────────────────────────
SCORE_INVERT: bool = False  # R5: hardcoded False — never changes
PRIOR_BOTTOM: float = 0.25  # worst-case win probability
PRIOR_TOP: float = 0.60  # best-case win probability
PRIOR_TOP_MAX: float = 0.65  # dynamic cap never exceeds this
CONFIDENCE_FLOOR: float = 0.50  # JULI never reports below 50-50

# ── Dynamic PRIOR_TOP (faithful rebuild port, immune-bounded) ──────────
# The brain EARNs a higher PRIOR_TOP cap from realized win rate, then
# blames down when losing. The 70/30 blend dampens short winning streaks
# from over-confidence. Hard-clamped to [DYNAMIC_PRIOR_TOP_MIN,
# DYNAMIC_PRIOR_TOP_MAX] — the ceiling is PRIOR_TOP_MAX (0.65), so the
# entry gate (R5) can NEVER over-believe beyond 0.65.
DYNAMIC_PRIOR_TOP_ENABLED: bool = True
DYNAMIC_PRIOR_TOP_MIN: float = 0.35  # floor — never below original structural cap
DYNAMIC_PRIOR_TOP_MAX: float = 0.65  # ceiling == PRIOR_TOP_MAX (R5 runtime guard)
DYNAMIC_PRIOR_TOP_SCALE: float = 0.6  # WR deviation × this = dynamic delta
DYNAMIC_PRIOR_TOP_MIN_TRADES: int = 20  # cold start — use static PRIOR_TOP
DYNAMIC_PRIOR_TOP_BLEND: float = 0.7  # 70% dynamic + 30% static (dampened learning)

# ── Death-spiral PROBE recovery (off-by-default; halt stays on) ─────────
# Faithful port of rebuild's death-spiral admission (ops/admission.py:150-152
# + ops/candidate.py:240-252 + memory_calibration.py:detect_death_spiral).
# When enabled, a HALT yields to ONE quality-gated PROBE entry so the
# organism can re-establish edge and resume learning; otherwise the halt
# (brain.hippocampus.check_safety_nets) is untouched. Off until production.
PROBE_RECOVERY_ENABLED: bool = False  # R6: literal switch, no env bypass
DEATH_SPIRAL_PROBE_MIN_LOSSES: int = 3  # consecutive-loss streak to qualify
PROBE_SCORE_FLOOR: float = 0.55  # quality gate: |score| must clear this
PROBE_PRICE_FLOOR: float = 5.0  # quality gate: price must clear this ($5)
DEATH_SPIRAL_COOLDOWN_SEC: float = 600.0  # 10-min cooldown between probes

# ── MoE port: Contrarian mean-reversion override (off-by-default) ───
# Faithful Prime translation of rebuild moe_meanrev_extreme → decision_short:1.0
# (overbought extreme → contrarian SHORT; oversold → contrarian LONG). Keeps the
# halt-safe default; only flips the entry verdict when CONTRARIAN_MODE is on.
CONTRARIAN_MODE_ENABLED: bool = False
CONTRARIAN_EXTREME_Z: float = 2.0  # |vwap_deviation z| >= this triggers override

# ── Calibration / prediction-error score nudge (MoE port) ────────────
# Faithful port of rebuild prediction_error_adjustment: the score band's
# realized WR minus its predicted win_prob, clamped to ±CALIB_BOUND. Nudges
# the pipeline score (sizing + final_dir) toward where realized data says,
# NOT the cortex verdict (verdicts stay cortex-only, R1). Off-by-default:
# zero live change until opted in.
CALIBRATION_NUDGE_ENABLED: bool = False
CALIB_BOUND: float = 0.10  # rebuild CALIB_BOUND / SCORE_CALIB_BOUND

# ── Exit hysteresis (rebuild HYSTERESIS_BARS, faithful) ──────────────
# A SOFT (TIER2 JULI) exit must persist HYSTERESIS_BARS consecutive
# evaluations before confirming; TIER1 hard stops always fire immediately
# (rebuild HYSTERESIS_HARD_OVERRIDE = True). Off-by-default: 1 (disabled)
# makes every soft-exit immediate, identical to prior behavior.
HYSTERESIS_EXIT_ENABLED: bool = False
HYSTERESIS_BARS: int = 3  # consecutive bars a soft exit trigger must persist

# Opt-in learning surfaces (default OFF -> byte-identical live path).
DELIBERATION_TRACE_ENABLED: bool = False  # Deliberator CoT trace (diagnostic)
HALIM_EVIDENCE_LEARNING: bool = False  # evidence-grounded HALIM CoT -> recs

# ── R:R and fees ──────────────────────────────────────────────────────
TARGET_R_R: float = 3.0  # 2.0×ATR stop : 6.0×ATR target = 3:1
FEE_RATE: float = 0.0001  # 0.01% per leg (institutional ECN)
FIXED_FEE: float = 0.01  # $0.01 per leg (negligible round-trip)
KELLY_FRACTION: float = 0.25  # fractional Kelly (25%)

# ── Indicator weights (abs sum = 1.0) ────────────────────────────────
# Signs encode the 1-bar edge direction on FAST tickers:
#   Positive weight → positive z-score → bullish (LONG)
#   Negative weight → positive z-score → bearish (SHORT)
# FAST tickers show momentum persistence (positive 1-bar corr).
INDICATOR_WEIGHTS: dict[str, float] = {
    "vpin": 0.10,  # corr=+0.014: high VPIN → buying pressure
    "orderbook_imbalance": 0.15,  # corr=+0.002: high OBI → buying
    "institutional_flow": 0.30,  # corr=+0.004: high inst → continuation up
    "momentum": 0.20,  # corr=+0.016: high momentum → trend follow
    "vwap_deviation": 0.25,  # corr=+0.012: price above VWAP → trend up
}

# ── Learning: asymmetric punishment (R8: single system) ──────────────
LEARNING_RATE: float = 0.02  # base step size per trade
REWARD_SCALE: float = 0.5  # 0.5× reward on WIN (normalized)
# PENALTY_SCALE: legacy hippocampus path. The LIVE learning loop
# (brain/reflection.py → Reflector) uses brain/config.py:PENALTY_SCALE
# (1.2), which matches rebuild's guardrail-validated WEIGHT_LOSS_AVERSION.
# Aligned here so both paths read the same value (brain-first decision
# 2026-09-07: live value wins; 2.0 was outside rebuild's guardrail range).
PENALTY_SCALE: float = 1.2  # punishment multiplier on LOSS (asymmetric)
WEIGHT_DECAY: float = 0.999  # per-trade geometric decay
WEIGHT_MIN: float = -2.0  # lower bound on indicator weights
WEIGHT_MAX: float = 2.0  # upper bound on indicator weights
R_BASELINE: float = 0.20  # minimum acceptable R:R for weight update

# ── Edge evaluation ───────────────────────────────────────────────────
EDGE_P_VALUE: float = 0.05
EDGE_MIN_TICKERS: int = 3
EDGE_PERMUTATIONS: int = 500
EDGE_MIN_SAMPLES: int = 200

# ── IB Gateway streaming (R6: hard-coded, no env bypass) ──────────────
LOOKBACK_BARS: int = EDGE_LOOKBACK + 20  # hist buffer to seed z-score window
DEPTH_ROWS: int = 5  # order book DOM levels per side (5 max)
IB_HOST: str = "127.0.0.1"  # IB Gateway / TWS default local host
IB_PAPER_PORT: int = 4002  # IB Gateway paper port (TWS paper: 7497)
IB_LIVE_PORT: int = 4001  # IB Gateway live port (TWS live: 7496)
IB_CLIENT_ID: int = 1  # API client ID

# ── Trading universe ───────────────────────────────────────────────────
FAST_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "SPY", "TSLA", "NVDA")
# Boot fallback universe — liquid US stocks served immediately by IB Gateway
# when scanner is still warming up. Replaces this as soon as MOST_ACTIVE
# scan results fill in.
LIQUID_US_SEED: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "GOOGL",
    "META",
    "AMD",
    "INTC",
    "NFLX",
    "AVGO",
    "JPM",
    "BAC",
    "XOM",
    "CVX",
    "LLY",
    "UNH",
    "COST",
    "WMT",
    "KO",
)
TELEMETRY_PORT: int = 8080  # HTTP health endpoint for cloudflared tunnel

# ── Extended hours (pre-market 4-9:30AM ET, post-market 4-8PM ET) ──────
ALLOW_EXTENDED_HOURS: bool = True  # enable outsideRth on all IB orders
