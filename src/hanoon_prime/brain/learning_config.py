"""brain.learning_config — constants for the strategy-learning subsystem.

Companion to ``config.py`` (which is at its R3b line budget): every
bounded modifier, threshold, and learning parameter for the meta-label,
horizon-bandit, regime-weight, and cross-asset wiring lives here. No
module may hardcode these values — import from here.
"""

from __future__ import annotations

from pathlib import Path

from .config import STATE_DIR

# ── Cross-asset lead-lag (brain/cross_asset.py via juli.py) ───────────
# The engine's raw modifier is clamped to this bound before it may touch
# the score pipeline — advisory, same size as the news bias.
CROSS_ASSET_MOD_BOUND: float = 0.04

# ── Meta-label shadow layer (brain/meta_label.py) ─────────────────────
META_MIN_SAMPLES: int = 20  # real closes before the scalar may act
META_LR: float = 0.15  # online logistic step size
META_WEIGHT_MAX: float = 2.0  # |w| clamp per feature
META_CUT: float = 0.45  # p(win) below this → scalar shrinks the size
META_SIZE_MIN: float = 0.5  # scalar floor (only ever REDUCES size)
META_FILE: Path = STATE_DIR / "juli_meta_label.json"

# ── Horizon contextual bandit (brain/horizon_bandit.py) ───────────────
BANDIT_EPS: float = 0.10  # exploration floor per selection
BANDIT_MIN_SAMPLES: int = 10  # trials in a cell before it can override
BANDIT_MARGIN: float = 0.10  # mean-reward lead needed to override
BANDIT_REWARD_SCALE: float = 5.0  # pnl_pct → reward via 0.5 + pnl*scale
BANDIT_FILE: Path = STATE_DIR / "juli_horizon_bandit.json"

# ── Per-regime weight vectors (brain/regime_weights.py) ───────────────
REGIME_MIN_TRADES: int = 15  # closes in a regime before its vector applies
REGIME_FILE: Path = STATE_DIR / "juli_regime_weights.json"

# ── Multi-timescale dopamine RPE (brain/rpe.py) ───────────────────────
# Three value channels with distinct timescales (Masset et al. 2025):
# fast ≈ 3-trade surprise, slow ≈ 50-trade tone, meta per-regime.
RPE_ALPHA_FAST: float = 0.33  # phasic channel learning rate
RPE_ALPHA_SLOW: float = 0.02  # tonic channel learning rate
RPE_ALPHA_META: float = 0.05  # per-regime expectation learning rate
RPE_LR_GAIN: float = 1.5  # surprise → learning-rate lift (× base LR)
RPE_LR_MIN: float = 0.5  # modulator floor (never stops learning)
RPE_LR_MAX: float = 2.0  # modulator ceiling (never over-learns)
RPE_FILE: Path = STATE_DIR / "juli_rpe.json"

# ── Homeostatic setpoint / interoception (brain/allostasis.py) ────────
ALLOS_MIN_TRADES: int = 10  # closes in a regime before its setpoint acts
ALLOS_ALPHA: float = 0.05  # setpoint adaptation (rolling norm, slow)
ALLOS_MARGIN: float = 0.05  # |deviation| beyond which state is strained
ALLOS_VIOLATION_MIN: int = 5  # sustained deviations → dyshomeostasis
ALLOS_TIGHTEN_STEP: float = 0.02  # extra threshold raise while dyshomeostatic
ALLOS_FILE: Path = STATE_DIR / "juli_allostasis.json"

# ── Somatic markers (brain/somatic.py) ────────────────────────────────
SOMATIC_MAX: float = 0.10  # marker bound (vmPFC-style bias, advisory)
SOMATIC_TILT_GAIN: float = 0.05  # pillar lean → negative pressure
SOMATIC_TONIC_GAIN: float = 0.05  # RPE mood magnitude
SOMATIC_PHASIC_GAIN: float = 0.03  # per-trade surprise magnitude
SOMATIC_DYS_PENALTY: float = -0.04  # allostatic alarm
SOMATIC_PRECISION_DAMP: float = 2.0  # negative marker → modifier dampen
SOMATIC_PRECISION_LOOR: float = 0.6  # never zero the modifiers' voice

# ── Context-dependent extinction (brain/extinction.py) ────────────────
EXTINCT_MAX: float = 0.10  # inhibition bound (can cancel EPISODIC_MOD_BOUND)
EXTINCT_MIN_PATTERNS: int = 5  # shares in a context before it can inhibit
EXTINCT_PERF_ALPHA: float = 0.3  # performance EWMA weight on latest outcome
EXTINCT_LOSS_BELOW: float = -0.02  # perf below this → grow the inhibition
EXTINCT_STEP: float = 0.01  # inhibition growth per degraded share
EXTINCT_DECAY: float = 0.005  # recovery per healthy share
EXTINCT_OVERLAP: int = 2  # shared signature dims for neighbor inhibition
EXTINCT_CELLS_MAX: int = 4096  # guard against unbounded signature map
EXTINCT_FILE: Path = STATE_DIR / "juli_extinction.json"

# ── Sleep replay scheduler (brain/sleep_scheduler.py) ─────────────────
SLEEP_THRESHOLD_SEC: float = 1800  # 30min inactivity before auto-replay
SLEEP_COOLDOWN_SEC: float = 3600  # at most one replay per hour of downtime
SLEEP_LOSS_WEIGHT: float = 3.0  # loser replay drive (3× winner)
SLEEP_WIN_WEIGHT: float = 1.0  # winner replay drive (learn, don't soothe)
SLEEP_INTERLEAVE_MAX: int = 6  # random historical traces per replay
SLEEP_MIN_PATTERNS: int = 3  # consolidated patterns before replay makes sense

# ── Metacognitive confidence-of-confidence (brain/metacog.py) ─────────
METACOG_BINS: int = 5  # coarse confidence buckets for calibration
METACOG_SAMPLES: int = 40  # rolling calibration window
METACOG_MIN_SAMPLES: int = 8  # penalize reliability only after enough data
METACOG_SHRINK_WEAK: float = 0.85  # sizing scalar at mildly unreliable
METACOG_SHRINK_BAD: float = 0.70  # sizing scalar at unreliable calibration
METACOG_SURPRISE_THRESHOLD: float = 0.55  # novelty floor that piques curiosity
METACOG_CURIOUS_SCALE: float = 1.06  # explore: nudge size up (stable pillar)
METACOG_RETREAT_SCALE: float = 0.80  # retreat: shrink size (falling pillar)
METACOG_FILE: Path = STATE_DIR / "juli_metacog.json"

# ── Researched-strategy library (brain/strategy_registry.py) ──────────
# Halim researches strategies from the web; the registry stores them as
# bounded modifier bundles (sizing scalar + score modifier). Everything a
# researched strategy may hold is clamped — advisory, never a verdict.
STRATEGY_MIN_SAMPLES: int = 10  # realized closes before a strategy may win
STRATEGY_MAX_POOL: int = 24  # library ceiling (keeps the pool auditable)
STRATEGY_SIZING_MIN: float = 0.8  # researched sizing floor
STRATEGY_SIZING_MAX: float = 1.2  # researched sizing ceiling
STRATEGY_SCORE_MOD_BOUND: float = 0.03  # researched score-modifier bound
STRATEGY_FILE: Path = STATE_DIR / "juli_strategy_registry.json"

# ── Strategy bandit (brain/strategy_bandit.py) ─────────────────────────
STRATEGY_EPS0: float = 0.20  # starting exploration share per regime
STRATEGY_EPS_FLOOR: float = 0.02  # exploration floor as JULI learns
STRATEGY_EPS_DECAY: float = 120.0  # strategies per decay-halving lifetime
STRATEGY_MARGIN: float = 0.15  # posterior lead over default to override
STRATEGY_REWARD_SCALE: float = 4.0  # pnl_pct → reward via 0.5 + pnl*scale
STRATEGY_BANDIT_FILE: Path = STATE_DIR / "juli_strategy_bandit.json"

# ── Shadow book (brain/shadow_book.py) ─────────────────────────────────
# Zero-size paper trials feed the advisory strategy organs (bandit +
# registry) so they stop starving for real closes. Offline default; only
# the whitelisted organs ever see shadow outcomes.
SHADOW_TTL: float = 5400.0  # paper hold before forced close (90 min)
SHADOW_MAX_OPEN: int = 12  # concurrent open paper positions
SHADOW_MAX_HISTORY: int = 600  # retained closed trials (ring buffer)
SHADOW_BOOK_FILE: Path = STATE_DIR / "juli_shadow_book.json"

# ── Strategy research cadence (brain/strategy_research.py) ─────────────
RESEARCH_BASE_URL: str = "http://127.0.0.1:8765"
RESEARCH_MAX_PER_CYCLE: int = 3  # strategies ingested from one research pass
RESEARCH_TIMEOUT: float = 45.0  # bounded research HTTP call (LM generation ~20s)
RESEARCH_INTERVAL_SEC: float = 300.0  # at most one research pass per 5 min
RESEARCH_TOPICS: tuple[str, ...] = (
    "mean reversion regimes",  # keep the brain honest: classic edges only
    "trend pullback entries",
    "volatility compression breakouts",
    "momentum continuation patterns",
    "range suppression fade",
)
