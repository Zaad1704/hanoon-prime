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
