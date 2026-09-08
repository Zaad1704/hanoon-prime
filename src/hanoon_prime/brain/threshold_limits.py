"""brain.threshold_limits — learned threshold defaults, bounds, helpers.

Pure data + a clamp helper. Split out of ``adaptive_thresholds`` so every
file stays under R3's 200-line cap even after black's 88-char expansion.
Values mirror the rebuild's DEFAULTS / BOUNDS exactly (verified in
``tests/test_adaptive_thresholds.py``). No behaviour lives here.
"""

from __future__ import annotations

from typing import Any, Tuple

# ── Defaults (mirrors rebuild DEFAULTS) ───────────────────────────────
DEFAULTS: dict[str, Any] = {
    "exit_base": 0.18,
    "exit_scale": 0.35,
    "exit_max": 0.50,
    "stale_minutes_force": 30.0,
    "stale_minutes_tighten": 15.0,
    "stale_loss_pct_force": -0.02,
    "stale_loss_pct_tighten": -0.03,
    "consolidation_pulses": 40,
    "consolidation_min_profit": 0.03,
    "consolidation_flat_pct": 0.003,
    "ride_winners_pnl": 0.03,
    "ride_winners_exit_lik": 0.25,
    "trail_keep_ratio": 0.55,
    "profit_lock_tiers": [[0.10, 0.04], [0.07, 0.03], [0.05, 0.02], [0.03, 0.01]],
    "flat_timeout_minutes": 20.0,
    "atr_stop_mult": 3.0,
    "max_risk_per_trade": 0.02,
    "exit_w_setup": 0.25,
    "exit_w_momentum": 0.12,
    "exit_w_flow": 0.10,
    "exit_w_giveback": 0.10,
    "exit_w_time": 0.18,
    "exit_w_stale": 0.10,
    "exit_w_episodic": 0.05,
    "exit_w_sentiment": 0.10,
}

BOUNDS: dict[str, Tuple[float, float]] = {
    "exit_base": (0.05, 0.35),
    "exit_scale": (0.10, 0.60),
    "exit_max": (0.25, 0.55),
    "stale_minutes_force": (15.0, 180.0),
    "stale_minutes_tighten": (10.0, 90.0),
    "stale_loss_pct_force": (-0.10, -0.005),
    "stale_loss_pct_tighten": (-0.15, -0.005),
    "consolidation_pulses": (10, 120),
    "consolidation_min_profit": (0.01, 0.08),
    "consolidation_flat_pct": (0.001, 0.01),
    "ride_winners_pnl": (0.01, 0.07),
    "ride_winners_exit_lik": (0.10, 0.50),
    "trail_keep_ratio": (0.30, 0.80),
    "flat_timeout_minutes": (10.0, 45.0),
    "atr_stop_mult": (1.5, 6.0),
    "max_risk_per_trade": (0.005, 0.05),
    "exit_w_setup": (0.05, 0.40),
    "exit_w_momentum": (0.03, 0.30),
    "exit_w_flow": (0.03, 0.25),
    "exit_w_giveback": (0.03, 0.25),
    "exit_w_time": (0.05, 0.35),
    "exit_w_stale": (0.03, 0.25),
    "exit_w_episodic": (0.02, 0.15),
    "exit_w_sentiment": (0.03, 0.25),
}

# ── Learning-rate constants (mirrors rebuild LR_UP / LR_DOWN) ─────────
LR_UP: float = 0.02  # conservative — winners take time
LR_DOWN: float = 0.04  # faster — cut what doesn't work

# Advisory exit-likelihood pillar keys (order = rebuild exit_w_* set).
PILLAR_KEYS: Tuple[str, ...] = (
    "exit_w_setup",
    "exit_w_momentum",
    "exit_w_flow",
    "exit_w_giveback",
    "exit_w_time",
    "exit_w_stale",
    "exit_w_episodic",
    "exit_w_sentiment",
)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to ``[low, high]``."""
    return max(low, min(high, float(value)))


__all__ = ["DEFAULTS", "BOUNDS", "LR_UP", "LR_DOWN", "PILLAR_KEYS", "clamp"]
