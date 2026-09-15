"""brain.learning_telemetry — exploration/consolidation observability.

Pure, side-effect-free lenses over the strategy-bandit posteriors, the
realized-EV store, the sleep-replay engine, and the pillar snapshot so the
telemetry server can show whether the brain is exploring, consolidating, or
drifting. Nothing here writes state or touches the network.
"""

from __future__ import annotations

from typing import Any

from .arm_stats import posterior_stats
from .learning_config import (
    STRATEGY_EPS_FLOOR,
    STRATEGY_MIN_SAMPLES,
    STRATEGY_REWARD_SCALE,
)

EXPLORE_RATIO_HIGH: float = 0.15
EXPLORE_RATIO_LOW: float = 0.05
COVERAGE_HIGH: float = 0.5
COMMITTED_HIGH: float = 0.6
COMMIT_VARIANCE: float = 0.05
COLD_START_TRIALS: int = 20
COLD_START_SELECTS: int = 50


def _regret_estimate(arms: dict[str, list[dict[str, Any]]]) -> float:
    """Pseudo-regret: Σ n·(best_mean − our_mean) over every tried arm.

    Posterior means stand in for true arm values, so this is the
    empirical-gap approximation of cumulative regret, not textbook regret.
    """
    total = 0.0
    for row in arms.values():
        tried = [
            (float(a.get("mean", 0.0)), float(a.get("n", 0)))
            for a in row
            if isinstance(a, dict)
        ]
        tried = [(m, n) for m, n in tried if n > 0.0]
        if not tried:
            continue
        best = max(m for m, _ in tried)
        total += sum(n * (best - m) for m, n in tried)
    return total


def _mode(
    selects: int,
    explores: int,
    total_trials: int,
    eps: float,
    coverage: float,
    committed: int,
    total_arms: int,
    ratio: float,
) -> str:
    """Derive the exploration/consolidation phase the brain is in."""
    if total_trials < COLD_START_TRIALS and selects < COLD_START_SELECTS:
        return "cold_start"
    if (
        ratio >= EXPLORE_RATIO_HIGH
        or eps > STRATEGY_EPS_FLOOR + 1e-9
        or coverage >= COVERAGE_HIGH
    ):
        return "exploring"
    if (
        total_arms
        and committed / total_arms >= COMMITTED_HIGH
        and ratio <= EXPLORE_RATIO_LOW
    ):
        return "consolidating"
    return "balanced"


def exploration(
    bandit: dict[str, Any],
    arms: dict[str, list[dict[str, Any]]],
    research_ingested: int = 0,
) -> dict[str, Any]:
    """Exploration lens: explore share, regret, posterior commitment."""
    selects = int(bandit.get("selects", 0))
    explores = int(bandit.get("explores", 0))
    overrides = int(bandit.get("overrides", 0))
    total_trials = int(bandit.get("total_trials", 0))
    eps = float(bandit.get("eps", 0.0))
    ratio = explores / float(max(1, selects))
    committed, thin, total = posterior_stats(
        arms, COMMIT_VARIANCE, STRATEGY_MIN_SAMPLES
    )
    regret = _regret_estimate(arms)
    coverage = thin / float(max(1, total)) if total else 0.0
    mode = _mode(
        selects,
        explores,
        total_trials,
        eps,
        coverage,
        committed,
        total,
        ratio,
    )
    return {
        "selects": selects,
        "explores": explores,
        "overrides": overrides,
        "explore_ratio": round(ratio, 3),
        "eps": round(eps, 3),
        "mode": mode,
        "regret_estimate": round(regret, 4),
        "regret_per_trial": round(regret / float(max(1, total_trials)), 4),
        "committed_ratio": round(committed / total, 3) if total else 0.0,
        "hypothesis_coverage": round(coverage, 3),
        "research_ingested": research_ingested,
    }


def _realized_win_rate(realized: dict[str, Any]) -> tuple[float, int]:
    """Empirical win rate and N from the realized conf-band dictionary."""
    band_wins = realized.get("band_wins") or {}  # array-safe: dict-typed
    band_losses = realized.get("band_losses") or {}  # array-safe: dict-typed
    wins = sum(float(v) for v in band_wins.values())
    losses = sum(float(v) for v in band_losses.values())
    n = wins + losses
    return (wins / n, int(n)) if n > 0 else (0.0, 0)


def _implied_edge(arms: dict[str, list[dict[str, Any]]]) -> float:
    """Best-supported arm's posterior surplus, converted back to pnl edge."""
    best = 0.5
    for row in arms.values():
        for arm in row:
            if not isinstance(arm, dict):
                continue
            if float(arm.get("n", 0)) >= STRATEGY_MIN_SAMPLES:
                best = max(best, float(arm.get("mean", 0.5)))
    return (best - 0.5) / STRATEGY_REWARD_SCALE


def consolidation(
    arms: dict[str, list[dict[str, Any]]],
    realized: dict[str, Any],
    sleep: dict[str, Any],
    memory: dict[str, Any],
    pillar: dict[str, Any],
) -> dict[str, Any]:
    """Consolidation lens: sleep replay, realized vs implied edge parity."""
    committed, _, total = posterior_stats(arms, COMMIT_VARIANCE, STRATEGY_MIN_SAMPLES)
    win_rate, n = _realized_win_rate(realized)
    last = sleep.get("last_replay") or {}  # array-safe: dict-typed
    implied = _implied_edge(arms)
    realized_edge = float(pillar.get("edge", 0.0))
    return {
        "sleep_cycles": int(sleep.get("cycle_count", 0)),
        "last_replay": {
            "patterns_replayed": int(last.get("patterns_replayed", 0)),
            "weights_updated": int(last.get("weights_updated", 0)),
            "mean_weight_change": round(float(last.get("mean_weight_change", 0.0)), 5),
        },
        "realized_trades": max(n, int(realized.get("total", 0))),
        "realized_win_rate": round(win_rate, 3),
        "implied_edge": round(implied, 4),
        "realized_edge": round(realized_edge, 4),
        "parity_gap": round(implied - realized_edge, 4),
        "pred_error_ema": round(float(memory.get("pred_error_ema", 0.0)), 4),
        "committed_arms": committed,
        "committed_ratio": round(committed / total, 3) if total else 0.0,
    }


def learning_state(
    exploration_block: dict[str, Any], consolidation_block: dict[str, Any]
) -> dict[str, Any]:
    """One-phase summary flagging whether replay has folded evidence in."""
    phase = str(exploration_block.get("mode", "balanced"))
    last = consolidation_block.get("last_replay") or {}  # array-safe: dict-typed
    if phase == "consolidating" and int(last.get("weights_updated", 0)) > 0:
        phase = "consolidating_replayed"
    return {
        "phase": phase,
        "exploring": bool(phase in ("cold_start", "exploring")),
        "consolidating": bool(phase.startswith("consolidating")),
        "stalled": bool(phase == "balanced"),
        "parity_gap": consolidation_block.get("parity_gap", 0.0),
        "regret_per_trial": exploration_block.get("regret_per_trial", 0.0),
    }


__all__ = [
    "COMMIT_VARIANCE",
    "EXPLORE_RATIO_HIGH",
    "EXPLORE_RATIO_LOW",
    "consolidation",
    "exploration",
    "learning_state",
]
