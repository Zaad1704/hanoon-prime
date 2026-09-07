"""brain.strategy_genome — live read-model of the brain's learned strategy.

The genome is NOT a parallel store: it is a live VIEW over the brain's
actual learned state — cortex weights (global + per-regime blend),
dynamics threshold, gate-advisor delta, horizon-bandit posteriors, and
meta-label calibration. ``diagnose()`` surfaces drift findings; there
is no second JSON that can silently diverge from the brain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import SIGNAL_THRESHOLD, THRESHOLD_MAX, THRESHOLD_MIN

if TYPE_CHECKING:
    from .orchestrator import NeuromorphicBrain

# A single indicator may not dominate the genome.
_MAX_DOMINANT_WEIGHT: float = 0.35


class StrategyGenome:
    """Live diagnostic view over the brain's learned strategy state."""

    def __init__(self, brain: "NeuromorphicBrain") -> None:
        """Bind the genome view to a live brain."""
        self._brain = brain

    def get_genome(self) -> dict[str, Any]:
        """Assemble the current genome from live brain state."""
        b = self._brain
        return {
            "weights": b.cortex.get_weights(),
            "threshold": round(b.dynamics.threshold, 4),
            "base_threshold": SIGNAL_THRESHOLD,
            "advisor_delta": b._advisor.threshold_delta(),
            "regime_weights": b._regime_weights.snapshot(),
            "horizon_bandit": b._bandit.snapshot(),
            "meta_label": b._meta.snapshot(),
            "learned_exit_trades": b._learned_exit.count,
            "version": 2,
        }

    def diagnose(self) -> list[str]:
        """Return diagnostic findings about the learned strategy."""
        issues: list[str] = []
        weights = self._brain.cortex.get_weights()
        total = sum(weights.values())
        if total < 0.8 or total > 1.2:
            issues.append(f"Weight sum={total:.4f} outside [0.8, 1.2]")
        for k, v in weights.items():
            if v > _MAX_DOMINANT_WEIGHT:
                issues.append(f"{k}={v:.4f} > {_MAX_DOMINANT_WEIGHT} (dominant)")
        thresh = self._brain.dynamics.threshold
        if not THRESHOLD_MIN <= thresh <= THRESHOLD_MAX:
            issues.append(
                f"Threshold={thresh:.4f} outside [{THRESHOLD_MIN}, {THRESHOLD_MAX}]"
            )
        advisor = self._brain._advisor.snapshot()
        if advisor.get("tightening"):
            issues.append(
                f"Advisor tightening: delta={advisor.get('delta')} wr={advisor.get('recent_wr')}"
            )
        meta = self._brain._meta.snapshot()
        if meta.get("sizing_active") and meta.get("brier") is not None:
            brier = float(meta["brier"])
            if brier > 0.5:
                issues.append(f"Meta-model Brier={brier:.4f} worse than chance")
        return issues
