"""hanoon_prime.ironclad_analysis — Post-trade analysis and diagnostics.

HALIM-integrated post-trade analysis, health assessments, and self-correction.
"""
from __future__ import annotations
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)


class HalimAnalyzer:
    """HALIM-powered diagnostics for trade and system analysis."""

    def __init__(self, halim: Any) -> None:
        self._halim = halim

    def analyze_trade(self, ticker: str, won: bool, pnl: float,
                      alpha: dict[str, float]) -> dict[str, Any]:
        """Deep post-trade analysis via HALIM API."""
        if self._halim is None: return {}
        try:
            return self._halim.analyze_trade(ticker, won, pnl, alpha)
        except Exception as e:
            log.debug("HALIM trade analysis failed: %s", e)
            return {}

    def get_postmortem(self, trades: list[Any], recent_n: int = 10) -> dict[str, Any]:
        """Generate post-mortem analysis of recent trades."""
        wins = sum(1 for t in trades if getattr(t, "won", False))
        losses = len(trades) - wins
        total_pnl = sum(getattr(t, "pnl", 0) for t in trades)
        return {
            "total_trades": len(trades), "win_rate": wins / max(len(trades), 1),
            "total_pnl": total_pnl,
            "avg_win": sum(getattr(t, "pnl", 0) for t in trades if getattr(t, "won", False)) / max(wins, 1),
            "avg_loss": sum(getattr(t, "pnl", 0) for t in trades if not getattr(t, "won", False)) / max(losses, 1),
            "halim_recommendations": self._get_halim_recs(),
        }

    def _get_halim_recs(self) -> list[dict[str, Any]]:
        """Get HALIM's recommendations for improvement."""
        if self._halim is None: return []
        try:
            return self._halim.get_improvement_recommendations()
        except Exception:
            return []

    def get_health_advice(self) -> dict[str, Any]:
        """Get HALIM's health assessment and recommendations."""
        if self._halim is None: return {}
        try:
            return self._halim.get_health_advice()
        except Exception as e:
            log.debug("HALIM health check failed: %s", e)
            return {}


def assess_neuromorphic_health(brain: Any) -> str:
    """Assess neuromorphic system health."""
    snap = brain.snapshot()
    neuromorphic = snap.get("neuromorphic", {})
    net = neuromorphic.get("network", {})
    synapse_count = net.get("synapse_count", 0)
    if synapse_count < 10: return "failing"
    decision_count = snap.get("decision_count", 0)
    if decision_count == 0: return "degrading"
    episodic_size = snap.get("episodic_size", 0)
    if episodic_size > 10000: return "degrading"
    return "optimal"


def apply_corrections(brain: Any, degradation: str) -> list[str]:
    """Apply built-in self-corrections."""
    actions: list[str] = []
    if degradation == "failing":
        log.warning("NEUROMORPHIC FAILURE DETECTED - INITIATING RECOVERY")
        brain._neuromorphic = None
        brain._sleep_engine = None
        if hasattr(brain, "_init_neuromorphic"): brain._init_neuromorphic()
        actions.append("neuromorphic_reset")
    elif degradation == "degrading":
        actions.append("threshold_adjust")
    return actions


def apply_halim_advice(advice: dict[str, Any]) -> list[str]:
    """Apply HALIM's diagnostic advice."""
    actions: list[str] = []
    if advice.get("adjust_threshold", False):
        actions.append(f"threshold_set_to_{advice.get('recommended_threshold', 0.58)}")
    if advice.get("reset_neural", False):
        actions.append("halim_recommended_neural_reset")
    if advice.get("recalibrate_risk", False):
        actions.append("halim_recalibrated_risk")
    return actions


__all__ = ["HalimAnalyzer", "assess_neuromorphic_health", "apply_corrections", "apply_halim_advice"]