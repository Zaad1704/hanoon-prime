"""hanoon_prime.ironclad — Ironclad pipeline integrity monitoring.

Provides cryptographic integrity, health monitoring, and self-correction
for the entire neuromorphic brain pipeline.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .ironclad_analysis import (
    HalimAnalyzer, assess_neuromorphic_health, apply_corrections, apply_halim_advice,
)

log = logging.getLogger(__name__)

_INGESTION_KEY: bytes = b"ironclad-ingestion-key-change-in-production"
_SNAPSHOT_KEY: bytes = b"ironclad-snapshot-key-change-in-production"


@dataclass
class IntegrityRecord:
    """Cryptographic integrity record for data."""
    data_hash: str
    signature: str
    timestamp: float
    source: str


@dataclass
class HealthReport:
    """System health status report."""
    timestamp: float
    uptime: float
    throughput: float
    error_rate: float
    latency_ms: float
    data_integrity: str
    neuromorphic_health: str


class IroncladMonitor:
    """Full pipeline integrity monitor with self-correction."""

    def __init__(self, julibrain: Any) -> None:
        self._brain = julibrain
        self._start_time = time.time()
        self._decision_count: int = 0
        self._error_count: int = 0
        self._latency_sum: float = 0.0
        self._snapshots: dict[str, IntegrityRecord] = {}
        self._health_history: list[HealthReport] = []
        self._last_degradation: Optional[str] = None
        halim = getattr(julibrain, "halim", None)
        self._halim_analyzer = HalimAnalyzer(halim) if halim else None

    # === Cryptographic Integrity ===

    def sign_data(self, data: Any, source: str) -> IntegrityRecord:
        """Create signed integrity record for data."""
        payload = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.sha256(payload.encode()).hexdigest()
        sig_key = _SNAPSHOT_KEY if "snapshot" in source else _INGESTION_KEY
        signature = hmac.new(sig_key, payload.encode(), hashlib.sha256).hexdigest()
        record = IntegrityRecord(data_hash, signature, time.time(), source)
        self._snapshots[source] = record
        return record

    def verify_data(self, data: Any, source: str) -> bool:
        """Verify data integrity against stored signature."""
        if source not in self._snapshots:
            return True
        record = self._snapshots[source]
        payload = json.dumps(data, sort_keys=True, default=str)
        expected_hash = hashlib.sha256(payload.encode()).hexdigest()
        if expected_hash != record.data_hash:
            log.warning("INTEGRITY FAILURE: %s hash mismatch", source)
            return False
        sig_key = _SNAPSHOT_KEY if "snapshot" in source else _INGESTION_KEY
        expected_sig = hmac.new(sig_key, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, record.signature):
            log.warning("INTEGRITY FAILURE: %s signature invalid", source)
            return False
        return True

    # === Health Monitoring ===

    def record_decision(self, latency_us: float) -> None:
        """Record a decision for health metrics."""
        self._decision_count += 1
        self._latency_sum += latency_us / 1000.0

    def record_error(self) -> None:
        """Record an error for health metrics."""
        self._error_count += 1

    def get_health_report(self) -> HealthReport:
        """Generate health status report."""
        uptime = time.time() - self._start_time
        dt = max(uptime, 0.001)
        throughput = self._decision_count / dt
        error_rate = self._error_count / max(self._decision_count, 1)
        avg_latency = (self._latency_sum / max(self._decision_count, 1)) * 1000
        nm_health = assess_neuromorphic_health(self._brain)
        integrity = "healthy" if nm_health == "optimal" else "degraded"
        report = HealthReport(time.time(), uptime, throughput, error_rate,
                              avg_latency, integrity, nm_health)
        self._health_history.append(report)
        if len(self._health_history) > 100: self._health_history = self._health_history[-100:]
        return report

    # === Self-Correction ===

    def monitor_and_correct(self) -> list[str]:
        """Monitor system and apply self-corrections."""
        actions: list[str] = []
        report = self.get_health_report()
        if report.neuromorphic_health != "optimal":
            self._last_degradation = report.neuromorphic_health
            actions.extend(apply_corrections(self._brain, self._last_degradation))
        if self._halim_analyzer:
            advice = self._halim_analyzer.get_health_advice()
            if isinstance(advice, dict):
                actions.extend(apply_halim_advice(advice))
        return actions

    # === Post-Trade Analysis ===

    def analyze_trade(self, ticker: str, won: bool, pnl: float, alpha: dict[str, float]) -> dict[str, Any]:
        """Post-trade analysis with HALIM integration."""
        analysis: dict[str, Any] = {
            "ticker": ticker, "won": won, "pnl": pnl, "alpha": alpha,
            "assessment": "win" if won else "loss",
        }
        if self._halim_analyzer:
            analysis["halim_insight"] = self._halim_analyzer.analyze_trade(ticker, won, pnl, alpha)
        self._brain.on_trade_close(ticker, won, pnl)
        if won:
            analysis["learning"] = "strengthen_winning_pathway"
        else:
            analysis["learning"] = "weaken_losing_pathway"
            analysis["suggested_stop_loss"] = alpha.get("vpin_bull", 0.5) * 2
        return analysis

    def get_postmortem(self, recent_n: int = 10) -> dict[str, Any]:
        """Generate post-mortem analysis of recent trades."""
        if not hasattr(self._brain, "episodic"):
            return {"error": "No episodic memory"}
        trades = self._brain.episodic.recent_trades(recent_n) if hasattr(self._brain.episodic, "recent_trades") else []
        if not trades: return {"message": "No trades to analyze"}
        if self._halim_analyzer:
            return self._halim_analyzer.get_postmortem(trades, recent_n)
        return {"total_trades": len(trades), "message": "No HALIM integration"}


_monitor: Optional[IroncladMonitor] = None


def get_monitor(julibrain: Any = None) -> IroncladMonitor:
    """Get or create the global integrity monitor."""
    global _monitor
    if _monitor is None:
        if julibrain is None: raise RuntimeError("Must call get_monitor(brain) first time")
        _monitor = IroncladMonitor(julibrain)
    return _monitor


def reset_monitor() -> None:
    """Reset the global monitor (for testing)."""
    global _monitor
    _monitor = None


__all__ = ["IroncladMonitor", "IntegrityRecord", "HealthReport", "get_monitor", "reset_monitor"]