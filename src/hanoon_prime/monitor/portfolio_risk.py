"""monitor.portfolio_risk — Portfolio risk management.

Continuous risk scalar, position count caps, concentration limits,
drawdown triggers. Calls pre_trade_risk_gate() before every entry.

Source: rebuild's risk/portfolio.py (simplified).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

_RISK_MIN: float = 0.30
_RISK_MAX: float = 1.0
_MAX_POSITIONS: int = 3
_MAX_SINGLE_EXPOSURE: float = 0.25
_DRAWDOWN_TRIGGER: float = -0.05
_DRAWDOWN_RECOVERY: float = 0.02


@dataclass
class PortfolioRiskState:
    risk_scalar: float = 1.0
    position_count: int = 0
    total_exposure: float = 0.0
    drawdown: float = 0.0
    blocked: bool = False
    block_reason: str = ""


class PortfolioRiskManager:
    """Portfolio-level risk management."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._risk_scalar: float = 1.0

    def update(self, equity: float, positions: dict[str, float]) -> PortfolioRiskState:
        """Update risk state with current equity and positions."""
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        dd = 0.0
        if self._peak_equity > 0:
            dd = (equity - self._peak_equity) / self._peak_equity
        self._risk_scalar = self._compute_risk_scalar(dd)
        n_pos = len(positions)
        exposure = sum(abs(v) for v in positions.values()) / max(equity, 1)
        blocked, reason = self._check_blocked(n_pos, exposure, dd)
        return PortfolioRiskState(
            risk_scalar=self._risk_scalar,
            position_count=n_pos,
            total_exposure=exposure,
            drawdown=dd,
            blocked=blocked,
            block_reason=reason,
        )

    def _compute_risk_scalar(self, drawdown: float) -> float:
        """Auto-generated docstring."""
        if drawdown < _DRAWDOWN_TRIGGER:
            # Scale down proportionally
            severity = abs(drawdown - _DRAWDOWN_TRIGGER) / 0.10
            self._risk_scalar = max(_RISK_MIN, 1.0 - severity * 0.5)
        elif drawdown > _DRAWDOWN_RECOVERY:
            # Recover slowly
            self._risk_scalar = min(_RISK_MAX, self._risk_scalar + 0.01)
        return self._risk_scalar

    def _check_blocked(
        self, n_pos: int, exposure: float, drawdown: float
    ) -> tuple[bool, str]:
        """Check if entry should be blocked."""
        if n_pos >= _MAX_POSITIONS:
            return True, f"max_positions={n_pos}"
        if exposure > _MAX_SINGLE_EXPOSURE * _MAX_POSITIONS:
            return True, f"exposure={exposure:.2f}"
        if drawdown < -0.10:
            return True, f"drawdown={drawdown:.2f}"
        return False, ""

    def pre_trade_risk_gate(self) -> bool:
        """Returns True if entry is allowed."""
        if self._risk_scalar < _RISK_MIN + 0.05:
            return False
        return True
