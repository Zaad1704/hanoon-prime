"""brain.policy.portfolio_gate — fast-side READ-ONLY portfolio checks.

The slow cortex owns the mutable PortfolioRiskManager; the published
``policy_state`` snapshot is the ONLY thing the fast cortex reads. These
pure functions faithfully reproduce pre_trade_risk_gate / adjust_size
(same checks, same reason strings, same order) so the fast path never
mutates slow-side state.
"""

from __future__ import annotations

from typing import Any

from ...immune import MAX_CONCURRENT_POSITIONS, MAX_POSITION_NOTIONAL

CONCENTRATION_CAP: float = 0.25
RISK_MIN: float = 0.30
STRESS_SIZE_CAP: float = 0.30
_BUDGET: float = MAX_POSITION_NOTIONAL * MAX_CONCURRENT_POSITIONS


def portfolio_gate(
    ticker: str, notional: float, portfolio: dict[str, Any]
) -> tuple[bool, str]:
    """(True, "") to admit, (False, reason) to veto, from the snapshot."""
    if (
        not portfolio.get("equity_synced", False)
        or float(portfolio.get("equity", 0.0)) <= 0
    ):
        return False, "equity_unsynced"
    if portfolio.get("stress_mode", False) and notional > STRESS_SIZE_CAP * _BUDGET:
        return False, f"stress_size>{STRESS_SIZE_CAP:.0%}"
    if float(portfolio.get("exposure", 0.0)) >= 1.0:
        return False, "exposure_cap"
    holdings = portfolio.get("holdings", {}) or {}
    held = abs(float(holdings.get(ticker, 0.0) or 0.0))
    conc = (held + notional) / float(portfolio["equity"])
    if conc > CONCENTRATION_CAP:
        return False, f"concentration={conc:.2f}"
    if int(portfolio.get("position_count", 0)) >= int(
        portfolio.get("max_positions", MAX_CONCURRENT_POSITIONS)
    ):
        return False, f"max_positions={int(portfolio.get('position_count', 0))}"
    if portfolio.get("stress_mode", False):
        return False, f"stress_mode dd={float(portfolio.get('drawdown', 0.0)):.2f}"
    if float(portfolio.get("risk_scalar", 1.0)) <= RISK_MIN * 0.5:
        return False, f"risk_scalar={float(portfolio.get('risk_scalar', 1.0)):.2f}"
    return True, ""


def scale_shares(shares: int, price: float, risk_scalar: float, exposure: float) -> int:
    """Apply the risk scalar + exposure dampening (mirrors adjust_size)."""
    if shares <= 0 or price <= 0:
        return shares
    damp = max(0.5, 1.0 - (abs(float(exposure)) / max(_BUDGET, 1.0)) * 0.5)
    return int(shares * price * float(risk_scalar) * damp / price)


__all__ = ["portfolio_gate", "scale_shares"]
