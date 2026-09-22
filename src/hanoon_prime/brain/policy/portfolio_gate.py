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
CORRELATION_THRESHOLD: float = 0.70
CORRELATION_PENALTY: float = 0.50
_BUDGET: float = MAX_POSITION_NOTIONAL * MAX_CONCURRENT_POSITIONS


def correlation_scalar(
    ticker: str,
    holdings: dict[str, Any],
    correlations: dict[str, float] | None = None,
) -> float:
    """Reduce sizing when opening a position correlated with existing holdings.

    ``correlations`` maps ``"{TICKER_A}_{TICKER_B}"`` to ρ ∈ [-1, 1].
    If ρ >= CORRELATION_THRESHOLD for any held ticker, returns
    CORRELATION_PENALTY (0.50x).  Otherwise returns 1.0 (no penalty).
    """
    if not correlations or not holdings:
        return 1.0
    held_tickers = [
        t for t, qty in holdings.items() if t != ticker and float(qty or 0) > 0
    ]
    if not held_tickers:
        return 1.0
    for other in held_tickers:
        key_fwd = f"{ticker}_{other}"
        key_rev = f"{other}_{ticker}"
        rho = correlations.get(key_fwd, correlations.get(key_rev, 0.0))
        if abs(rho) >= CORRELATION_THRESHOLD:
            return CORRELATION_PENALTY
    return 1.0


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
    holdings = portfolio.get("holdings", {}) or {}  # array-safe: dict-typed
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


def scale_shares(
    shares: int,
    price: float,
    risk_scalar: float,
    exposure: float,
    ticker: str = "",
    holdings: dict[str, Any] | None = None,
    correlations: dict[str, float] | None = None,
) -> int:
    """Apply risk scalar + exposure dampening + correlation penalty."""
    if shares <= 0 or price <= 0:
        return shares
    damp = max(0.5, 1.0 - (abs(float(exposure)) / max(_BUDGET, 1.0)) * 0.5)
    corr = correlation_scalar(ticker, holdings or {}, correlations)
    return int(shares * price * float(risk_scalar) * damp * corr / price)


__all__ = ["portfolio_gate", "scale_shares", "correlation_scalar"]
