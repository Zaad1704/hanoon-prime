"""tests/test_portfolio_gate — fast-side READ-ONLY portfolio checks.

The slow cortex owns the mutable PortfolioRiskManager; its published
snapshot is the ONLY thing the fast path reads. These pure functions must
faithfully reproduce pre_trade_risk_gate / adjust_size (same checks, same
reasons, same order).
"""

from hanoon_prime.brain.policy.portfolio_gate import portfolio_gate, scale_shares
from hanoon_prime.immune import MAX_CONCURRENT_POSITIONS, MAX_POSITION_NOTIONAL

BUDGET = MAX_POSITION_NOTIONAL * MAX_CONCURRENT_POSITIONS

BASE = {
    "equity": 100_000.0,
    "equity_synced": True,
    "risk_scalar": 1.0,
    "drawdown": 0.0,
    "stress_mode": False,
    "exposure": 0.0,
    "position_count": 0,
    "max_positions": MAX_CONCURRENT_POSITIONS,
    "holdings": {},
}


def test_equity_unsynced_rejects():
    p = dict(BASE, equity_synced=False)
    assert portfolio_gate("TSLA", 1_000.0, p) == (False, "equity_unsynced")


def test_zero_equity_rejects():
    p = dict(BASE, equity=0.0)
    assert portfolio_gate("TSLA", 1_000.0, p)[0] is False


def test_stress_size_cap_rejects_over_budget_fraction():
    p = dict(BASE, stress_mode=True)
    over = BUDGET * 0.31
    ok, reason = portfolio_gate("TSLA", over, p)
    assert ok is False and reason.startswith("stress_size")


def test_stress_small_size_admitted_then_blocked_by_stress_mode():
    p = dict(BASE, stress_mode=True, drawdown=0.30)
    ok, reason = portfolio_gate("TSLA", 100.0, p)
    assert ok is False and reason.startswith("stress_mode")


def test_exposure_cap_rejects():
    p = dict(BASE, exposure=1.05)
    assert portfolio_gate("TSLA", 1_000.0, p)[0] is False


def test_concentration_rejects():
    p = dict(BASE, holdings={"NVD": 2_000.0})
    ok, reason = portfolio_gate("NVD", 24_000.0, p)
    assert ok is False and reason.startswith("concentration")


def test_concentration_admits_under_cap():
    p = dict(BASE, holdings={"NVD": 2_000.0})
    assert portfolio_gate("NVD", 2_000.0, p) == (True, "")


def test_max_positions_rejects():
    p = dict(BASE, position_count=MAX_CONCURRENT_POSITIONS)
    assert portfolio_gate("TSLA", 1_000.0, p)[0] is False


def test_risk_scalar_floor_rejects():
    p = dict(BASE, risk_scalar=0.15)
    ok, reason = portfolio_gate("TSLA", 1_000.0, p)
    assert ok is False and reason.startswith("risk_scalar")


def test_clean_portfolio_admits():
    assert portfolio_gate("TSLA", 1_000.0, BASE) == (True, "")


def test_portfolio_gate_order_never_bypasses_unsynced():
    # Even with authorized-style flags high, equity_unsynced is authoritative.
    p = dict(BASE, equity_synced=False, exposure=0.0)
    assert portfolio_gate("TSLA", 1.0, p)[0] is False


def test_scale_shares_applies_risk_scalar():
    assert scale_shares(100, 10.0, 1.0, 0.0) == 100
    assert scale_shares(100, 10.0, 0.5, 0.0) == 50


def test_scale_shares_zero_safe():
    assert scale_shares(0, 10.0, 1.0, 0.0) == 0
    assert scale_shares(100, 0.0, 1.0, 0.0) == 100
