"""tests/test_realized_ev.py — behavioral coverage for the realized-EV gate.

Covers R19: band tracking, persistence, the learned gate refusing losing
bands while admitting recovery and thin-data, the short penalty, and the
IRONYCLADE source filter on on_trade_close.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hanoon_prime.brain.config import _IRONYCLADE, DIRECTION_EXP_BOUND
from hanoon_prime.brain.realized_ev import (
    RealizedStats,
    ev_gate_should_enter,
    verify_learning_gate,
)
from hanoon_prime.edge import score_to_win_prob


def _populate(
    stats: RealizedStats, wins: int, losses: int, score: float = 0.62
) -> None:
    for _ in range(wins):
        stats.add_outcome(score, won=True, pnl_pct=0.02, direction=1)
    for _ in range(losses):
        stats.add_outcome(score, won=False, pnl_pct=-0.01, direction=1)


def test_realized_stats_band_tracking() -> None:
    r = RealizedStats(persist=False)
    r.add_outcome(0.62, won=True, pnl_pct=0.05, direction=1)
    r.add_outcome(0.62, won=True, pnl_pct=0.06, direction=1)
    r.add_outcome(0.62, won=True, pnl_pct=0.07, direction=1)
    r.add_outcome(0.62, won=False, pnl_pct=-0.02, direction=1)
    wr, rel, n = r.band_wr(0.62)
    assert n == 4
    assert wr == pytest.approx(0.75)
    assert 0.0 < rel < 1.0
    rr, rr_rel, _ = r.realized_rr()
    assert rr == pytest.approx((0.06) / (0.02))  # avg win / avg loss
    assert r.is_gate_closed(0.62) is False


def test_realized_stats_persists_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "realized.json"
    r = RealizedStats(filepath=path, persist=True)
    r.add_outcome(0.62, won=True, pnl_pct=0.05, direction=1)
    r.add_outcome(0.62, won=False, pnl_pct=-0.02, direction=1)
    r2 = RealizedStats(filepath=path, persist=True)
    assert r2.total_trades == 2
    _wr, _rel, n = r2.band_wr(0.62)
    assert n == 2


def test_ev_gate_refuses_losing_band() -> None:
    r = RealizedStats(persist=False)
    _populate(r, wins=2, losses=28, score=0.62)  # WR ~0.067, n=30 => gate closed
    info = ev_gate_should_enter(0.62, score_to_win_prob(0.62), r, direction=1)
    assert info["should_enter"] is False
    assert info["gate_closed"] is True
    assert info["reason"].startswith("learned gate")


def test_ev_gate_admits_recovery_band() -> None:
    r = RealizedStats(persist=False)
    _populate(r, wins=25, losses=0, score=0.62)  # winning band, n>=20
    info = ev_gate_should_enter(0.62, score_to_win_prob(0.62), r, direction=1)
    assert info["should_enter"] is True
    assert info["gate_closed"] is False


def test_ev_gate_thin_data_falls_back_to_structural() -> None:
    empty = RealizedStats(persist=False)
    wp = score_to_win_prob(0.6)
    info = ev_gate_should_enter(0.6, wp, empty, direction=1)
    assert info["should_enter"] is True
    # Structural fallback must reproduce edge.compute_ev gross_ev exactly.
    assert info["ev"] == pytest.approx(wp * 3.0 - (1.0 - wp))
    assert info["band_rel"] == 0.0


def test_ev_gate_edge_barely_positive_is_rejected() -> None:
    empty = RealizedStats(persist=False)
    info = ev_gate_should_enter(0.02, score_to_win_prob(0.02), empty, direction=1)
    assert info["should_enter"] is False
    assert "EV" in info["reason"]


def test_ev_gate_short_direction_penalty() -> None:
    empty = RealizedStats(persist=False)
    wp = score_to_win_prob(0.6)
    long_ev = ev_gate_should_enter(0.6, wp, empty, direction=1)["ev"]
    short_ev = ev_gate_should_enter(0.6, wp, empty, direction=-1)["ev"]
    assert short_ev == pytest.approx(long_ev * (1.0 - DIRECTION_EXP_BOUND))
    assert short_ev < long_ev


def test_verify_learning_gate_canary_passes() -> None:
    result = verify_learning_gate()
    assert result["losing_band_refused"] is True
    assert result["recovery_band_admitted"] is True
    assert result["thin_data_admitted"] is True
    assert result["all_pass"] is True


def test_on_trade_close_ironyclade_filter() -> None:
    """Synthetic/backtest fills must NOT update the realized learning loop."""
    from hanoon_prime.brain.orchestrator import NeuromorphicBrain

    brain = NeuromorphicBrain()
    brain._realized = RealizedStats(persist=False)  # hermetic
    brain.on_trade_close("T", won=True, pnl_pct=0.05, direction=1, source="real_trade")
    assert brain._realized.total_trades == 1
    # backtest/paper sources outside IRONYCLADE are ignored by the learned gate.
    brain.on_trade_close("T", won=False, pnl_pct=-0.02, direction=-1, source="backtest")
    assert brain._realized.total_trades == 1
    assert "backtest" not in _IRONYCLADE
    assert _IRONYCLADE == frozenset({"real_trade", "ib_fill", "ib_paper"})


def test_on_trade_close_ib_fill_updates_realized() -> None:
    """An ib_fill source (IRONYCLADE member) feeds the realized gate."""
    from hanoon_prime.brain.orchestrator import NeuromorphicBrain

    brain = NeuromorphicBrain()
    brain._realized = RealizedStats(persist=False)
    brain.on_trade_close("T", won=True, pnl_pct=0.04, direction=1, source="ib_fill")
    assert brain._realized.total_trades == 1
