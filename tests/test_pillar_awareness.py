"""tests/test_pillar_awareness — Juli's win/loss pillar awareness math."""

from __future__ import annotations

from pathlib import Path

from hanoon_prime.brain.pillar_awareness import (
    STATE_FALLEN,
    STATE_TIPPING,
    STATE_UPRIGHT,
    STATE_WARMING,
    compute_pillar_awareness,
    win_loss_record,
)
from hanoon_prime.brain.realized_ev import RealizedStats
from hanoon_prime.immune import PILLAR_MIN_TRADES


def _record(wins: int, losses: int, win: float = 0.02, loss: float = 0.01) -> dict:
    return win_loss_record([(1, win, 1)] * wins + [(0, -loss, 1)] * losses)


def test_break_even_tracks_payoff_ratio() -> None:
    """2:1 payoff -> break-even win rate 1/3."""
    rec = _record(5, 7)
    assert rec["r_r"] == 2.0
    assert rec["break_even_wr"] == round(1 / 3, 4)


def test_upright_when_edge_non_negative() -> None:
    p = compute_pillar_awareness(_record(5, 7))
    assert p["state"] == STATE_UPRIGHT
    assert p["upright"] is True
    assert p["tilt"] == 0.0
    assert p["edge"] > 0


def test_tipping_on_shallow_lean() -> None:
    p = compute_pillar_awareness(_record(3, 9))
    assert p["state"] == STATE_TIPPING
    assert p["upright"] is False
    assert 0.0 < p["tilt"] < 1.0


def test_fallen_on_deep_lean() -> None:
    p = compute_pillar_awareness(_record(2, 10))
    assert p["state"] == STATE_FALLEN
    assert p["tilt"] == 1.0


def test_all_losses_never_upright() -> None:
    """Zero wins with losses -> break-even 1.0 -> fully fallen."""
    rec = _record(0, 12)
    assert rec["break_even_wr"] == 1.0
    assert compute_pillar_awareness(rec)["state"] == STATE_FALLEN


def test_no_trades_is_warming() -> None:
    p = compute_pillar_awareness(win_loss_record([]))
    assert p["state"] == STATE_WARMING
    assert p["trades"] == 0


def test_below_min_trades_is_warming() -> None:
    p = compute_pillar_awareness(_record(0, PILLAR_MIN_TRADES - 1))
    assert p["state"] == STATE_WARMING


def test_none_record_is_warming() -> None:
    assert compute_pillar_awareness(None)["state"] == STATE_WARMING


def test_malformed_samples_are_skipped() -> None:
    rec = win_loss_record([("x", "y", "z"), (1, 0.02, 1)])  # type: ignore[list-item]
    assert rec["trades"] == 1
    assert rec["wins"] == 1


def test_realized_stats_win_loss_record() -> None:
    stats = RealizedStats(persist=False)
    for _ in range(5):
        stats.add_outcome(0.6, True, 0.02)
    for _ in range(7):
        stats.add_outcome(0.6, False, -0.01)
    rec = stats.win_loss_record()
    assert rec["record"] == "5W-7L"
    assert compute_pillar_awareness(rec)["state"] == STATE_UPRIGHT


def test_realized_conf_bins_survive_restart(tmp_path: Path) -> None:
    """Regression: conf_wins/conf_losses were dropped by _load()."""
    path = tmp_path / "realized.json"
    first = RealizedStats(filepath=path)
    first.add_confidence_outcome(0.70, True)
    first.add_confidence_outcome(0.70, False)
    second = RealizedStats(filepath=path)
    assert dict(second._conf_wins) == {3: 1}
    assert dict(second._conf_losses) == {3: 1}
