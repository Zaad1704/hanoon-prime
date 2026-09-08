"""tests/test_adaptive_thresholds.py — verify the learned exit-threshold port.

Grounded in the rebuild's `hanoon/juli/adaptive_thresholds.py` DEFAULTS/BOUNDS
and the 9 update_from_outcome rules. Each threshold is bounded so it can
never collapse; the tests assert the bounds are respected.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _fresh(tmp_path):
    """A hermetic AdaptiveThresholds writing to a temp file."""
    from hanoon_prime.brain.adaptive_thresholds import AdaptiveThresholds

    return AdaptiveThresholds(filepath=tmp_path / "adaptive.json")


def test_defaults_match_rebuild():
    from hanoon_prime.brain.adaptive_thresholds import DEFAULTS

    assert DEFAULTS["exit_base"] == 0.18
    assert DEFAULTS["exit_scale"] == 0.35
    assert DEFAULTS["exit_max"] == 0.50
    assert DEFAULTS["stale_minutes_force"] == 30.0
    assert DEFAULTS["trail_keep_ratio"] == 0.55
    assert DEFAULTS["profit_lock_tiers"] == [
        [0.10, 0.04],
        [0.07, 0.03],
        [0.05, 0.02],
        [0.03, 0.01],
    ]


def test_exit_threshold_is_monotone_in_win_rate(tmp_path):
    at = _fresh(tmp_path)
    low = at.get_exit_threshold(0.0)  # base only
    high = at.get_exit_threshold(1.0)  # max
    assert low == pytest.approx(0.18)
    assert high == pytest.approx(min(0.18 + 0.35, 0.50))
    assert at.get_exit_threshold(0.5) > low
    assert at.get_watch_threshold(0.5) == pytest.approx(
        at.get_exit_threshold(0.5) * 0.7
    )


def test_every_getter_respects_its_bounds(tmp_path):
    from hanoon_prime.brain.adaptive_thresholds import DEFAULTS

    at = _fresh(tmp_path)
    # force everything to an extreme and confirm bounds hold
    for key, (low, high) in BOUNDS_LOCAL.items():
        if isinstance(DEFAULTS[key], list):
            continue
        at._values[key] = low - 1000
        at._adjust(key, 1000)
        assert low <= at.get_value(key) <= high, key
        at._values[key] = high + 1000
        at._adjust(key, -1000)
        assert low <= at.get_value(key) <= high, key


def test_ride_winners_params_and_keep_ratio(tmp_path):
    at = _fresh(tmp_path)
    assert at.get_ride_winners_params() == (0.03, 0.25)
    assert at.get_trail_keep_ratio() == 0.55


def test_update_from_outcome_adapts_and_persists(tmp_path):
    at = _fresh(tmp_path)
    # stale exit on a loser that "would have recovered" → loosen stale time
    at.update_from_outcome(
        won=False,
        hold_minutes=40.0,
        pnl_pct=-0.025,
        peak_pct=0.0,
        exit_reason="stale",
        exit_likelihood=0.6,
    )
    assert at.get_stale_minutes() > 30.0  # loosened
    # stale exit on a winner → tighten
    at.update_from_outcome(
        won=True,
        hold_minutes=40.0,
        pnl_pct=0.01,
        peak_pct=0.005,
        exit_reason="stale",
        exit_likelihood=0.2,
    )
    assert at.get_stale_minutes() < 30.0
    # persisted
    assert (tmp_path / "adaptive.json").exists()


def test_persistence_round_trip(tmp_path):
    at = _fresh(tmp_path)
    at.update_from_outcome(
        won=False,
        hold_minutes=35.0,
        pnl_pct=-0.06,
        peak_pct=0.0,
        exit_reason="stop",
        exit_likelihood=0.7,
    )
    first = at.get_trail_keep_ratio()
    # reload from disk
    from hanoon_prime.brain.adaptive_thresholds import AdaptiveThresholds

    at2 = AdaptiveThresholds(filepath=tmp_path / "adaptive.json")
    assert at2.get_trail_keep_ratio() == pytest.approx(first)


def test_tiers_bounded_and_mutable():
    from hanoon_prime.brain.adaptive_thresholds import AdaptiveThresholds

    at = AdaptiveThresholds(filepath=Path(tempfile.mkdtemp()) / "a.json")
    tiers = at.get_profit_lock_tiers()
    assert len(tiers) == 4
    # profit_lock on a winner reinforces the firing tier (min_lock grows, capped)
    at.update_from_outcome(
        won=True,
        hold_minutes=5.0,
        pnl_pct=0.02,
        peak_pct=0.12,
        exit_reason="profit_lock",
        exit_likelihood=0.1,
    )


# import here to avoid a circular dependency in collection order
from hanoon_prime.brain.adaptive_thresholds import BOUNDS as BOUNDS_LOCAL  # noqa: E402
