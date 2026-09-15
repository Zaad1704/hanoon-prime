"""tests/test_learning_telemetry — exploration/consolidation observability tests."""
from __future__ import annotations

from typing import Any

from hanoon_prime.brain.learning_telemetry import (
    consolidation,
    exploration,
    learning_state,
)
from hanoon_prime.brain.strategy_bandit import StrategyBandit


def _arms() -> dict[str, list[dict[str, Any]]]:
    return {
        "unknown": [
            {"strategy": "default", "mean": 0.51, "n": 60, "ab": [31.6, 30.4]},
            {"strategy": "trend-pullback", "mean": 0.55, "n": 12, "ab": [6.6, 5.4]},
        ],
        "vol": [
            {"strategy": "default", "mean": 0.45, "n": 20, "ab": [9.5, 12.5]},
        ],
    }


def _bandit(
    selects: int,
    explores: int,
    overrides: int,
    total_trials: int,
    eps: float,
) -> dict[str, Any]:
    return {
        "selects": selects,
        "explores": explores,
        "overrides": overrides,
        "total_trials": total_trials,
        "eps": eps,
    }


def test_cold_start_when_neither_exploring_nor_consolidating() -> None:
    block = exploration(_bandit(10, 1, 0, 3, 0.2), _arms())
    assert block["mode"] == "cold_start"
    assert block["explore_ratio"] == 0.1


def test_exploring_when_eps_above_floor() -> None:
    block = exploration(_bandit(200, 2, 0, 150, 0.12), _arms())
    assert block["mode"] == "exploring"
    assert block["eps"] == 0.12


def test_exploring_when_ratio_high_even_at_floor() -> None:
    block = exploration(_bandit(100, 30, 0, 80, 0.02), _arms())
    assert block["mode"] == "exploring"
    assert block["explore_ratio"] == 0.3


def test_consolidating_when_committed_and_low_ratio() -> None:
    block = exploration(_bandit(100, 2, 5, 92, 0.02), _arms())
    assert block["mode"] == "consolidating"
    assert block["committed_ratio"] == 1.0
    assert block["hypothesis_coverage"] == 0.0
    assert block["regret_estimate"] == 2.4


def test_regret_estimate_manual() -> None:
    rows = {
        "r1": [
            {"strategy": "a", "mean": 0.6, "n": 10, "ab": [6.5, 4.5]},
            {"strategy": "b", "mean": 0.4, "n": 5, "ab": [2.5, 3.5]},
            {"strategy": "c", "mean": 0.9, "n": 0, "ab": [1.0, 1.0]},
        ]
    }
    block = exploration(_bandit(15, 3, 0, 15, 0.0), rows)
    assert block["regret_estimate"] == 1.0
    assert block["regret_per_trial"] == round(1.0 / 15.0, 4)


def test_consolidation_parity_and_replay() -> None:
    realized = {"total": 200, "band_wins": {0: 36}, "band_losses": {1: 164}}
    sleep = {
        "cycle_count": 3,
        "last_replay": {
            "patterns_replayed": 24,
            "weights_updated": 100,
            "mean_weight_change": 0.0005,
        },
    }
    memory = {"pred_error_ema": 0.31}
    pillar = {"edge": -0.10}
    block = consolidation(_arms(), realized, sleep, memory, pillar)
    assert block["realized_win_rate"] == 0.18
    assert block["realized_trades"] == 200
    assert block["implied_edge"] == 0.0125
    assert block["parity_gap"] == round(0.0125 - (-0.10), 4)
    assert block["committed_arms"] == 3
    assert block["committed_ratio"] == 1.0
    assert block["sleep_cycles"] == 3
    assert block["last_replay"]["weights_updated"] == 100


def test_learning_state_maps_phases() -> None:
    exploring = exploration(_bandit(200, 40, 0, 150, 0.12), _arms())
    cons = consolidation(_arms(), {}, {}, {}, {})
    state = learning_state(exploring, cons)
    assert state["phase"] == "exploring"
    assert state["exploring"] is True
    assert state["consolidating"] is False

    consolidating = exploration(_bandit(100, 2, 5, 92, 0.02), _arms())
    replayed = consolidation(
        _arms(),
        {},
        {"cycle_count": 1, "last_replay": {"weights_updated": 50}},
        {},
        {},
    )
    state = learning_state(consolidating, replayed)
    assert state["phase"] == "consolidating_replayed"
    assert state["consolidating"] is True


def test_bandit_snapshot_arms_carry_ab_and_variance(tmp_path: Any) -> None:
    bandit = StrategyBandit(path=tmp_path / "bandit.json")
    bandit.update("unknown", "default", 0.05)
    bandit.update("unknown", "default", -0.02)
    arms = bandit.snapshot()["arms"]
    entry = arms["unknown"][0]
    assert "ab" in entry and len(entry["ab"]) == 2
    assert entry["variance"] >= 0.0
    assert entry["n"] == 2
