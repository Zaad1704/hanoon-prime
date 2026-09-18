"""tests/test_trust.py — weighted composite pipeline trust score."""

from __future__ import annotations

from hanoon_prime.monitor.trust import (
    DEGRADED_FLOOR,
    HEALTHY_FLOOR,
    assess,
    checks_from_snapshot,
    score_checks,
    status_for,
)


def _snap(failing: dict[str, str]) -> dict[str, object]:
    return {
        "healthy": not failing,
        "failing": failing,
        "vitals": {"decision_count": 4, "feed_age": 5.0, "market_open": True},
        "stall_cycles": 0,
    }


def test_all_green_is_healthy() -> None:
    out = assess(_snap({}))
    assert out["status"] == "HEALTHY"
    assert out["score"] >= HEALTHY_FLOOR
    assert out["failing"] == []


def test_one_failure_is_degraded() -> None:
    out = assess(_snap({"bars_fresh": "stale"}))
    assert out["status"] == "DEGRADED"
    assert 0 <= out["score"] < HEALTHY_FLOOR
    assert "bars_fresh" in out["failing"]


def test_brain_stall_is_critical() -> None:
    # ib + bars + brain all down: weighted score falls below DEGRADED_FLOOR
    out = assess(
        _snap(
            {
                "ib_connected": "Gateway link",
                "brain_advancing": "15 cycles",
                "bars_fresh": "stale",
            }
        )
    )
    assert out["status"] == "CRITICAL"
    assert out["score"] < DEGRADED_FLOOR
    assert set(out["failing"]) == {"ib_connected", "brain_advancing", "bars_fresh"}


def test_weights_present_for_all_signal_checks() -> None:
    checks = checks_from_snapshot(_snap({}))
    names = [c.name for c in checks]
    assert names == [
        "ib_connected",
        "bars_fresh",
        "brain_advancing",
        "entry_evals",
        "subs_present",
    ]
    assert all(c.weight >= 1 for c in checks)


def test_status_for_thresholds() -> None:
    assert status_for(HEALTHY_FLOOR + 1) == "HEALTHY"
    assert status_for(DEGRADED_FLOOR + 1) == "DEGRADED"
    assert status_for(DEGRADED_FLOOR - 1) == "CRITICAL"


def test_score_checks_accepts_custom_list() -> None:
    out = score_checks(
        [
            checks_from_snapshot(_snap({}))[0],
            type("TC", (), {"name": "x", "ok": False, "weight": 1, "detail": "y"})(),
        ]
    )
    assert out["score"] > 0
    assert "x" in out["failing"]
