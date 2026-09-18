"""tests/test_vitals_log.py — rotating CSV vitals trend writer."""

from __future__ import annotations

import time
from typing import Any

from hanoon_prime.monitor.vitals_log import VitalsLog, _flatten


def _snap(
    market_open: bool = True,
    decision_count: int = 3,
    feed_age: float = 2.0,
    stall_cycles: int = 0,
    healthy: bool = True,
) -> dict[str, Any]:
    return {
        "healthy": healthy,
        "failing": {} if healthy else {"bars_fresh": "stale"},
        "vitals": {
            "ts": time.time(),
            "market_open": market_open,
            "session": "rth",
            "session_active": market_open,
            "ib_connected": True,
            "decision_count": decision_count,
            "feed_age": feed_age,
        },
        "stall_cycles": stall_cycles,
    }


def test_flatten_reduces_snapshot() -> None:
    row = _flatten(_snap())
    assert row["market_open"] is True
    assert row["session"] == "rth"
    assert row["decision_count"] == 3
    assert row["healthy"] is True
    assert set(row) == {
        "ts",
        "market_open",
        "session",
        "session_active",
        "ib_connected",
        "decision_count",
        "feed_age",
        "stall_cycles",
        "healthy",
    }


def test_record_then_tail(tmp_path) -> None:
    v = VitalsLog(tmp_path)
    for _ in range(3):
        v.record(_snap(decision_count=5, market_open=True))
    rows = v.tail(10)
    assert len(rows) == 3
    assert all(r["decision_count"] == 5 for r in rows)
    assert all(r["healthy"] is True for r in rows)


def test_tail_is_newest_first_within_set(tmp_path) -> None:
    v = VitalsLog(tmp_path)
    for i in range(5):
        v.record(_snap(decision_count=i))
    rows = v.tail(3)
    assert [r["decision_count"] for r in rows] == [2, 3, 4]


def test_since_filters_by_ts(tmp_path) -> None:
    v = VitalsLog(tmp_path)
    base = 1_000_000.0
    for i in range(4):
        snap = _snap(decision_count=i)
        snap["vitals"]["ts"] = base + i  # distinct, ordered timestamps
        v.record(snap)
    first = v.tail(4)[0]["ts"]
    assert first == base
    rows = v.since(first)
    assert len(rows) == 4
    rows = v.since(base + 1)
    assert len(rows) == 3


def test_rotation_bounds_rows_per_file(tmp_path) -> None:
    v = VitalsLog(tmp_path, rows_per_file=2, max_files=3)
    for i in range(5):
        v.record(_snap(decision_count=i))
    files = sorted(p.name for p in tmp_path.glob("vitals_*.csv"))
    assert len(files) >= 2
    rows = v.tail(50)
    assert len(rows) == 5
    assert [r["decision_count"] for r in rows] == [0, 1, 2, 3, 4]


def test_prune_bounds_file_count(tmp_path) -> None:
    v = VitalsLog(tmp_path, rows_per_file=1, max_files=2)
    for i in range(6):
        v.record(_snap(decision_count=i))
    files = sorted(tmp_path.glob("vitals_*.csv"))
    assert len(files) <= 2
    rows = v.tail(50)
    # oldest pruned data is gone; only the two retained files remain
    assert len(rows) <= 2


def test_records_bool_round_trip(tmp_path) -> None:
    v = VitalsLog(tmp_path)
    v.record(_snap(market_open=False, healthy=False))
    rows = v.tail(1)
    assert rows[0]["market_open"] is False
    assert rows[0]["session_active"] is False
    assert rows[0]["healthy"] is False
