"""tests/test_telemetry_facade.py — Telemetry facade (behavior-identical).

The NeuromorphicBrain god object no longer renders its own telemetry:
orchestrator.snapshot() delegates to brain/telemetry_facade.build_brain_snapshot.
These tests pin the contract so the extraction cannot drift.
"""

from __future__ import annotations

from hanoon_prime.brain.orchestrator import NeuromorphicBrain
from hanoon_prime.brain.telemetry_facade import SNAPSHOT_KEYS, build_brain_snapshot


def test_delegation_is_identical():
    brain = NeuromorphicBrain()
    delegated = brain.snapshot()
    direct = build_brain_snapshot(brain)
    assert set(delegated) == set(direct)
    assert delegated["memory"] == direct["memory"]
    assert delegated["nash"] == direct["nash"]
    assert delegated["episodic_size"] == direct["episodic_size"]
    assert set(brain.snapshot()).issubset(set(SNAPSHOT_KEYS))


def test_snapshot_key_contract():
    brain = NeuromorphicBrain()
    snapshot = brain.snapshot()
    for key in ("memory", "realized", "nash", "advisor", "strategy_research"):
        assert key in snapshot
    assert "neuromorphic" in snapshot


def test_no_neuromorphic_is_handled():
    brain = NeuromorphicBrain()
    brain._neuromorphic = None
    brain.snapshot()


def test_sleep_engine_key_mirrors_engine():
    brain = NeuromorphicBrain()
    assert "sleep_engine" in build_brain_snapshot(brain)

    brain._sleep_engine = None
    assert "sleep_engine" not in build_brain_snapshot(brain)
