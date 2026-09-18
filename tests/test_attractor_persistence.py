"""tests/test_attractor_persistence.py — B1: AttractorMemory survives restarts.

AttractorMemory now persists best-effort to a ``filepath`` (STATE_DIR for the
live bridge) and reloads on construction, so sleep replay has cross-restart
patterns after the C1 repair.
"""

from __future__ import annotations

import json

import pytest

from hanoon_prime.brain.neurons.attractor import AttractorMemory
from hanoon_prime.brain.neurons.bridge import NeuromorphicBridge


def test_roundtrip_persists_and_reloads(tmp_path):
    path = tmp_path / "attractor_memory.json"
    mem = AttractorMemory(filepath=path)
    mem.store("T", [0.9] * 11, won=True, pnl_pct=1.2)
    mem.store("T", [0.9] * 11, won=True, pnl_pct=0.8)  # same basin -> wins/tc advance
    mem.store("AAPL", [0.1] * 11, won=False, pnl_pct=-0.5)

    reloaded = AttractorMemory(filepath=path)
    assert len(reloaded.get_patterns()) == 2
    by_ticker = {a.ticker: a for a in reloaded.get_patterns()}
    assert by_ticker["T"].wins == 1
    assert by_ticker["T"].trade_count == 1
    assert by_ticker["T"].pnl_pct == 0.8
    assert by_ticker["AAPL"].won is False


def test_merge_then_persist_updates_counts(tmp_path):
    path = tmp_path / "attractor_memory.json"
    mem = AttractorMemory(filepath=path)
    mem.store("T", [0.9] * 11, won=True, pnl_pct=1.2)  # create: tc=0, wins=0
    mem.store("T", [0.91] * 11, won=False, pnl_pct=-0.4)  # merge: losses=1
    mem.store("T", [0.92] * 11, won=True, pnl_pct=0.5)  # merge: wins=1

    reloaded = AttractorMemory(filepath=path)
    att = reloaded.get_patterns(ticker="T")[0]
    assert att.trade_count == 2
    assert att.wins == 1
    assert att.losses == 1
    assert att.won is True


def test_corrupt_file_loads_gracefully(tmp_path):
    path = tmp_path / "attractor_memory.json"
    path.write_text("{not json", encoding="utf-8")
    mem = AttractorMemory(filepath=path)
    assert len(mem) == 0


def test_bridge_default_is_in_memory_until_orchestrator_opt_in():
    bridge = NeuromorphicBridge()
    assert bridge.memory._filepath is None
    default = NeuromorphicBridge._memory_path()
    assert default.name == "attractor_memory.json"
    assert "runtime" in str(default)


def test_saved_blob_is_json_declarative(tmp_path):
    from hanoon_prime.brain.neurons.attractor import Attractor

    path = tmp_path / "attractor_memory.json"
    mem = AttractorMemory(filepath=path)
    payload = json.dumps(
        {
            "attractors": [
                {
                    "ticker": "T",
                    "center": [0.5, 0.5],
                    "won": True,
                    "pnl_pct": 0.5,
                    "trade_count": 1,
                    "wins": 1,
                    "losses": 0,
                }
            ]
        }
    )
    path.write_text(payload, encoding="utf-8")
    reloaded = AttractorMemory(filepath=path)
    assert isinstance(reloaded.get_patterns()[0], Attractor)
    assert reloaded.get_patterns()[0].wins == 1
