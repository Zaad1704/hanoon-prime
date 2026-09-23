"""tests/test_strategy_research — strategy research system tests."""

from __future__ import annotations

import json
import time
from typing import Any

from hanoon_prime.brain.strategy_bandit import DEFAULT_STRATEGY, StrategyBandit
from hanoon_prime.brain.strategy_registry import StrategyRegistry
from hanoon_prime.brain.strategy_research import (
    StrategyResearch,
    _candidates,
    _research_query,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _reg(tmp_path, **overrides):
    p = tmp_path / "registry.json"
    return StrategyRegistry(path=p)


def _bandit(tmp_path):
    p = tmp_path / "bandit.json"
    return StrategyBandit(path=p)


def _ok_payload(strategies=None):
    if strategies is None:
        strategies = [
            {
                "name": "test-pullback",
                "thesis": "Buy dip in uptrend",
                "entry": "price < 20ema",
                "exit": "price > 50ema",
                "risk": "2xATR stop",
                "conditions": "trend_up regime",
                "regime": "trend_up",
                "confidence": 0.6,
            }
        ]
    return {
        "ok": True,
        "strategies": strategies,
        "regime_hint": "trend_up",
        "evidence": True,
    }


def _fake_query(payload):
    def _q(base_url: str, query: str) -> dict:
        return payload

    return _q


def _boom_query(base_url: str, query: str) -> dict:
    raise RuntimeError("halim down")


# ── Registry tests ───────────────────────────────────────────────────


def test_registry_seeds_priors(tmp_path):
    reg = _reg(tmp_path)
    ids = reg.ids_for("trend_up")
    names = [reg.get(i)["name"] for i in ids]
    assert "trend-pullback" in names


def test_registry_dedupes_by_slug(tmp_path):
    reg = _reg(tmp_path)
    n = reg.count()
    id1 = reg.ingest(
        {"name": "new-alpha", "entry": "go", "exit": "stop", "risk": "stop"}
    )
    id2 = reg.ingest(
        {"name": "new-alpha", "entry": "go", "exit": "stop", "risk": "stop"}
    )
    assert id1 == id2
    assert reg.count() == n + 1


def test_registry_pool_cap(tmp_path):
    from hanoon_prime.brain.learning_config import STRATEGY_MAX_POOL

    reg = _reg(tmp_path)
    for i in range(STRATEGY_MAX_POOL + 2):
        reg.ingest(
            {
                "name": f"strat-{i}",
                "entry": "go",
                "exit": "stop",
                "risk": "stop",
            }
        )
    assert reg.count() <= STRATEGY_MAX_POOL


def test_registry_nudge_bounds(tmp_path):
    reg = _reg(tmp_path)
    nudge = reg.nudge_for(DEFAULT_STRATEGY)
    assert nudge["sizing"] == 1.0
    assert nudge["score_mod"] == 0.0


def test_registry_persistence(tmp_path):
    reg1 = _reg(tmp_path)
    sid = reg1.ingest(
        {"name": "pers-strat", "entry": "go", "exit": "stop", "risk": "stop"}
    )
    assert sid
    reg2 = _reg(tmp_path)
    assert reg2.get(sid) is not None


# ── Bandit tests ─────────────────────────────────────────────────────


def test_bandit_select_returns_default_when_no_ids():
    b = StrategyBandit()
    sid, reason = b.select("trend_up", [])
    assert sid == DEFAULT_STRATEGY
    assert reason == "default"


def test_bandit_update_and_select(tmp_path):
    b = _bandit(tmp_path)
    b.update("trend_up", "test-strat", 0.5)
    sid, _reason = b.select("trend_up", ["test-strat"])
    assert sid in (DEFAULT_STRATEGY, "test-strat")


def test_bandit_override_gated_by_min_samples(tmp_path):
    b = _bandit(tmp_path)
    for _ in range(20):
        b.update("range", "strat-1", 0.5)
        b.update("range", DEFAULT_STRATEGY, -0.1)
    wins, total = 0, 30
    for i in range(total):
        if i % 2 == 0:
            b.update("range", "strat-1", 0.4)
            wins += 1
        else:
            b.update("range", DEFAULT_STRATEGY, -0.2)
    sid, reason = b.select("range", ["strat-1"])
    assert sid == "strat-1" or reason in ("default", "explore")


# ── Research client tests ────────────────────────────────────────────


def test_maybe_run_throttles(tmp_path):
    reg = _reg(tmp_path)
    sr = StrategyResearch(reg, interval_sec=9999)
    r1 = sr.maybe_run(query_fn=_fake_query(_ok_payload()))
    assert len(r1) >= 1
    r2 = sr.maybe_run(query_fn=_fake_query(_ok_payload()))
    assert r2 == []  # throttled


def test_maybe_run_disables(tmp_path):
    reg = _reg(tmp_path)
    sr = StrategyResearch(reg)
    assert sr.maybe_run(query_fn=_fake_query(_ok_payload()), enabled=False) == []


def test_maybe_run_handles_query_failure(tmp_path):
    reg = _reg(tmp_path)
    sr = StrategyResearch(reg, interval_sec=0)
    sr._last_run = 0.0
    result = sr.maybe_run(query_fn=_boom_query)
    assert result == []


def test_candidates_extracts_list():
    assert _candidates({"ok": True, "strategies": [{"name": "x"}]}) == [{"name": "x"}]
    assert _candidates({"ok": True}) == []


def test_maybe_run_ingests_into_registry(tmp_path):
    reg = _reg(tmp_path)
    n_before = reg.count()
    sr = StrategyResearch(reg, interval_sec=0)
    sr._last_run = 0.0
    ingested = sr.maybe_run(query_fn=_fake_query(_ok_payload()))
    assert reg.count() == n_before + len(ingested)


def test_registry_force_seeds_priors_at_capacity(tmp_path):
    """A full pool of researched strategies still admits seeded priors."""
    from hanoon_prime.brain.learning_config import STRATEGY_MAX_POOL
    from hanoon_prime.brain.strategy_priors import seeded_priors

    reg = _reg(tmp_path)
    while reg.count() < STRATEGY_MAX_POOL:
        assert reg.ingest(
            {
                "name": f"fill-{reg.count()}",
                "entry": "go",
                "exit": "stop",
                "risk": "stop",
            }
        )
    for prior in seeded_priors():
        assert reg.get(_slug_id(prior["name"])) is not None, prior["name"]
    assert reg.count() <= STRATEGY_MAX_POOL


def _slug_id(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]
