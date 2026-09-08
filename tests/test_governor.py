"""tests/test_governor — entry pacing (cycle budget + reuse cooldown).

Ports the FIX-2026-09-08-02 throttle semantics that ib_cycle used to own:
a hard cap on bracket entries per cycle and a per-ticker re-entry cooldown.
"""

import time

from hanoon_prime.brain.policy.governor import Governor
from hanoon_prime.immune import ENTRY_REUSE_COOLDOWN_SEC, MAX_ENTRIES_PER_CYCLE


def test_budget_cap_rejects_after_limit():
    g = Governor()
    g.begin_cycle()
    for i in range(MAX_ENTRIES_PER_CYCLE):
        ok, reason = g.may_enter(f"T{i}")
        assert ok, f"T{i} should be admitted; reason={reason}"
    ok, reason = g.may_enter("T_EXTRA")
    assert ok is False
    assert reason == "cycle_budget"


def test_begin_cycle_resets_budget():
    g = Governor()
    g.begin_cycle()
    for i in range(MAX_ENTRIES_PER_CYCLE + 1):
        g.may_enter(f"T{i}")
    g.begin_cycle()
    ok, reason = g.may_enter("T_AGAIN")
    assert ok is True, reason


def test_reuse_cooldown_blocks_reentry():
    g = Governor()
    g.begin_cycle()
    assert g.may_enter("NVD")[0] is True
    g.note_entry("NVD")
    g.begin_cycle()
    ok, reason = g.may_enter("NVD")
    assert ok is False
    assert reason == "reuse_cooldown"


def test_cooldown_expires():
    g = Governor()
    g.begin_cycle()
    assert g.may_enter("NVD")[0] is True
    g.note_entry("NVD")
    g._last_entry["NVD"] = time.time() - (ENTRY_REUSE_COOLDOWN_SEC + 1.0)
    g.begin_cycle()
    assert g.may_enter("NVD")[0] is True


def test_unentered_tickers_have_no_cooldown():
    g = Governor()
    g.begin_cycle()
    ok, reason = g.may_enter("FRESH")
    assert ok is True and reason == "ok"


def test_budget_only_consumed_on_approval():
    g = Governor()
    g.begin_cycle()
    assert g.may_enter("NVD")[0] is True
    g.note_entry("NVD")
    g.begin_cycle()
    ok, reason = g.may_enter("NVD")
    assert ok is False and reason == "reuse_cooldown"
    ok, _ = g.may_enter("TSLA")
    assert ok is True, "denied re-entry must not consume the cycle budget"
