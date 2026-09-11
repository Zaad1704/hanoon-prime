"""tests/test_brain_pipeline — decide_entry is THE single decision point.

Every evaluated ticker returns a Verdict; a halted brain yields a visible
VETOED (never silence); the governor caps the cycle; invalid snaps are
rejected at the validity stage.
"""

import time
from dataclasses import replace
from types import SimpleNamespace

from hanoon_prime.brain.orchestrator import NeuromorphicBrain
from hanoon_prime.brain.policy.verdict import ENTER, HOLD, VETOED
from hanoon_prime.brain.risk import SizingResult
from hanoon_prime.brain.shared_state import DEFAULT_POLICY_STATE


def _snap():
    return {
        "prices": [1.0] * 40,
        "bid": 100.0,
        "ask": 100.1,
        "mid": 100.05,
        "last": 100.0,
        "ts": time.time(),
        "volume": 1000,
    }


def _brain():
    b = NeuromorphicBrain(enable_neuromorphic=False)
    b.tick = lambda alpha, ticker, **kw: _fake_result()
    b.state.update(
        policy_state={
            **DEFAULT_POLICY_STATE,
            "equity_synced": True,
            "equity": 100_000.0,
        },
        account_feed={"equity": 100_000.0, "daily_pnl": 0.0, "positions": {}},
    )
    return b


def _fake_result(side="BUY", score=0.9, direction=1):
    return {
        "price": 100.0,
        "verdict": side,
        "signals": {"entry": 1.0},
        "thought": SimpleNamespace(score=score, direction=direction, confidence=0.5),
    }


def test_halted_state_produces_visible_vetoed():
    b = _brain()
    b.state.update(
        policy_state={
            **DEFAULT_POLICY_STATE,
            "authorized": False,
            "halted": True,
            "pause_reason": "daily_loss_limit",
        }
    )
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == VETOED
    assert v.reason == "daily_loss_limit"
    assert v.stage == "safety"
    assert v.ticker == "NVD"
    assert v.score == 0.9  # cortex conviction surfaces on the halted veto


def test_session_disabled_veto_preserves_score():
    """A disabled session still records real |score| on its veto."""
    b = _brain()
    b.tick = lambda alpha, ticker, **kw: _fake_result()
    b.trading_policy = replace(b.trading_policy, session_rth=False)
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == VETOED
    assert v.reason == "session_disabled"
    assert v.score == 0.9


def test_low_penny_veto_preserves_score():
    """A sub-dollar tick vetoed as low_penny still carries the cortex score."""
    b = _brain()
    b.tick = lambda alpha, ticker, **kw: _fake_result(score=0.5)
    snap = _snap()
    snap["last"] = 0.50  # below PENNY_PRICE (1.00); |score| 0.5 < PENNY_SCORE_BAR 0.85
    v = b.decide_entry("PENY", snap, {}, "rth")
    assert v.action == VETOED
    assert v.reason == "low_penny_score"
    assert v.score == 0.5


def test_deliberation_coherence_populates_trace(monkeypatch):
    """The bounded Deliberator runs in-path and publishes its CoT trace.

    Diagnostic only: the candidate score is published, not blended into raw,
    so the verdict is byte-identical. Flag-gated via the orchestrator module
    binding so the default-off path is a true no-op.
    """
    import hanoon_prime.brain.orchestrator as orch
    from hanoon_prime.cortex import Thought

    monkeypatch.setattr(orch, "DELIBERATION_TRACE_ENABLED", True)
    b = _brain()
    thought = Thought(verdict="BUY", score=0.62, direction=1, confidence=0.8)
    ctx = {"base": thought, "thinker_mod": 0.04}
    b._deliberation_coherence(ctx, 1.1, 0.01, 0.06, "TIC")
    assert "deliberation_trace" in ctx
    assert "deliberation_candidate_score" in ctx
    trace = ctx["deliberation_trace"]
    assert "halim_mod" in trace and "episodic_mod" in trace
    assert b.state.get("deliberation_trace") is not None


def test_deliberation_coherence_disabled_is_noop():
    """With the flag off (default), the diagnostic path is a no-op."""
    from hanoon_prime.cortex import Thought

    b = _brain()
    thought = Thought(verdict="BUY", score=0.62, direction=1, confidence=0.8)
    ctx = {"base": thought, "thinker_mod": 0.04}
    b._deliberation_coherence(ctx, 1.1, 0.01, 0.06, "TIC")
    assert "deliberation_trace" not in ctx
    assert "deliberation_candidate_score" not in ctx


def test_valid_edge_admitted_with_size():
    b = _brain()
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == ENTER
    assert v.sizing is not None and v.sizing.shares > 0
    assert v.stop is not None and v.target is not None


def test_direction_vetoed():
    b = _brain()
    b.tick = lambda alpha, ticker, **kw: _fake_result(side="SELL", direction=-1)
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == VETOED and v.reason == "direction_rejected"
    # Cortex conviction must survive ONTO the vetoed Verdict so the log/journal/
    # inspection reflect real conviction — not a misleading 0.000 default.
    assert v.score == 0.9


def test_zero_conviction_direction_is_no_signal():
    """A direction with |score| below the conviction floor is a HOLD,
    not a confident-sounding veto — no short/long claimed at score 0.000."""
    b = _brain()
    b.tick = lambda alpha, ticker, **kw: _fake_result(
        side="SELL", direction=-1, score=0.0
    )
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == HOLD and v.reason == "no_signal"


def test_no_signal_is_hold():
    b = _brain()
    b.tick = lambda alpha, ticker, **kw: _fake_result(
        side="HOLD", score=0.5, direction=0
    )
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == HOLD and v.reason == "no_signal"


def test_governor_cap_vetoes_third():
    b = _brain()
    for i in range(2):
        assert b.decide_entry(f"T{i}", _snap(), {}, "rth").action == ENTER
    v3 = b.decide_entry("T3", _snap(), {}, "rth")
    assert v3.action == VETOED and v3.reason == "cycle_budget"


def test_invalid_snapshot_vetoed():
    b = _brain()
    v = b.decide_entry("NVD", {"prices": [1.0] * 3}, {}, "rth")
    assert v.action == VETOED and v.reason == "no_data"


def test_probe_override_bypasses_halt():
    b = _brain()
    from hanoon_prime.brain.probe_recovery import ProbeRecovery

    b.probe = ProbeRecovery()
    b.probe.maybe_probe = lambda score, bid, ask, losses: True
    b.state.update(
        policy_state={
            **DEFAULT_POLICY_STATE,
            "authorized": False,
            "halted": True,
            "pause_reason": "consecutive_losses",
            "consecutive_losses": 3,
        }
    )
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert v.action == ENTER
    assert v.reason == "probe_recovery" and v.stage == "probe_recovery"


def test_sizing_result_type_used_when_fake_tick_omits_it():
    """decide_entry must still size from thought when tick() has no sizing."""
    b = _brain()
    v = b.decide_entry("NVD", _snap(), {}, "rth")
    assert isinstance(v.sizing, SizingResult)
