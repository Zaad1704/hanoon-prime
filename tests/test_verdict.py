"""tests/test_verdict — the brain decision contract.

One Verdict is produced for every evaluated ticker; a veto is observable
data (reason + stage), never a silent omission.
"""

from types import SimpleNamespace

from hanoon_prime.brain.policy.verdict import ENTER, HOLD, VETOED, Verdict


def test_constants_are_distinct_verdict_strings():
    assert {ENTER, HOLD, VETOED} == {"ENTER", "HOLD", "VETOED"}


def test_verdict_to_dict_excludes_execution_context():
    v = Verdict(
        ticker="NVD",
        action=ENTER,
        reason="ok",
        stage="governor",
        score=0.9,
        direction=1,
        horizon="scalp",
        thought=SimpleNamespace(direction=1, score=0.9),
    )
    d = v.to_dict()
    assert d["ticker"] == "NVD"
    assert d["action"] == ENTER
    assert d["reason"] == "ok"
    assert d["stage"] == "governor"
    assert d["horizon"] == "scalp"
    assert d["score"] == 0.9
    assert "thought" not in d


def test_verdict_as_execution_context_returns_thought_shape():
    v = Verdict(ticker="NVD", action=ENTER, score=0.9, direction=1)
    ctx = v.as_execution_context()
    assert ctx.direction == 1
    assert ctx.score == 0.9
    assert ctx.verdict == ENTER


def test_hold_verdict_defaults():
    v = Verdict(ticker="TSLA")
    assert v.action == HOLD
    assert v.reason == ""
    assert v.stage == ""
    assert v.horizon == "scalp"
    assert v.direction == 0


def test_verdict_holds_sizing_ref():
    from hanoon_prime.brain.risk import SizingResult

    sizing = SizingResult(shares=10, risk_pass=True)
    v = Verdict(ticker="NVD", action=ENTER, sizing=sizing)
    assert v.sizing is sizing
    assert v.sizing.shares == 10
