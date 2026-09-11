"""tests/test_pillar — directional conviction balance (the learning 'pillar').

Verifies inside_man.pillar_balance: the tipped-pillar signature (one direction
dominating vetoes/conviction — today's 34k SHORT direction_rejected @ 0.675)
vs the upright balanced state the webapp renders as a see-saw.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

import hanoon_prime.inspection.pillar as pillar
from hanoon_prime.inspection.checks import FAIL, OK, WARN, CheckResult
from hanoon_prime.inspection.ctx import InspectionContext
from hanoon_prime.inspection.pillar import pillar_balance


@pytest.fixture(autouse=True)
def _direction_mode_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default direction_mode to 'both' so short-skew FAIL tests stay honest.

    The policy-aware downgrade only fires for policy-blocked sides
    (direction_mode != "both"); the existing FAIL tests must still FAIL.
    """
    monkeypatch.setattr(pillar.TRADING_CONFIG, "direction_mode", "both")
    # Mock snapshot() so tests don't depend on a running bot's telemetry.
    monkeypatch.setattr(
        pillar, "snapshot", lambda _ctx: {"config": {"direction_mode": "both"}}
    )


def _ctx(tmp_path: Path) -> InspectionContext:
    return InspectionContext(base_dir=tmp_path)


def _write_log(ctx: InspectionContext, *lines: str) -> None:
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    ctx.log_path.write_text("".join(lines))


def _eval(token: str) -> str:
    """One EVAL log line carrying a single verdict token (live bot format).

    Token shape: TICKER:ACTION(score,side)[stage:reason] — balanced parens here
    on purpose so the fixture never masks a real tokenizer bug.
    """
    ts = datetime.datetime.now().strftime("%H:%M:%S") + ".000"
    return f"{ts} INFO juli EVAL {token}\n"


def test_pillar_ok_balanced(tmp_path: Path) -> None:
    """Equal long/short conviction -> ratio 0 -> pillar upright."""
    ctx = _ctx(tmp_path)
    _write_log(
        ctx,
        _eval("AAPL:HOLD(0.600,L)[trading_policy:enter]"),
        _eval("TSLA:SELL(-0.600,S)[trading_policy:exit]"),
    )
    r = pillar_balance(ctx)
    assert r.status == OK
    assert r.evidence["imbalance_ratio"] == 0.0
    assert r.evidence["long_conviction"] == 0.6
    assert r.evidence["short_conviction"] == 0.6


def test_pillar_ok_warmup_no_lines(tmp_path: Path) -> None:
    """No EVAL lines yet -> warming up, not a failure."""
    r = pillar_balance(_ctx(tmp_path))
    assert r.status == OK
    assert "warming up" in r.detail


def test_pillar_warn_no_scored_verdicts(tmp_path: Path) -> None:
    """EVAL lines exist but no signed conviction -> tipped data, not balance."""
    ctx = _ctx(tmp_path)
    _write_log(ctx, _eval("AAPL:VETOED(0.000,-)[validity:no_data]"))
    r = pillar_balance(ctx)
    assert r.status == WARN
    assert "no scored verdicts" in r.detail


def test_pillar_warn_tipping(tmp_path: Path) -> None:
    """Mild skew (ratio 0.4) -> WARN tipping."""
    ctx = _ctx(tmp_path)
    _write_log(
        ctx,
        _eval("AAPL:HOLD(0.600,L)[trading_policy:enter]"),
        _eval("TSLA:VETOED(-0.700,S)[trading_policy:direction_rejected]"),
        _eval("NVDA:VETOED(-0.700,S)[trading_policy:direction_rejected]"),
    )
    assert pillar_balance(ctx).status == WARN


def test_pillar_fail_short_dominated(tmp_path: Path) -> None:
    """5 SHORT vetoes vs 1 LONG -> ratio 0.707 -> pillar fallen."""
    ctx = _ctx(tmp_path)
    lines = [_eval("AAPL:HOLD(0.600,L)[trading_policy:enter]")]
    for _ in range(5):
        lines.append(_eval("TSLA:VETOED(-0.700,S)[trading_policy:direction_rejected]"))
    _write_log(ctx, *lines)
    r = pillar_balance(ctx)
    assert r.status == FAIL
    assert "SHORT" in r.detail
    assert r.evidence["imbalance_ratio"] > 0.6
    assert r.evidence["vetoes_short"] == 5


def test_pillar_fail_veto_skew(tmp_path: Path) -> None:
    """Even modest conviction but 3x veto skew -> FAIL."""
    ctx = _ctx(tmp_path)
    # long=1.8 (3 buys), short=1.5 (3 sells) -> ratio 0.1 -> OK on geometry,
    # but vetoes 3 short vs 1 long (>3x skew) -> FAIL.
    lines = [
        _eval("AAPL:BUY(0.600,L)[trading_policy:enter]"),
        _eval("MSFT:BUY(0.600,L)[trading_policy:enter]"),
        _eval("GOOG:BUY(0.600,L)[trading_policy:enter]"),
        _eval("AAPL:VETOED(-0.500,S)[trading_policy:direction_rejected]"),
        _eval("MSFT:VETOED(-0.500,S)[trading_policy:direction_rejected]"),
        _eval("GOOG:VETOED(-0.500,S)[trading_policy:direction_rejected]"),
    ]
    _write_log(ctx, *lines)
    r = pillar_balance(ctx)
    assert r.status == FAIL
    assert r.evidence["veto_skew"] >= 3.0


def test_pillar_evidence_has_learning_fields(tmp_path: Path) -> None:
    """Webapp feed carries the learning-context fields for correlation."""
    ctx = _ctx(tmp_path)
    _write_log(
        ctx,
        _eval("AAPL:HOLD(0.600,L)[trading_policy:enter]"),
        _eval("TSLA:SELL(-0.600,S)[trading_policy:exit]"),
    )
    r = pillar_balance(ctx)
    assert isinstance(r, CheckResult)
    for key in ("halim_modifier", "learning_active", "band", "net"):
        assert key in r.evidence


def test_pillar_joint_is_inside_man() -> None:
    """Registered under the inside_man joint (Inside Man confirms the pillar)."""
    from hanoon_prime.inspection.joints import SPECS

    spec = next(s for s in SPECS if s.name == "pillar_balance")
    assert spec.joint == "inside_man"
    assert spec.report is True


def test_pillar_long_only_short_skew_is_policy_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """shorts blocked (long_only): SHORT veto skew is configured, not a defect."""
    monkeypatch.setattr(pillar.TRADING_CONFIG, "direction_mode", "long_only")
    monkeypatch.setattr(
        pillar, "snapshot", lambda _ctx: {"config": {"direction_mode": "long_only"}}
    )
    ctx = _ctx(tmp_path)
    lines = [_eval("AAPL:HOLD(0.600,L)[trading_policy:enter]")]
    for _ in range(5):
        lines.append(_eval("TSLA:VETOED(-0.700,S)[trading_policy:direction_rejected]"))
    _write_log(ctx, *lines)
    r = pillar_balance(ctx)
    assert r.status == WARN  # NOT FAIL — shorts intentionally disabled
    assert "long_only" in r.detail
    assert r.evidence["direction_mode"] == "long_only"


def test_pillar_both_mode_low_penny_score_short_is_policy_warn(
    tmp_path: Path,
) -> None:
    """low_penny_score SHORT vetoes under direction_mode='both' -> WARN.

    The reason-aware check recognises that low_penny_score is a trading-policy
    gate that blocks both directions equally (penny-bar confidence floor),
    so the SHORT veto skew is configured policy, not a cortex defect — even
    when direction_mode='both' would have allowed shorts.
    """
    ctx = _ctx(tmp_path)
    lines = [_eval("AAPL:HOLD(0.600,L)[trading_policy:enter]")]
    for _ in range(5):
        lines.append(_eval("TSLA:VETOED(-0.700,S)[trading_policy:low_penny_score]"))
    _write_log(ctx, *lines)
    r = pillar_balance(ctx)
    assert r.status == WARN  # policy-gated, not a brain defect
    assert "policy-gated" in r.detail
    assert r.evidence["policy_vetoes_short"] == 5
    assert r.evidence["vetoes_short"] == 5
    assert r.evidence["direction_mode"] == "both"
