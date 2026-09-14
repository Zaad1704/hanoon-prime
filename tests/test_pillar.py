"""tests/test_pillar — the win/loss pillar (edge vs break-even).

Verifies inside_man.pillar_balance: the upright (edge >= 0) success state vs
the side-agnostic lean that tips/falls the pillar. Juli's awareness is
published in ``brain_state.pillar``; the check reads it, or derives it from
the persisted realized ``rr_samples`` when the live pulse is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hanoon_prime.inspection.pillar as pillar
from hanoon_prime.brain.pillar_awareness import (
    compute_pillar_awareness,
    win_loss_record,
)
from hanoon_prime.inspection.checks import FAIL, OK, WARN, CheckResult
from hanoon_prime.inspection.ctx import InspectionContext
from hanoon_prime.inspection.pillar import pillar_balance


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No live telemetry in unit tests — pin direction_mode to 'both'."""
    monkeypatch.setattr(
        pillar, "snapshot", lambda _ctx: {"config": {"direction_mode": "both"}}
    )


def _ctx(tmp_path: Path) -> InspectionContext:
    return InspectionContext(base_dir=tmp_path)


def _record(wins: int, losses: int, win: float = 0.02, loss: float = 0.01) -> dict:
    """Realized record with a 2:1 payoff (break-even win rate = 1/3)."""
    return win_loss_record([(1, win, 1)] * wins + [(0, -loss, 1)] * losses)


def _pillar(wins: int, losses: int, **kw: float) -> dict:
    return compute_pillar_awareness(_record(wins, losses, **kw))


def _write_state(
    ctx: InspectionContext,
    *,
    brain_state: dict | None = None,
    realized: dict | None = None,
) -> None:
    ctx.runtime.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if brain_state is not None:
        data["brain_state"] = brain_state
    if realized is not None:
        data["realized"] = realized
    ctx.state_path.write_text(json.dumps(data))


# ── status mapping (edge vs break-even) ───────────────────────────────


def test_pillar_upright_is_ok(tmp_path: Path) -> None:
    """5W-7L at 2:1 payoff: win rate .417 clears break-even .333 -> upright."""
    ctx = _ctx(tmp_path)
    _write_state(ctx, brain_state={"pillar": _pillar(5, 7)})
    r = pillar_balance(ctx)
    assert r.status == OK
    assert r.evidence["pillar_state"] == "upright"
    assert r.evidence["upright"] is True
    assert r.evidence["edge"] > 0
    assert r.evidence["tilt"] == 0.0


def test_pillar_exactly_break_even_is_upright(tmp_path: Path) -> None:
    """4W-8L: win rate == break-even -> edge 0 -> still upright."""
    ctx = _ctx(tmp_path)
    _write_state(ctx, brain_state={"pillar": _pillar(4, 8)})
    r = pillar_balance(ctx)
    assert r.status == OK
    assert r.evidence["edge"] == 0.0


def test_pillar_tipping_is_warn(tmp_path: Path) -> None:
    """3W-9L: edge -.083 (> -0.10) -> tipping, not yet fallen."""
    ctx = _ctx(tmp_path)
    _write_state(ctx, brain_state={"pillar": _pillar(3, 9)})
    r = pillar_balance(ctx)
    assert r.status == WARN
    assert r.evidence["pillar_state"] == "tipping"


def test_pillar_fallen_is_fail(tmp_path: Path) -> None:
    """2W-10L: edge -.167 <= -0.10 -> pillar fallen."""
    ctx = _ctx(tmp_path)
    _write_state(ctx, brain_state={"pillar": _pillar(2, 10)})
    r = pillar_balance(ctx)
    assert r.status == FAIL
    assert r.evidence["pillar_state"] == "fallen"
    assert r.evidence["tilt"] == 1.0


def test_pillar_all_losses_falls(tmp_path: Path) -> None:
    """0W-12L: break-even 1.0 -> edge -1.0 -> fallen, not a false upright."""
    ctx = _ctx(tmp_path)
    _write_state(ctx, brain_state={"pillar": _pillar(0, 12)})
    r = pillar_balance(ctx)
    assert r.status == FAIL
    assert r.evidence["edge"] == -1.0


def test_pillar_no_state_is_warming(tmp_path: Path) -> None:
    r = pillar_balance(_ctx(tmp_path))
    assert r.status == OK
    assert "warming up" in r.detail
    assert r.evidence["pillar_state"] == "warming"


def test_pillar_few_trades_is_warming(tmp_path: Path) -> None:
    """Below PILLAR_MIN_TRADES the pillar is still warming up."""
    ctx = _ctx(tmp_path)
    _write_state(ctx, brain_state={"pillar": _pillar(1, 1)})
    r = pillar_balance(ctx)
    assert r.status == OK
    assert r.evidence["pillar_state"] == "warming"


def test_pillar_winning_record_is_upright(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _write_state(ctx, brain_state={"pillar": _pillar(9, 3)})
    r = pillar_balance(ctx)
    assert r.status == OK
    assert r.evidence["edge"] > 0
    assert r.evidence["win_loss_record"] == "9W-3L"


# ── derivation + evidence shape ───────────────────────────────────────


def test_pillar_derives_from_realized_snapshot(tmp_path: Path) -> None:
    """Without a live pulse, the check folds the persisted rr_samples."""
    ctx = _ctx(tmp_path)
    samples = [[1, 0.02, 1]] * 5 + [[0, -0.01, 1]] * 7
    _write_state(ctx, realized={"rr_samples": samples})
    r = pillar_balance(ctx)
    assert r.status == OK
    assert r.evidence["win_loss_record"] == "5W-7L"
    assert r.evidence["wins"] == 5
    assert r.evidence["losses"] == 7


def test_pillar_evidence_has_all_fields(tmp_path: Path) -> None:
    """Webapp feed carries win/loss awareness + learning context."""
    ctx = _ctx(tmp_path)
    _write_state(ctx, brain_state={"pillar": _pillar(5, 7)})
    r = pillar_balance(ctx)
    assert isinstance(r, CheckResult)
    for key in (
        "pillar_state",
        "upright",
        "tilt",
        "edge",
        "win_rate",
        "break_even_wr",
        "wins",
        "losses",
        "trades",
        "win_loss_record",
        "net_pnl",
        "r_r",
        "halim_modifier",
        "learning_active",
        "band",
        "imbalance_ratio",
    ):
        assert key in r.evidence


def test_pillar_keeps_directional_secondary_evidence(tmp_path: Path) -> None:
    """Directional conviction/veto geometry is retained as context."""
    ctx = _ctx(tmp_path)
    _write_state(ctx, brain_state={"pillar": _pillar(5, 7)})
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    ctx.log_path.write_text(
        "10:00:00.000 INFO juli EVAL AAPL:HOLD(0.600,L)[trading_policy:enter]\n"
    )
    r = pillar_balance(ctx)
    assert r.evidence["long_conviction"] == 0.6
    assert r.status == OK


def test_pillar_joint_is_inside_man() -> None:
    """Registered under the inside_man joint (Inside Man confirms the pillar)."""
    from hanoon_prime.inspection.joints import SPECS

    spec = next(s for s in SPECS if s.name == "pillar_balance")
    assert spec.joint == "inside_man"
    assert spec.report is True
