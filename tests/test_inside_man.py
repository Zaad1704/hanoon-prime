"""tests/test_inside_man — direct unit tests for the Inside Man checks.

Locks the runtime guards that back the Sep-11 fixes. Previously these were
only exercised via the full-manifest mock in test_inspection_joints.py; an
OCD guardian pins each branch (OK / FAIL / WARN / edge) in isolation so a
refactor can't silently widen a blind spot.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from hanoon_prime.inspection.checks import FAIL, OK, WARN, CheckResult
from hanoon_prime.inspection.ctx import InspectionContext
from hanoon_prime.inspection.inside_man import (
    brain_halim_bounded,
    conf_bin_loss_streak,
    exits_ib_pnl_fed,
    veto_conviction_integrity,
)
from hanoon_prime.memory import Journal


def _ctx(tmp_path: Path) -> InspectionContext:
    return InspectionContext(base_dir=tmp_path)


def _write_log(ctx: InspectionContext, *lines: str) -> None:
    """Write raw log lines to the ctx's hanoon_prime.log (fresh memo)."""
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    ctx.log_path.write_text("".join(lines))


def _eval_line(token: str) -> str:
    """One EVAL log line carrying a single verdict token."""
    ts = datetime.datetime.now().strftime("%H:%M:%S") + ".000"
    return f"{ts} INFO ib_cycle EVAL {token}\n"


def _eval_line(token: str) -> str:
    """One EVAL log line carrying a single verdict token."""
    ts = datetime.datetime.now().strftime("%H:%M:%S") + ".000"
    return f"{ts} INFO ib_cycle EVAL {token}\n"


# ── brain_halim_bounded ──────────────────────────────────────────────


def test_brain_halim_bounded_ok_within_band(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {"brain_state": {"halim_modifier": 1.0}}
    r = brain_halim_bounded(ctx)
    assert r.status == OK
    assert r.evidence["halim_modifier"] == 1.0


@pytest.mark.parametrize("mod", [0.0, 0.5, 1.5, 0.7, 1.25])
def test_brain_halim_bounded_ok_contract_values(tmp_path: Path, mod: float) -> None:
    """0.0 (cold) and the full [0.5, 1.5] warm band are valid."""
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {"brain_state": {"halim_modifier": mod}}
    assert brain_halim_bounded(ctx).status == OK


@pytest.mark.parametrize("mod", [0.03, 0.05, 0.49, 1.51, 2.0, -0.5])
def test_brain_halim_bounded_fails_outside_contract(tmp_path: Path, mod: float) -> None:
    """Anything outside {0.0} ∪ [0.5, 1.5] is a corrupted HALIM modifier."""
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {"brain_state": {"halim_modifier": mod}}
    r = brain_halim_bounded(ctx)
    assert r.status == FAIL


def test_brain_halim_bounded_cold_is_ok_not_fail(tmp_path: Path) -> None:
    """Cold HALIM (0.0) must NOT hard-fail — that was the prod false-halt."""
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {"brain_state": {"halim_modifier": 0.0}}
    r = brain_halim_bounded(ctx)
    assert r.status == OK
    assert "cold" in r.detail.lower()


def test_brain_halim_bounded_fails_non_numeric(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {"brain_state": {"halim_modifier": "NaN"}}
    r = brain_halim_bounded(ctx)
    assert r.status == FAIL


def test_brain_halim_bounded_warns_when_brain_state_not_dict(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {"brain_state": "not-a-dict"}
    assert brain_halim_bounded(ctx).status == WARN


# ── exits_ib_pnl_fed ─────────────────────────────────────────────────


def _close(ticker: str, reason: str) -> dict:
    return {"event": "position_closed", "ticker": ticker, "reason": reason}


def test_exits_ib_pnl_fed_ok_when_no_closed_trades(tmp_path: Path) -> None:
    """No closed trades yet → warming up, not a failure."""
    ctx = _ctx(tmp_path)
    r = exits_ib_pnl_fed(ctx)
    assert r.status == OK
    assert "warming up" in r.detail


def test_exits_ib_pnl_fed_ok_when_profit_lock_fired(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    j = Journal(ctx.journal_path)
    j.append(_close("AAPL", "profit_lock peak=2.1%"))
    r = exits_ib_pnl_fed(ctx)
    assert r.status == OK
    assert r.evidence["pnl_exits"] == 1


def test_exits_ib_pnl_fed_ok_when_giveback_fired(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    j = Journal(ctx.journal_path)
    j.append(_close("TSLA", "giveback 12%"))
    r = exits_ib_pnl_fed(ctx)
    assert r.status == OK
    assert r.evidence["pnl_exits"] == 1


def test_exits_ib_pnl_fed_warns_when_no_pnl_exits(tmp_path: Path) -> None:
    """Closed trades exist but no profit_lock/giveback → ib_pnl regressed to 0."""
    ctx = _ctx(tmp_path)
    j = Journal(ctx.journal_path)
    j.append(_close("AAPL", "hard_stop"))
    r = exits_ib_pnl_fed(ctx)
    assert r.status == WARN
    assert "ib_pnl may not be flowing" in r.detail


# ── conf_bin_loss_streak ─────────────────────────────────────────────


def test_conf_bin_loss_streak_ok_empty(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {"realized": {"conf_wins": {}, "conf_losses": {}}}
    r = conf_bin_loss_streak(ctx)
    assert r.status == OK
    assert r.evidence["conf_bins"] == 0


def test_conf_bin_loss_streak_ok_when_bin_has_wins(tmp_path: Path) -> None:
    """A bin with losses BUT >=1 win is NOT a loss streak."""
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {
        "realized": {"conf_wins": {"2": 1}, "conf_losses": {"2": 10}}
    }
    assert conf_bin_loss_streak(ctx).status == OK


def test_conf_bin_loss_streak_ok_when_bin_below_threshold(tmp_path: Path) -> None:
    """9 losses (below CONF_LOSS_STREAK_WARN=10) is not yet a streak."""
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {"realized": {"conf_wins": {}, "conf_losses": {"2": 9}}}
    assert conf_bin_loss_streak(ctx).status == OK


def test_conf_bin_loss_streak_warns_on_losing_bin(tmp_path: Path) -> None:
    """0 wins + >=10 losses is an active money-burner — aggressive learning."""
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {
        "realized": {"conf_wins": {}, "conf_losses": {"2": 10}}
    }
    r = conf_bin_loss_streak(ctx)
    assert r.status == WARN
    assert r.evidence["losing_bins"] == [{"bin": "2", "losses": 10, "wins": 0}]


def test_conf_bin_loss_streak_warns_when_realized_missing(tmp_path: Path) -> None:
    """A non-dict `realized` (e.g. None / stale) is structurally suspect."""
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = {"brain_state": {}, "realized": None}
    r = conf_bin_loss_streak(ctx)
    assert r.status == WARN
    assert "realized stats missing" in r.detail


def test_conf_bin_loss_streak_sees_persisted_realized_end_to_end(
    tmp_path: Path,
) -> None:
    """The consolidator surfaces realized conf-bins into state.json so the
    Inside Man check can observe them live.

    Regression lock for the wiring fix: previously `_persist_data` omitted
    `realized`, so `conf_bin_loss_streak` could never see a losing bin and
    silently reported OK. This drives the real RealizedStats → _persist_data
    → JSON round-trip → check path end-to-end.
    """
    from hanoon_prime.brain.consolidation import ConsolidationEngine
    from hanoon_prime.brain.realized_ev import RealizedStats
    from hanoon_prime.brain.shared_state import BrainState

    rs = RealizedStats(persist=False)
    for _ in range(10):
        rs.add_confidence_outcome(0.62, won=False)  # bin 2, 0 wins
    ce = ConsolidationEngine(brain_state=BrainState(), realized=rs)
    data = ce._persist_data()
    assert "realized" in data

    # Simulate the file round-trip: state.json stores string keys on disk.
    on_disk = json.loads(json.dumps(data))
    ctx = _ctx(tmp_path)
    ctx.memo["runtime_state"] = on_disk
    r = conf_bin_loss_streak(ctx)
    assert r.status == WARN
    assert r.evidence["losing_bins"] == [{"bin": "2", "losses": 10, "wins": 0}]


# ── veto_conviction_integrity ────────────────────────────────────────


def test_veto_integrity_ok_when_no_eval_yet(tmp_path: Path) -> None:
    """No EVAL lines yet → warming up, not a regression."""
    ctx = _ctx(tmp_path)
    r = veto_conviction_integrity(ctx)
    assert r.status == OK
    assert r.evidence["scanned_eval_lines"] == 0


def test_veto_integrity_ok_when_direction_veto_has_conviction(tmp_path: Path) -> None:
    """A real -0.6 SELL signal rejected by policy is a legitimate veto."""
    ctx = _ctx(tmp_path)
    _write_log(
        ctx, _eval_line("AAPL:VETOED(-0.600,S)[trading_policy:direction_rejected]")
    )
    assert veto_conviction_integrity(ctx).status == OK


def test_veto_integrity_ok_when_low_score_veto_is_not_direction(tmp_path: Path) -> None:
    """Low-score veto for a non-direction reason (e.g. no_data) is fine."""
    ctx = _ctx(tmp_path)
    _write_log(ctx, _eval_line("AAPL:VETOED(0.000,-)[validity:no_data]"))
    r = veto_conviction_integrity(ctx)
    assert r.status == OK


def test_veto_integrity_fails_on_zero_score_direction_veto(tmp_path: Path) -> None:
    """The exact regression: |score|=0 but labelled direction_rejected."""
    ctx = _ctx(tmp_path)
    _write_log(
        ctx, _eval_line("AAPL:VETOED(0.000,-)[trading_policy:direction_rejected]")
    )
    r = veto_conviction_integrity(ctx)
    assert r.status == FAIL
    assert "zero-conviction" in r.detail
    assert r.evidence["bogus_vetoes"] == 1


def test_veto_integrity_fails_on_near_zero_score(tmp_path: Path) -> None:
    """0.015 < 0.02 floor is still the regression signature."""
    ctx = _ctx(tmp_path)
    _write_log(
        ctx, _eval_line("AAPL:VETOED(0.015,L)[trading_policy:direction_rejected]")
    )
    assert veto_conviction_integrity(ctx).status == FAIL


def test_veto_integrity_ignores_non_veto_verdicts(tmp_path: Path) -> None:
    """BUY/HOLD/SELL verdicts never match the veto regex."""
    ctx = _ctx(tmp_path)
    _write_log(
        ctx,
        _eval_line("AAPL:HOLD(0.600,L)[trading_policy:enter]"),
        _eval_line("TSLA:SELL(-0.700,S)[trading_policy:exit]"),
    )
    assert veto_conviction_integrity(ctx).status == OK


def test_veto_integrity_fails_with_multiple_bogus(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _write_log(
        ctx,
        _eval_line("AAPL:VETOED(0.000,-)[trading_policy:direction_rejected]"),
        _eval_line("TSLA:VETOED(0.010,-)[trading_policy:direction_rejected]"),
        _eval_line("NVDA:HOLD(0.600,L)[trading_policy:enter]"),
    )
    r = veto_conviction_integrity(ctx)
    assert r.status == FAIL
    assert r.evidence["bogus_vetoes"] == 2
