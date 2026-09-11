"""tests/test_side_policy_conviction — the "why not winning?" diagnostic.

side_policy_conviction answers two questions each poll: (1) which policy gate
(veto reason) is vetoing recent verdicts, and (2) for direction_rejected
vetoes, whether the cortex had real conviction behind the rejection. A high
mean |score| on direction_rejected vetoes means real conviction is being
discarded by the side policy (e.g. SHORT disallowed on a cash account).
"""
from __future__ import annotations

import datetime
from pathlib import Path

from hanoon_prime.inspection.ctx import InspectionContext
from hanoon_prime.inspection.side_policy import side_policy_conviction


def _ctx(tmp_path: Path) -> InspectionContext:
    return InspectionContext(base_dir=tmp_path)


def _eval_line(token: str) -> str:
    """One EVAL log line in the live bot format (juli logger)."""
    ts = datetime.datetime.now().strftime("%H:%M:%S") + ".000"
    return f"{ts} INFO juli EVAL {token}\n"


def _write_log(ctx: InspectionContext, *lines: str) -> None:
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    ctx.log_path.write_text("".join(lines))


def test_ok_when_warming_up(tmp_path: Path) -> None:
    r = side_policy_conviction(_ctx(tmp_path))
    assert r.status == "OK"
    assert r.evidence["checked_eval_lines"] == 0


def test_ok_when_only_low_penny_vetoes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _write_log(ctx, _eval_line("AEON:VETOED(0.600,-)[trading_policy:low_penny_score]"))
    r = side_policy_conviction(ctx)
    assert r.status == "OK"
    assert r.evidence["direction_rejected"] == 0


def test_ok_when_zero_conviction_rejected(tmp_path: Path) -> None:
    """|score| < DIRECTION_MIN_SCORE should HOLD, not veto; a 0.0 token means
    no real-conviction 'discarded' signal here (that's veto_conviction's job)."""
    ctx = _ctx(tmp_path)
    _write_log(
        ctx, _eval_line("AAPL:VETOED(0.000,-)[trading_policy:direction_rejected]")
    )
    r = side_policy_conviction(ctx)
    assert r.status == "OK"
    assert r.evidence["mean_abs_score"] == 0.0


def test_warn_on_real_conviction_discarded(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _write_log(
        ctx,
        _eval_line("AAPL:VETOED(0.700,SHORT)[trading_policy:direction_rejected]"),
        _eval_line("TSLA:VETOED(0.650,SHORT)[trading_policy:direction_rejected]"),
    )
    r = side_policy_conviction(ctx)
    assert r.status == "WARN"
    assert r.evidence["direction_rejected"] == 2
    assert r.evidence["mean_abs_score"] >= 0.5


def test_ok_when_buys_admitted(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _write_log(ctx, _eval_line("NVD:BUY(0.850,L)[trading_policy:enter]"))
    r = side_policy_conviction(ctx)
    assert r.status == "OK"
    assert r.evidence["vetoes"] == 0
