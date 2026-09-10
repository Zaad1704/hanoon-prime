"""telemetry / pipeline / session / notify joint checks."""
import datetime

from hanoon_prime.inspection import runtime
from hanoon_prime.inspection.checks import FAIL, OK, WARN
from hanoon_prime.inspection.ctx import InspectionContext


def _now_t() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _healthy_memo(
    ctx: InspectionContext, *, session: str = "post_market", active: bool = False
) -> InspectionContext:
    ctx.memo["health"] = {
        "status": "ok",
        "connected": True,
        "session": session,
        "session_active": active,
        "position_count": 0,
        "positions": [],
    }
    ctx.memo["snapshot"] = {"health": {"status": "ok"}}
    return ctx


def test_health_ok_pass(tmp_path) -> None:
    assert (
        runtime.health_ok(_healthy_memo(InspectionContext(base_dir=tmp_path))).status
        == OK
    )


def test_health_ok_fail_disconnected(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["health"] = {"status": "disconnected", "connected": False}
    assert runtime.health_ok(ctx).status == FAIL


def test_health_ok_unreachable_fails(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["health"] = {"_unreachable": "boom"}
    assert runtime.health_ok(ctx).status == FAIL


def test_snapshot_fresh_ok_and_fail(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["snapshot"] = {"health": {}}
    assert runtime.snapshot_fresh(ctx).status == OK
    ctx.memo["snapshot"] = {}
    assert runtime.snapshot_fresh(ctx).status == FAIL


def test_positions_surface_warn_missing_field(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["health"] = {"status": "ok"}
    assert runtime.positions_surface(ctx).status == WARN


def test_heartbeat_fresh(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    assert runtime.heartbeat_fresh(ctx).status == FAIL  # no heartbeat ever
    log = ctx.log_path
    log.parent.mkdir(parents=True)
    log.write_text(f"{_now_t()}.000 INFO ib_cycle HEARTBEAT open=0 journal=1\n")
    assert runtime.heartbeat_fresh(InspectionContext(base_dir=tmp_path)).status == OK


def test_cycle_only_expected_when_active(tmp_path) -> None:
    ctx = _healthy_memo(InspectionContext(base_dir=tmp_path), active=True)
    assert runtime.cycle_flows_when_active(ctx).status == FAIL  # active but no cycles
    log = ctx.log_path
    log.parent.mkdir(parents=True)
    log.write_text(f"{_now_t()}.000 INFO ib_cycle CYCLE bars=1 open=0 d=0 x=0\n")
    ctx2 = _healthy_memo(InspectionContext(base_dir=tmp_path), active=True)
    assert runtime.cycle_flows_when_active(ctx2).status == OK


def test_sleep_is_expected(tmp_path) -> None:
    ctx = _healthy_memo(InspectionContext(base_dir=tmp_path), active=False)
    assert runtime.sleep_is_expected(ctx).status == WARN  # inactive, no marker
    log = ctx.log_path
    log.parent.mkdir(parents=True)
    log.write_text(
        f"{_now_t()}.000 INFO ib_cycle SESSION SLEEP: post_market inactive — whole system idle\n"
    )
    ctx2 = _healthy_memo(InspectionContext(base_dir=tmp_path), active=False)
    assert runtime.sleep_is_expected(ctx2).status == OK


def test_state_matches_clock_unknown_session_is_warn(tmp_path) -> None:
    ctx = _healthy_memo(InspectionContext(base_dir=tmp_path), session="bogus_trading")
    assert runtime.state_matches_clock(ctx).status == WARN


def test_positions_reconciled_residual_tolerated(tmp_path) -> None:
    ctx = _healthy_memo(InspectionContext(base_dir=tmp_path))
    ctx.memo["health"]["position_count"] = 2
    ctx.memo["runtime_state"] = {"brain_state": {"positions_open": 0}}
    assert runtime.positions_reconciled(ctx).status == OK
    ctx.memo["runtime_state"] = {"brain_state": {"positions_open": 2}}
    assert runtime.positions_reconciled(ctx).status == OK
    ctx.memo["runtime_state"] = {"brain_state": {"positions_open": 3}}
    assert runtime.positions_reconciled(ctx).status == WARN


def test_telegram_configured(tmp_path, monkeypatch) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    monkeypatch.setattr(runtime, "_get_token", lambda: "t")
    monkeypatch.setattr(runtime, "_get_chat_id", lambda: "c")
    assert runtime.telegram_configured(ctx).status == OK
    monkeypatch.setattr(runtime, "_get_token", lambda: "")
    assert runtime.telegram_configured(ctx).status == WARN


def test_send_healthy(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["ledger"] = {"notify": {"last_ok": 1e20}}
    assert runtime.send_healthy(ctx).status == OK
    ctx.memo["ledger"] = {"notify": {"last_ok": 0.0}}
    assert runtime.send_healthy(ctx).status == WARN
