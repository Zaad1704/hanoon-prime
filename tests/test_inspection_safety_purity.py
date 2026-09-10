"""safety + purity joint checks."""
from hanoon_prime.inspection import purity, safety
from hanoon_prime.inspection.checks import FAIL, OK, WARN
from hanoon_prime.inspection.ctx import InspectionContext


def _log(ctx: InspectionContext, lines: list[str]) -> None:
    ctx.log_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.log_path.write_text("".join(lines))


def test_guard_trigger_fails(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    _log(ctx, ["12:00:00.000 INFO ib_cycle NETTING GUARD: blocked reversal\n"])
    assert safety.no_netting_guard(ctx).status == FAIL
    _log(ctx, ["12:00:00.000 INFO ib_cycle HEARTBEAT open=0\n"])
    assert safety.no_netting_guard(InspectionContext(base_dir=tmp_path)).status == OK


def test_traceback_fails(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    _log(ctx, ["12:00:00.000 ERROR t Traceback (most recent call last):\n"])
    assert safety.no_traceback(ctx).status == FAIL


def test_safety_halt_fails(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    _log(ctx, ["12:00:00.000 WARNING policy SAFETY HALT: daily_loss_limit\n"])
    assert safety.no_safety_halt(ctx).status == FAIL


def test_learn_blocked_flags(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    _log(ctx, ["12:00:00.000 WARNING juli LEARN BLOCKED: closes pending\n"])
    assert safety.no_learn_blocked(ctx).status == FAIL


def test_policy_flags_deferred_when_inactive(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["health"] = {"session_active": False}
    ctx.memo["runtime_state"] = {"brain_state": {"policy_state": {"enabled": False}}}
    assert safety.policy_flags(ctx).status == OK


def test_policy_flags_warn_disabled_while_active(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["health"] = {"session_active": True}
    ctx.memo["runtime_state"] = {
        "brain_state": {"policy_state": {"enabled": False, "authorized": True}}
    }
    assert safety.policy_flags(ctx).status == WARN


def test_drawdown_ok_when_equity_zero(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["runtime_state"] = {
        "brain_state": {"policy_state": {"equity": 0.0, "daily_pnl": -100.0}}
    }
    assert safety.drawdown_bound(ctx).status == OK  # rule re-arms after equity sync


def test_drawdown_within_and_break(tmp_path) -> None:
    base = {"brain_state": {"policy_state": {"equity": 10000.0, "daily_pnl": -80.0}}}
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["runtime_state"] = base
    assert safety.drawdown_bound(ctx).status == OK  # -0.8%
    ctx.memo["runtime_state"] = {
        "brain_state": {"policy_state": {"equity": 10000.0, "daily_pnl": -200.0}}
    }
    assert safety.drawdown_bound(ctx).status == FAIL  # -2%


def test_no_test_episodes(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["juli_state"] = {
        "episodes": [{"ticker": "NVDA", "vector": []}, {"ticker": "TEST", "vector": []}]
    }
    assert purity.no_test_episodes(ctx).status == FAIL
    ctx.memo["juli_state"] = {"episodes": [{"ticker": "NVDA", "vector": []}]}
    assert purity.no_test_episodes(ctx).status == OK


def test_weights_bounds(tmp_path) -> None:
    ok_weights = {f"w{i}": 0.1 for i in range(20)}
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["juli_state"] = {"weights": ok_weights}
    assert purity.weights_finite_in_band(ctx).status == OK
    ctx.memo["juli_state"] = {
        "weights": {"w0": 5.0, **{f"w{i}": 0.1 for i in range(1, 20)}}
    }
    assert purity.weights_finite_in_band(ctx).status == FAIL  # out of [-2,2]
    ctx.memo["juli_state"] = {
        "weights": {"w0": float("nan"), **{f"w{i}": 0.1 for i in range(1, 20)}}
    }
    assert purity.weights_finite_in_band(ctx).status == FAIL  # NaN
    ctx.memo["juli_state"] = {"weights": {"w0": 0.1}}
    assert purity.weights_finite_in_band(ctx).status == FAIL  # sparse


def test_brain_fields_bounded(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    good = {
        "brain_state": {
            "threshold": 0.58,
            "pred_error": 0.4,
            "risk_ceiling": 1.0,
            "positions_open": 2,
        }
    }
    ctx.memo["runtime_state"] = good
    assert purity.brain_fields_bounded(ctx).status == OK
    ctx.memo["runtime_state"] = {
        "brain_state": {
            "threshold": 0.9,
            "pred_error": 1.5,
            "risk_ceiling": -1.0,
            "positions_open": -3,
        }
    }
    assert purity.brain_fields_bounded(ctx).status == FAIL
