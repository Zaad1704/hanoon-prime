"""safety + purity joint checks."""
from hanoon_prime.brain.config import DEFAULT_WEIGHTS
from hanoon_prime.inspection import purity, safety, weight_purity
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


def test_error_burst_quiet_ok(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    _log(ctx, ["12:00:00.000 INFO ib_cycle HEARTBEAT open=0\n"])
    assert safety.no_error_burst(ctx).status == OK


def test_error_burst_warns_then_fails(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    _log(ctx, ["12:00:00.000 ERROR ib_insync.ib cancelMktData: no reqId\n"] * 5)
    assert safety.no_error_burst(ctx).status == WARN
    _log(ctx, ["12:00:00.000 ERROR ib_insync.ib cancelMktData: no reqId\n"] * 25)
    assert safety.no_error_burst(InspectionContext(base_dir=tmp_path)).status == FAIL


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


def _regime_counts(active: int, inactive: int) -> dict[str, int]:
    return {"range": active, "trend_up": inactive}


def test_regime_weights_healthy_ok(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    healthy = {k: max(-0.1, min(0.1, v)) for k, v in DEFAULT_WEIGHTS.items()}
    ctx.memo["regime_weights"] = {
        "vectors": {"range": healthy},
        "counts": _regime_counts(active=40, inactive=3),
    }
    assert weight_purity.regime_weights_bounded(ctx).status == OK


def test_regime_weights_drift_fails(tmp_path) -> None:
    """The live incident vector (abs_sum ≈ 4.16, weights at -0.55) must FAIL."""
    ctx = InspectionContext(base_dir=tmp_path)
    drifted = dict(DEFAULT_WEIGHTS)
    drifted.update(
        {"adx": -0.55, "vw_macd_hist": -0.45, "volume_profile_proximity": -0.3}
    )
    ctx.memo["regime_weights"] = {
        "vectors": {"range": drifted},
        "counts": _regime_counts(active=40, inactive=3),
    }
    assert weight_purity.regime_weights_bounded(ctx).status == FAIL


def test_cortex_score_degenerate_ok(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["juli_state"] = {"weights": dict(DEFAULT_WEIGHTS)}
    ctx.memo["regime_weights"] = {"vectors": {}, "counts": {}}
    assert weight_purity.cortex_score_degenerate(ctx).status == OK


def test_cortex_score_collapse_fails(tmp_path) -> None:
    """A budget pinned at ~0 forces the score to 0.0 → the brain is blind."""
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["juli_state"] = {"weights": {k: 0.0 for k in DEFAULT_WEIGHTS}}
    ctx.memo["regime_weights"] = {"vectors": {}, "counts": {}}
    assert weight_purity.cortex_score_degenerate(ctx).status == FAIL


def test_cortex_negative_polarity_warns(tmp_path) -> None:
    """Non-positive signed sum flags the exact corruption the guard used to hit."""
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["juli_state"] = {"weights": {k: -v for k, v in DEFAULT_WEIGHTS.items()}}
    ctx.memo["regime_weights"] = {"vectors": {}, "counts": {}}
    result = weight_purity.cortex_score_degenerate(ctx)
    assert result.status == WARN
    assert "non-positive signed" in result.detail
