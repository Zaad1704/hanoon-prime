"""Unit tests for the check model + runner harness."""
import pytest

from hanoon_prime.inspection.checks import (
    FAIL,
    MANIFEST_STATUS,
    OK,
    UNVERIFIABLE,
    WARN,
    CheckResult,
    CheckSpec,
    run_check,
)


class _FakeCtx:
    def __init__(self, raise_it: bool = False) -> None:
        self.raise_it = raise_it

    def data(self) -> str:
        if self.raise_it:
            raise RuntimeError("instrument down")
        return "x"


def test_ok_result_fields() -> None:
    r = CheckResult(joint="pipeline", name="heartbeat_fresh", status=OK, detail="fresh")
    assert r.joint == "pipeline"
    assert r.name == "heartbeat_fresh"
    assert r.status == OK
    assert r.evidence == {}


def test_spec_plus_runner_ok() -> None:
    def chk(ctx: _FakeCtx) -> CheckResult:
        return CheckResult(
            "pipeline", "heartbeat_fresh", OK, evidence={"d": ctx.data()}
        )

    spec = CheckSpec("pipeline", "heartbeat_fresh", chk, hard=True)
    r = run_check(spec, _FakeCtx())  # type: ignore[arg-type]
    assert r.status == OK
    assert r.evidence == {"d": "x"}


def test_runner_converts_instrument_failure_to_unverifiable() -> None:
    def chk(ctx: _FakeCtx) -> CheckResult:
        ctx.data()  # raise
        return CheckResult("pipeline", "heartbeat_fresh", OK)

    spec = CheckSpec("pipeline", "heartbeat_fresh", chk)
    r = run_check(spec, _FakeCtx(raise_it=True))  # type: ignore[arg-type]
    assert r.status == UNVERIFIABLE
    assert "instrument down" in r.detail
    assert "error" in r.evidence


def test_manifest_status_aggregation() -> None:
    ok = CheckResult("a", "a1", OK)
    warn = CheckResult("a", "a2", WARN)
    fail_hard = CheckResult("a", "a3", FAIL)
    unv = CheckResult("a", "a4", UNVERIFIABLE)
    hard = {("a", "a3")}
    assert MANIFEST_STATUS([ok], set()) == OK
    assert MANIFEST_STATUS([ok, warn], set()) == WARN
    assert MANIFEST_STATUS([ok, unv], set()) == WARN
    assert MANIFEST_STATUS([ok, warn, fail_hard], hard) == FAIL
    assert MANIFEST_STATUS([ok, fail_hard], hard) == FAIL
