"""digest emit + gated heal."""

from hanoon_prime.inspection import digest, heal, joints
from hanoon_prime.inspection.checks import FAIL, OK, CheckResult
from hanoon_prime.inspection.ctx import InspectionContext


def _manifest(*results: CheckResult) -> joints.Manifest:
    return joints.Manifest(
        ts=1_700_000_000.0, git_head="deadbeef", pid=1, results=tuple(results)
    )


def _capture(store: list[str]):
    def f(text: str, ctx: InspectionContext) -> bool:
        store.append(text)
        return True

    return f


def test_digest_sends_once_per_day(tmp_path, monkeypatch) -> None:
    sent: list[str] = []
    monkeypatch.setattr(digest, "_notify", _capture(sent))
    ctx = InspectionContext(base_dir=tmp_path)
    m = _manifest(
        CheckResult("telemetry", "health_ok", FAIL, detail="down"),
        CheckResult("halim", "halim_state_matches_clock", FAIL, detail="down"),
    )
    ok, msg = digest.digest_send(ctx, m)
    assert ok and msg == "sent"
    assert len(sent) == 1
    assert "status: FAIL" in sent[0]
    assert "health_ok=FAIL" in sent[0]
    ok2, msg2 = digest.digest_send(ctx, m)
    assert not ok2 and "already" in msg2
    assert len(sent) == 1  # no duplicate day


def test_chunks_fit_limit() -> None:
    text = "x" * 9500
    parts = digest._chunks(text)
    assert all(len(c) <= 4000 for c in parts)
    assert sum(len(c) for c in parts) == len(text)


def test_heal_dry_run_scope(tmp_path, monkeypatch) -> None:
    ctx = InspectionContext(base_dir=tmp_path, heal_enabled=True)
    m = _manifest(
        CheckResult("processes", "halim_alive", FAIL, detail="down"),
        CheckResult("processes", "cloudflared_alive", FAIL, detail="down"),
        CheckResult("processes", "bot_alive", FAIL, detail="down"),
    )
    out = heal.heal(ctx, m, dry_run=True)
    assert any("halim_alive" in o for o in out)
    assert any("cloudflared_alive" in o for o in out)
    assert not any("bot_alive" in o for o in out)
    assert heal._ledger_day_count(ctx) == 0  # dry run records nothing


def test_heal_budget_exhausted(tmp_path, monkeypatch) -> None:
    ctx = InspectionContext(base_dir=tmp_path, heal_enabled=True)
    heal.record_heal(ctx, "test-one")
    heal.record_heal(ctx, "test-two")
    m = _manifest(CheckResult("processes", "halim_alive", FAIL, detail="down"))
    out = heal.heal(ctx, m)
    assert any("budget" in o for o in out)


def test_heal_disabled(tmp_path, monkeypatch) -> None:
    ctx = InspectionContext(base_dir=tmp_path, heal_enabled=False)
    m = _manifest(CheckResult("processes", "halim_alive", FAIL, detail="down"))
    out = heal.heal(ctx, m)
    assert any("disabled" in o for o in out)


def test_heal_runs_and_records(tmp_path, monkeypatch) -> None:
    ctx = InspectionContext(base_dir=tmp_path, heal_enabled=True)
    ran: list[list[str]] = []
    monkeypatch.setattr(heal, "_run", lambda cmd, dry: (ran.append(cmd), "ok")[1])
    m = _manifest(CheckResult("processes", "halim_alive", FAIL, detail="down"))
    out = heal.heal(ctx, m)
    assert any("halim_alive" in o for o in out)
    assert ran
    assert heal._ledger_day_count(ctx) == 1
