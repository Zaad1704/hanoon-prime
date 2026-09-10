"""joint registry, manifest aggregation, CLI smoke."""
from hanoon_prime.inspection import joints
from hanoon_prime.inspection.checks import FAIL, OK
from hanoon_prime.inspection.ctx import InspectionContext


def test_spec_registry_no_duplicates() -> None:
    names = [(s.joint, s.name) for s in joints.SPECS]
    assert len(names) == len(set(names))
    assert len(joints.HARD_KEYS) == sum(1 for s in joints.SPECS if s.hard)
    assert len(joints.SPECS) >= 30


def test_full_manifest_empty_stack(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path, telemetry_url="http://127.0.0.1:1")
    m = joints.run_all(ctx)
    assert m.pid > 0
    assert len(m.results) == len(joints.SPECS)
    assert m.status == FAIL  # health down / process down => hard fail
    assert any(
        r.joint == "telemetry" and r.name == "health_ok" and r.status == FAIL
        for r in m.results
    )


def test_full_manifest_healthy_mock_stack(tmp_path, monkeypatch) -> None:
    import datetime as dt

    from hanoon_prime.inspection import halim as halim_mod
    from hanoon_prime.inspection import runtime
    from hanoon_prime.inspection import system as system_mod
    from hanoon_prime.memory import Journal

    ctx = InspectionContext(
        base_dir=tmp_path, prev_journal_count=1, telemetry_url="http://127.0.0.1:1"
    )
    ctx.memo["health"] = {
        "status": "ok",
        "connected": True,
        "session": "post_market",
        "session_active": False,
        "position_count": 0,
        "positions": [],
    }
    ctx.memo["snapshot"] = {"health": {}}
    ctx.memo["positions"] = {"positions": [], "total_pnl": 0.0, "count": 0}
    ctx.memo["account"] = {"account_summary": {}, "positions_open": 0}
    ctx.memo["runtime_state"] = {
        "brain_state": {
            "positions_open": 0,
            "threshold": 0.58,
            "pred_error": 0.4,
            "risk_ceiling": 1.0,
            "policy_state": {
                "equity": 100.0,
                "equity_synced": True,
                "enabled": False,
                "authorized": True,
                "daily_pnl": 0.0,
            },
        }
    }
    ctx.memo["juli_state"] = {
        "weights": {f"w{i}": 0.1 for i in range(20)},
        "episodes": [{"ticker": "NVDA", "vector": []}],
    }
    ctx.memo["regime_weights"] = {"vectors": {}, "counts": {}}
    ctx.memo["ledger"] = {"notify": {"last_ok": 1e20}}
    j = Journal(ctx.journal_path)
    j.append({"event": "verdict", "action": "HOLD", "score": 0.5, "ticker": "NVDA"})
    j.append({"event": "verdict", "action": "HOLD", "score": 0.5, "ticker": "NVDA"})
    now = dt.datetime.now().strftime("%H:%M:%S")
    log = ctx.log_path
    log.parent.mkdir(parents=True)
    log.write_text(
        f"{now}.000 INFO ib_cycle SESSION SLEEP: post_market inactive — whole system idle\n"
        f"{now}.000 INFO ib_cycle HEARTBEAT open=0 journal=10\n"
    )
    monkeypatch.setattr(system_mod, "_pidread", lambda c, s: (1234, True, ""))
    monkeypatch.setattr(system_mod, "_pgrep", lambda p: [1234])
    monkeypatch.setattr(
        system_mod, "_cmdline", lambda pid: "/x/.venv/bin/python -m hanoon_prime.cli"
    )
    monkeypatch.setattr(halim_mod, "halim_probe", lambda c: "asleep")
    monkeypatch.setattr(runtime, "_get_token", lambda: "t")
    monkeypatch.setattr(runtime, "_get_chat_id", lambda: "c")
    m = joints.run_all(ctx)
    bad = [
        f"{r.joint}.{r.name}:{r.status}:{r.detail}" for r in m.results if r.status != OK
    ]
    assert m.status == OK, bad


def test_manifest_hard_and_anomaly_partition(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    m = joints.run_all(ctx)
    assert all((r.joint, r.name) in joints.HARD_KEYS for r in m.hard_fails)
    assert all(
        (r.joint, r.name) in {(s.joint, s.name) for s in joints.SPECS if s.report}
        for r in m.anomalies
    )


def test_live_failure_checks_are_hard(tmp_path) -> None:
    keys = {(s.joint, s.name) for s in joints.SPECS if s.hard}
    assert ("telemetry", "positions_marked_live") in keys
    assert ("safety", "no_error_burst") in keys


def test_cli_manifest_json(tmp_path, monkeypatch, capsys) -> None:
    from hanoon_prime.inspection import __main__ as cli

    monkeypatch.setattr(cli, "_ctx", lambda: InspectionContext(base_dir=tmp_path))
    rc = cli.main(["manifest", "--json"])
    out = capsys.readouterr().out
    assert '"status"' in out
    assert rc in (0, 2, 4)
