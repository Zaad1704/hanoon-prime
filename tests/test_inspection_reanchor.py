"""journal chain re-anchor boot step."""
import json

from hanoon_prime.inspection import reanchor
from hanoon_prime.inspection.ctx import InspectionContext
from hanoon_prime.memory import Journal


def _good_ctx(tmp_path, n: int = 3) -> InspectionContext:
    ctx = InspectionContext(base_dir=tmp_path)
    j = Journal(ctx.journal_path)
    for _ in range(n):
        j.append({"event": "verdict", "action": "HOLD", "score": 0.5, "ticker": "NVDA"})
    return ctx


def test_intact_chain_noop(tmp_path) -> None:
    ctx = _good_ctx(tmp_path)
    changed, out = reanchor.reanchor_if_broken(ctx)
    assert changed is False
    assert "intact" in out


def test_broken_chain_gets_reseed(tmp_path) -> None:
    ctx = _good_ctx(tmp_path)
    jp = ctx.journal_path
    lines = jp.read_bytes().splitlines()
    row = json.loads(lines[1])
    row["action"] = "VETOED"  # corrupt a middle hash
    lines[1] = json.dumps(row).encode()
    jp.write_bytes(b"\n".join(lines) + b"\n")
    changed, out = reanchor.reanchor_if_broken(ctx)
    assert changed is True
    assert "re-anchored" in out
    tail = Journal(ctx.journal_path).tail(5)
    assert tail[-1]["event"] == "chain_reseed"


def test_empty_journal_intact(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    changed, out = reanchor.reanchor_if_broken(ctx)
    assert changed is False
    assert "intact" in out


def test_cli_reanchor(tmp_path, monkeypatch, capsys) -> None:
    from hanoon_prime.inspection import __main__ as cli

    monkeypatch.setattr(cli, "_ctx", lambda: InspectionContext(base_dir=tmp_path))
    rc = cli.main(["reanchor"])
    assert rc == 0
    assert "intact" in capsys.readouterr().out
