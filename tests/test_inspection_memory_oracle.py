"""memory / execution-oracle / halim joint checks."""
import json

from hanoon_prime.inspection import halim, journals, oracle
from hanoon_prime.inspection.checks import FAIL, OK, WARN
from hanoon_prime.inspection.ctx import InspectionContext


def _mk_journal(tmp_path, rows: int = 5) -> None:
    from hanoon_prime.memory import Journal

    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    j = Journal(jp)
    for _ in range(rows):
        j.append({"event": "verdict", "action": "HOLD", "score": 0.5, "ticker": "NVDA"})


def test_journal_grows(tmp_path) -> None:
    _mk_journal(tmp_path, 5)
    ctx = InspectionContext(base_dir=tmp_path, prev_journal_count=4)
    assert journals.journal_grows(ctx).status == OK
    ctx2 = InspectionContext(base_dir=tmp_path, prev_journal_count=9)
    assert journals.journal_grows(ctx2).status == FAIL


def test_journal_grows_first_run_ok(tmp_path) -> None:
    _mk_journal(tmp_path, 5)
    ctx = InspectionContext(base_dir=tmp_path)
    assert journals.journal_grows(ctx).status == OK


def test_verdicts_valid_ok(tmp_path) -> None:
    _mk_journal(tmp_path)
    ctx = InspectionContext(base_dir=tmp_path)
    assert journals.verdicts_valid(ctx).status == OK


def test_verdicts_valid_bad_action(tmp_path) -> None:
    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    jp.write_text(
        json.dumps(
            {
                "seq": 0,
                "event": "verdict",
                "action": "FOO",
                "score": 0.5,
                "ticker": "NVDA",
            }
        )
        + "\n"
    )
    ctx = InspectionContext(base_dir=tmp_path)
    assert journals.verdicts_valid(ctx).status == FAIL


def test_verdicts_valid_nan_score(tmp_path) -> None:
    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    jp.write_text(
        json.dumps(
            {
                "seq": 0,
                "event": "verdict",
                "action": "HOLD",
                "score": float("nan"),
                "ticker": "NVDA",
            }
        )
        + "\n"
    )
    ctx = InspectionContext(base_dir=tmp_path)
    assert journals.verdicts_valid(ctx).status == FAIL


def test_seq_forward_gap_is_warn(tmp_path) -> None:
    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    jp.write_text(
        json.dumps(
            {
                "seq": 0,
                "event": "verdict",
                "action": "HOLD",
                "score": 0.5,
                "ticker": "NVDA",
            }
        )
        + "\n"
        + json.dumps(
            {
                "seq": 5,
                "event": "verdict",
                "action": "HOLD",
                "score": 0.5,
                "ticker": "NVDA",
            }
        )
        + "\n"
    )
    ctx = InspectionContext(base_dir=tmp_path)
    assert journals.seq_forward(ctx).status == WARN


def test_chain_intact_from_anchor(tmp_path) -> None:
    _mk_journal(tmp_path, 5)
    ctx = InspectionContext(base_dir=tmp_path)
    assert journals.chain_intact_from_anchor(ctx).status == OK


def test_oracle_enters_minted(tmp_path) -> None:
    from hanoon_prime.memory import Journal

    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    j = Journal(jp)
    j.append(
        {
            "event": "verdict",
            "action": "ENTER",
            "score": 0.6,
            "ticker": "ACCL",
            "direction": 1,
        }
    )
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["health"] = {"positions": [], "position_count": 0}
    assert oracle.enters_minted(ctx).status == WARN  # unminted ENTER
    j.append(
        {
            "event": "verdict",
            "action": "ENTER",
            "score": 0.6,
            "ticker": "NVDA",
            "direction": 1,
        }
    )
    j.append(
        {
            "event": "position_closed",
            "ticker": "ACCL",
            "pnl": 0.0,
            "entry_price": 1.0,
            "shares": 1.0,
        }
    )
    ctx.memo["health"] = {"positions": ["NVDA"], "position_count": 1}
    assert oracle.enters_minted(ctx).status == OK


def test_oracle_closes_reconciled(tmp_path) -> None:
    from hanoon_prime.memory import Journal

    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    j = Journal(jp)
    j.append(
        {
            "event": "position_closed",
            "ticker": "NVDA",
            "pnl": 0.0,
            "entry_price": 10.0,
            "shares": 1.0,
        }
    )
    ctx = InspectionContext(base_dir=tmp_path)
    assert oracle.closes_reconciled(ctx).status == OK
    j.append({"event": "position_closed", "ticker": "NVDA"})
    assert oracle.closes_reconciled(ctx).status == WARN


def test_oracle_equity_synced(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["runtime_state"] = {
        "brain_state": {"policy_state": {"equity_synced": False, "equity": 0.0}}
    }
    assert oracle.equity_synced(ctx).status == WARN
    ctx.memo["runtime_state"] = {
        "brain_state": {"policy_state": {"equity_synced": True, "equity": 100.0}}
    }
    assert oracle.equity_synced(ctx).status == OK


def test_halim_state_matches_clock(tmp_path, monkeypatch) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    monkeypatch.setattr(halim, "halim_probe", lambda c: "ok")
    assert halim.halim_state_matches_clock(ctx).status == OK
    monkeypatch.setattr(halim, "halim_probe", lambda c: "asleep")
    assert halim.halim_state_matches_clock(ctx).status == OK
    monkeypatch.setattr(halim, "halim_probe", lambda c: "down")
    assert halim.halim_state_matches_clock(ctx).status == FAIL
