"""Tests for InspectionContext + live-surface probes."""

import json
import re
from pathlib import Path

from hanoon_prime.inspection import probe
from hanoon_prime.inspection.ctx import InspectionContext


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_pid_file_and_properties(tmp_path: Path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    assert ctx.pid_file("bot").name == "bot.pid"
    assert ctx.pid_file("bot").parent.name == "pids"


def test_git_head_none_on_non_repo(tmp_path: Path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    assert ctx.git_head() is None


def test_line_ts_local_clock() -> None:
    from datetime import datetime

    ts = probe.line_ts("12:34:56.789 INFO ib_cycle HEARTBEAT open=0")
    assert ts is not None
    assert datetime.fromtimestamp(ts).hour == 12


def test_health_unreachable_flagged(tmp_path: Path) -> None:
    ctx = InspectionContext(base_dir=tmp_path, telemetry_url="http://127.0.0.1:1")
    h = probe.health(ctx)
    assert "_unreachable" in h


def test_session_lines_bounded_by_last_start_marker(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "hanoon_prime.log"
    log.parent.mkdir(parents=True)
    lines = [
        "10:00:00.000 INFO ib_adapter Starting (seed=OLD seed)\n",
        "10:00:01.000 INFO ib_cycle CYCLE bars=1\n",
        "11:00:00.000 INFO ib_adapter Starting (seed=CURRENT)\n",
        "11:00:01.000 INFO ib_cycle HEARTBEAT open=0\n",
        "11:00:02.000 INFO ib_cycle HEARTBEAT open=1\n",
    ]
    log.write_text("".join(lines))
    ctx = InspectionContext(base_dir=tmp_path)
    got = probe.session_lines(ctx)
    assert len(got) == 2
    assert all("11:00" in ln for ln in got)


def test_journal_chain_state_intact(tmp_path: Path) -> None:
    from hanoon_prime.memory import Journal

    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    j = Journal(jp)
    j.append({"event": "verdict", "action": "HOLD", "score": 0.5, "ticker": "X"})
    j.append({"event": "verdict", "action": "HOLD", "score": 0.5, "ticker": "X"})
    ctx = InspectionContext(base_dir=tmp_path)
    cs = probe.journal_chain_state(ctx)
    assert cs["breaks"] == 0
    assert cs["anchor_seq"] == 0
    assert cs["gaps_after_anchor"] == 0


def test_journal_chain_state_break_sets_anchor_after_it(tmp_path: Path) -> None:
    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    jp.write_text(
        '{"seq":0,"prev_hash":null,"event":"x","hash":"AAAA"}\n'
        '{"seq":1,"prev_hash":"BBBB","event":"x","hash":"CCCC"}\n'
    )
    ctx = InspectionContext(base_dir=tmp_path)
    cs = probe.journal_chain_state(ctx)
    assert cs["breaks"] == 1
    assert cs["last_break_seq"] == 1
    assert cs["anchor_seq"] is None  # nothing after the anchor
    assert cs["gaps_after_anchor"] == 0


def test_halim_probe_down_on_bad_url(tmp_path: Path) -> None:
    ctx = InspectionContext(base_dir=tmp_path, halim_url="http://127.0.0.1:1")
    assert probe.halim_probe(ctx) == "down"


def test_last_line_age_present(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "hanoon_prime.log"
    log.parent.mkdir(parents=True)
    log.write_text("12:00:00.000 INFO ib_cycle HEARTBEAT open=0\n")
    ctx = InspectionContext(base_dir=tmp_path)
    age = probe.last_line_age(ctx, probe.HEARTBEAT_MARKER)
    assert age is not None
    assert 0 <= age < 86400
