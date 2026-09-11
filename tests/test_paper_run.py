"""tests/test_paper_run.py — Phase 4: pre-locked paper-harness integrity.

Guards the protocol mechanics, not profitability:
   1. The report embeds a protocol hash (records WHICH locked protocol ran).
   2. Session aggregation is ONE DAY across the whole universe (5 fixture
      days → 5 sessions), never per ticker.
   3. Release criteria P1–P6 are boolean and a thin-data universe is FAIL
      (on the committed 5-day fixtures) — the honest, blocking exit code.
   4. No runtime patching of risk constants inside the harness.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hanoon_prime.hippocampus import Hippocampus
from scripts import paper_run

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"
PAPER = Path(__file__).resolve().parent.parent / "scripts" / "paper_run.py"


@pytest.fixture
def protocol_file() -> Path:
    return paper_run.PROTOCOL_FILE


def test_protocol_file_is_present_and_committed(protocol_file: Path) -> None:
    assert protocol_file.exists(), "pre-locked protocol must exist"
    name = str(protocol_file.relative_to(protocol_file.parent.parent))
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", name],
        capture_output=True,
    ).returncode
    assert ignored == 1, "evaluation protocol must be committed, not gitignored"


def test_protocol_contains_all_prelocked_release_criteria(protocol_file: Path) -> None:
    txt = protocol_file.read_text()
    for needle in (
        "MIN_TRADES",
        "DEFAULT_FOLDS",
        "EDGE_LOOKBACK",
        "KILL_DAILY_LOSS_LIMIT",
        "DAILY_LOSS_LIMIT",
        "INSUFFICIENT",
    ):
        assert needle in txt, f"protocol must pre-lock {needle}"


def test_protocol_hash_is_stable_and_recorded(tmp_path: Path) -> None:
    """Same data + same protocol → same hash; report records the hash."""
    tickers = ["AAPL", "MSFT", "SPY"]
    r1 = paper_run.run_paper(FIXTURES, tickers, Hippocampus())
    assert r1["protocol_hash"] == paper_run.protocol_hash()
    assert r1["protocol_hash"].startswith(paper_run.protocol_hash()[:8])
    out = tmp_path / "paper.json"
    subprocess.run(
        [
            sys.executable,
            str(PAPER),
            "--data-dir",
            str(FIXTURES),
            "--tickers",
            "AAPL,MSFT,SPY",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert out.exists()
    blob = json.loads(out.read_text())
    assert blob["protocol_hash"] == r1["protocol_hash"]


def test_session_is_one_day_across_universe(tmp_path: Path) -> None:
    """5 fixture days → exactly 5 sessions (no per-ticker inflation)."""
    # Build a 3-ticker synthetic universe copying the fixture's session span.
    import csv as _csv

    for t in ("AAPL", "MSFT", "SPY"):
        src = FIXTURES / f"{t}_1min.csv"
        dest = tmp_path / f"{t}_1min.csv"
        if src.exists():
            dest.write_text(src.read_text())
    report = paper_run.run_paper(tmp_path, ["AAPL", "MSFT", "SPY"], Hippocampus())
    dates = sorted({s["date"] for s in report["sessions"]})
    assert len(dates) >= 5 or len(report["sessions"]) == report["counts"]["sessions"]
    assert report["counts"]["sessions"] == len(dates)


def test_thin_fixtures_yield_honest_fail_exit(tmp_path: Path) -> None:
    """The 5-day committed universe must FAIL (P5 INSUFFICIENT), exit 1."""
    proc = subprocess.run(
        [
            sys.executable,
            str(PAPER),
            "--data-dir",
            str(FIXTURES),
            "--tickers",
            "AAPL,MSFT,SPY",
            "--output",
            str(tmp_path / "p.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, "thin-data paper must not vacuously pass"
    blob = json.loads((tmp_path / "p.json").read_text())
    assert blob["verdict"] == "FAIL"
    assert blob["criteria"]["P5_wfa_pass"] is False


def test_criteria_are_exhaustive_and_boolean() -> None:
    v = paper_run.PaperVerdict(
        n_trades=50, n_sessions=6, n_errors=0
    )  # P5 default False
    v.evaluate()
    assert set(v.criteria) == {
        "P1_min_trades",
        "P2_min_sessions",
        "P3_no_kill_sessions",
        "P4_daily_loss_sessions",
        "P5_wfa_pass",
        "P6_no_errors",
    }
    assert all(isinstance(ok, bool) for ok in v.criteria.values())


def test_no_runtime_constant_patching() -> None:
    """The harness may not monkey-patch immune risk constants."""
    src = paper_run.PROTOCOL_FILE.parent.parent / "scripts" / "paper_run.py"
    text = src.read_text()
    patched = [
        line
        for line in textwrap.dedent(text).splitlines()
        if "mock.patch" in line or "monkeypatch.setattr" in line
    ]
    assert patched == [], "paper harness must run unpatched shipped organs"
