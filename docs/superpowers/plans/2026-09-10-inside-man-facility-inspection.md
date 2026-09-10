# Inside Man — Facility Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `hanoon_prime.inspection` — a library that verifies every joint of the running stack, consumed by (a) a refactored `production_monitor.py` daemon, (b) a daily Telegram digest, and (c) an on-demand `manifest` command — plus gated auto-heal for three non-trading services and a journal chain re-anchor boot step.

**Architecture:** A `CheckSpec`/`CheckResult` model where each check is a pure read-only function returning `OK|WARN|FAIL|UNVERIFIABLE`. `run_all(ctx)` produces a `Manifest`. `production_monitor.py` is refactored to derive hard rules + bug-catcher anomalies from the manifest (exit codes 0/2/3/4 and the ledger/streak/gate1 bookkeeping preserved). `heal.py` and `reanchor.py` add the gated actions. One set of checks, three faces.

**Tech Stack:** Python 3.12, stdlib only (urllib, subprocess, re, json), pytest, existing pre-commit hooks (black 88, isort, ruff, mypy `--strict`, complexity ≤40 lines/≤3 nest, file length ≤200 lines for `src/`).

**Spec:** `docs/superpowers/specs/2026-09-10-inside-man-facility-inspection-design.md` (the plan argues from the spec; executors read both).

## Global Constraints

- No `print()` in `src/` (R10). CLI output lives only in `hanoon_prime/inspection/__main__.py`.
- Mypy runs `--strict` with `disallow-untyped-defs`, `warn-return-any`, `disallow-incomplete-defs`, `strict=true` on `src/hanoon_prime` (pyproject `[tool.mypy]`, `files=["src/hanoon_prime"]`). Every new function **must** carry full type annotations and never return unguarded `Any`. `Any` values from `dict[str, Any]` reads must be narrowed/`isinstance`-guarded before use in typed positions.
- `src/**/*.py` ≤ 200 lines, every function ≤ 40 lines, nesting ≤ 3 (pre-commit complexity + file-length hooks).
- black line-length 88, isort profile black. Re-run `pre-commit` before commit; if black reformats, `git add` the reformatted files again (only re-stage intended files).
- `scripts/production_state.json`, `runtime/`, `logs/` are gitignored and never staged. `.planning/` never staged.
- Push to **both** remotes: `origin` and `origin-sajib` (same message, same HEAD).
- Never mutate bot state, trades, weights, or the journal from the inspection library. Heal actions limited to: `halim_serve`, `cloudflared`, `ib_gateway_watchdog` (max 2/day each). Journal `chain_reseed` happens only via `start.command` while the stack is fully stopped.
- Team naming: log markers are `ib_adapter Starting (seed=`, `ib_cycle HEARTBEAT`, `ib_cycle CYCLE `, `SESSION SLEEP`. Journal event names: `verdict`, `position_closed`, `pipeline_incident`, `halt`, `ib_state_snapshot`, and (new) `chain_reseed`.

---

### Task 1: check model + runner harness

**Files:**
- Create: `src/hanoon_prime/inspection/__init__.py` (thin exports only — filled in fully at Task 6/7)
- Create: `src/hanoon_prime/inspection/checks.py`
- Test: `tests/test_inspection_checks.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `CheckResult(joint: str, name: str, status: str, detail: str = "", evidence: dict[str, object] = {})`; `CheckSpec(joint, name, fn: Callable[[InspectionContext], CheckResult], hard: bool = False, report: bool = False)`; constants `OK/WARN/FAIL/UNVERIFIABLE`; `CheckFn`; `VALID_ACTIONS`; `run_check(spec, ctx) -> CheckResult`; `MANIFEST_STATUS(results: list[CheckResult], hard_keys: AbstractSet[tuple[str, str]]) -> str`. Later tasks import these names.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inspection_checks.py
"""Unit tests for the check model + runner harness."""
from hanoon_prime.inspection.checks import (
    OK, WARN, FAIL, UNVERIFIABLE, CheckResult, CheckSpec, run_check,
    MANIFEST_STATUS,
)
import pytest


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
        return CheckResult("pipeline", "heartbeat_fresh", OK, evidence={"d": ctx.data()})
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
```

- [ ] **Step 2: Run the test to see it fail**

Run: `.venv/bin/python -m pytest tests/test_inspection_checks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hanoon_prime.inspection'`.

- [ ] **Step 3: Write the implementation**

```python
# src/hanoon_prime/inspection/__init__.py
"""hanoon_prime.inspection — the Inside Man: joint-by-joint verification."""
```

```python
# src/hanoon_prime/inspection/checks.py
"""Check model + runner harness for the Inside Man.

Every joint is verified through CheckSpec functions returning CheckResult.
The harness wraps each fn so an instrument failure is reported as
UNVERIFIABLE — never a raise, and never a false FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AbstractSet, Callable

if TYPE_CHECKING:
    from .ctx import InspectionContext

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
UNVERIFIABLE = "UNVERIFIABLE"

CheckFn = Callable[["InspectionContext"], "CheckResult"]

VALID_ACTIONS = {"BUY", "SELL", "HOLD", "VETOED", "PASS", "OPEN", "CLOSE", "ENTER"}


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check. status is OK|WARN|FAIL|UNVERIFIABLE."""

    joint: str
    name: str
    status: str
    detail: str = ""
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckSpec:
    """Declarative registration of a single check against a joint."""

    joint: str
    name: str
    fn: CheckFn
    hard: bool = False  # FAIL => guardian hard violation (exit 2, streak reset)
    report: bool = False  # non-hard finding => bug-catcher anomaly feed (exit 4)


def run_check(spec: CheckSpec, ctx: "InspectionContext") -> CheckResult:
    """Run one check; convert any instrument failure to UNVERIFIABLE."""
    try:
        return spec.fn(ctx)
    except Exception as exc:  # noqa: BLE001 — instrument failure ≠ verdict
        return CheckResult(
            spec.joint,
            spec.name,
            UNVERIFIABLE,
            detail=f"instrument error: {exc}",
            evidence={"error": str(exc)},
        )


def MANIFEST_STATUS(results: "list[CheckResult]", hard_keys: "AbstractSet[tuple[str, str]]") -> str:
    """Any hard FAIL => FAIL; else any WARN/UNVERIFIABLE => WARN; else OK."""
    hard_fails = [
        r for r in results if (r.joint, r.name) in hard_keys and r.status == FAIL
    ]
    if hard_fails:
        return FAIL
    if any(r.status in (WARN, UNVERIFIABLE) for r in results):
        return WARN
    return OK
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_inspection_checks.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

Run: `git add src/hanoon_prime/inspection/__init__.py src/hanoon_prime/inspection/checks.py tests/test_inspection_checks.py && git commit -m "feat(inspection): check model + runner harness"`
Note: if pre-commit reformats, re-`git add` the reformatted files and re-commit.

---

### Task 2: InspectionContext + live probes

**Files:**
- Create: `src/hanoon_prime/inspection/ctx.py`
- Create: `src/hanoon_prime/inspection/probe.py`
- Modify: `src/hanoon_prime/inspection/__init__.py` (add re-exports)
- Test: `tests/test_inspection_ctx.py` (covers ctx + probe)

**Interfaces:**
- Consumes: `checks.py` names (none directly), standard library only.
- Produces (later tasks rely on these exact names):

```
class InspectionContext:
    base_dir: Path
    telemetry_url: str = "http://127.0.0.1:8080"
    halim_url: str = "http://127.0.0.1:8765"
    prev_journal_count: int | None = None
    heal_enabled: bool = True
    runtime / pid_dir / logs_dir / log_path / state_path / juli_state_path /
    realized_path / journal_path / state_file / venv_python / launch_detached /
    halim_start / halim_log_path  (Path properties)
    pid_file(service: str) -> Path
    git_head() -> str | None

probe:
    health(ctx) -> dict[str, Any]
    snapshot(ctx) -> dict[str, Any]
    runtime_state(ctx) / juli_state(ctx) / ledger(ctx) -> dict[str, Any]
    session_lines(ctx) -> list[str]
    line_ts(line: str) -> float | None
    last_line_age(ctx, marker: re.Pattern[str]) -> float | None
    journal(ctx) -> Journal
    journal_tail(ctx, n: int) -> list[dict[str, Any]]
    journal_chain_state(ctx) -> dict[str, Any]
    halim_probe(ctx) -> str            # "ok" | "asleep" | "down"
    (marker constants: START_MARKER, HEARTBEAT_MARKER, CYCLE_MARKER,
     SLEEP_MARKER, GUARD_MARKER, TRACE_MARKER, SAFETY_HALT_MARKER,
     LEARN_BLOCKED_MARKER)
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inspection_ctx.py
"""Tests for InspectionContext + live-surface probes."""
import json
import re
from pathlib import Path

from hanoon_prime.inspection.ctx import InspectionContext
from hanoon_prime.inspection import probe


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
    assert cs["anchor_seq"] is None       # nothing after the anchor
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_inspection_ctx.py -v`
Expected: FAIL — `ModuleNotFoundError: hanoon_prime.inspection.ctx`.

- [ ] **Step 3: Write the implementation**

Write `ctx.py` and `probe.py` with the exact code below (kept compact; both files stay ≤200 lines).

```python
# src/hanoon_prime/inspection/ctx.py
"""hanoon_prime.inspection.ctx — where everything lives.

Resolves repo paths, pidfiles, git HEAD, and the runtime surfaces the checks
poke. Cheap reads are memoized per InspectionContext instance so one probe
tick touches each source (log, json files, journal, HTTP) only once.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InspectionContext:
    """Paths, urls, and cached probes for one manifest tick."""

    base_dir: Path
    telemetry_url: str = "http://127.0.0.1:8080"
    halim_url: str = "http://127.0.0.1:8765"
    prev_journal_count: int | None = None
    heal_enabled: bool = True
    memo: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.base_dir = self.base_dir.resolve()

    @property
    def runtime(self) -> Path:
        return self.base_dir / "runtime"

    @property
    def pid_dir(self) -> Path:
        return self.runtime / "pids"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def log_path(self) -> Path:
        return self.logs_dir / "hanoon_prime.log"

    @property
    def state_path(self) -> Path:
        return self.runtime / "state.json"

    @property
    def juli_state_path(self) -> Path:
        return self.runtime / "juli_state.json"

    @property
    def realized_path(self) -> Path:
        return self.runtime / "juli_realized.json"

    @property
    def journal_path(self) -> Path:
        return self.runtime / "journal_live.jsonl"

    @property
    def state_file(self) -> Path:
        return self.base_dir / "scripts" / "production_state.json"

    @property
    def venv_python(self) -> Path:
        return self.base_dir / ".venv" / "bin" / "python"

    @property
    def launch_detached(self) -> Path:
        return self.base_dir / "scripts" / "launch_detached.py"

    @property
    def halim_start(self) -> Path:
        return self.base_dir / "scripts" / "halim_start.sh"

    @property
    def halim_log_path(self) -> Path:
        return self.logs_dir / "halim_serve.log"

    def pid_file(self, service: str) -> Path:
        return self.pid_dir / f"{service}.pid"

    def git_head(self) -> str | None:
        """Repo HEAD (cached). None when git is unavailable."""
        key = "git_head"
        if key not in self.memo:
            try:
                out = subprocess.run(
                    ["git", "-C", str(self.base_dir), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.memo[key] = (
                    out.stdout.strip() if out.returncode == 0 else None
                )
            except (OSError, subprocess.SubprocessError):
                self.memo[key] = None
        value = self.memo[key]
        return value if isinstance(value, str) else None
```

```python
# src/hanoon_prime/inspection/probe.py
"""Live pokers: health, snapshot, state, journal, halim probe.

Each function takes an InspectionContext and is memoized per instance so a
tick touches each source at most once.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from ..memory import Journal
from .ctx import InspectionContext

START_MARKER = re.compile(r"ib_adapter\s+Starting \(seed=")
HEARTBEAT_MARKER = re.compile(r"ib_cycle\s+HEARTBEAT")
CYCLE_MARKER = re.compile(r"ib_cycle\s+CYCLE ")
SLEEP_MARKER = re.compile(r"SESSION SLEEP")
GUARD_MARKER = re.compile(r"NETTING GUARD")
TRACE_MARKER = re.compile(r"Traceback \(most recent call last\)")
SAFETY_HALT_MARKER = re.compile(r"SAFETY HALT")
LEARN_BLOCKED_MARKER = re.compile(r"LEARN BLOCKED")
_TS_RE = re.compile(r"(\d\d):(\d\d):(\d\d)\.\d+")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _http_get(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        return {"_unreachable": str(exc)}


def health(ctx: InspectionContext) -> dict[str, Any]:
    if "health" not in ctx.memo:
        ctx.memo["health"] = _http_get(f"{ctx.telemetry_url}/health")
    value = ctx.memo["health"]
    return value if isinstance(value, dict) else {}


def snapshot(ctx: InspectionContext) -> dict[str, Any]:
    if "snapshot" not in ctx.memo:
        ctx.memo["snapshot"] = _http_get(f"{ctx.telemetry_url}/snapshot")
    value = ctx.memo["snapshot"]
    return value if isinstance(value, dict) else {}


def runtime_state(ctx: InspectionContext) -> dict[str, Any]:
    key = "runtime_state"
    if key not in ctx.memo:
        ctx.memo[key] = _read_json(ctx.state_path)
    value = ctx.memo[key]
    return value if isinstance(value, dict) else {}


def juli_state(ctx: InspectionContext) -> dict[str, Any]:
    key = "juli_state"
    if key not in ctx.memo:
        ctx.memo[key] = _read_json(ctx.juli_state_path)
    value = ctx.memo[key]
    return value if isinstance(value, dict) else {}


def ledger(ctx: InspectionContext) -> dict[str, Any]:
    key = "ledger"
    if key not in ctx.memo:
        ctx.memo[key] = _read_json(ctx.state_file)
    value = ctx.memo[key]
    return value if isinstance(value, dict) else {}


def session_lines(ctx: InspectionContext) -> list[str]:
    """Every log line after the LAST 'ib_adapter Starting' marker."""
    key = "session_lines"
    if key in ctx.memo:
        value = ctx.memo[key]
        return list(value) if isinstance(value, list) else []
    session: list[str] = []
    try:
        with open(ctx.log_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if START_MARKER.search(line):
                    session = []
                else:
                    session.append(line)
    except OSError:
        pass
    ctx.memo[key] = session
    return session


def line_ts(line: str) -> float | None:
    """Local-wall-clock timestamp for a log line (HH:MM:SS, no date)."""
    m = _TS_RE.match(line)
    if not m:
        return None
    now = datetime.now()
    try:
        return now.replace(
            hour=int(m.group(1)), minute=int(m.group(2)), second=int(m.group(3))
        ).timestamp()
    except ValueError:
        return None


def last_line_age(
    ctx: InspectionContext, marker: re.Pattern[str]
) -> float | None:
    """Age in seconds of the most recent log line matching marker."""
    latest: float | None = None
    try:
        with open(ctx.log_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if marker.search(line):
                    t = line_ts(line)
                    if t is not None:
                        latest = t
    except OSError:
        return None
    return (time.time() - latest) if latest is not None else None


def journal(ctx: InspectionContext) -> Journal:
    if "journal" not in ctx.memo:
        ctx.memo["journal"] = Journal(ctx.journal_path)
    value = ctx.memo["journal"]
    if isinstance(value, Journal):
        return value
    j = Journal(ctx.journal_path)
    ctx.memo["journal"] = j
    return j


def journal_tail(ctx: InspectionContext, n: int) -> list[dict[str, Any]]:
    return journal(ctx).tail(n)


def journal_chain_state(ctx: InspectionContext) -> dict[str, Any]:
    """Hash-chain anchor + contiguity after the last re-anchor / break."""
    key = "chain"
    if key in ctx.memo:
        value = ctx.memo[key]
        return dict(value) if isinstance(value, dict) else {}
    entries = journal(ctx).entries()
    last_break = -1
    last_reseed = -1
    breaks = 0
    prev: str | None = None
    for i, e in enumerate(entries):
        if e.get("event") == "chain_reseed":
            last_reseed = i
        if e.get("prev_hash") != prev:
            breaks += 1
            last_break = i
        prev = e.get("hash")
    anchor = max(
        last_reseed + 1 if last_reseed >= 0 else 0,
        last_break + 1 if last_break >= 0 else 0,
    )
    gaps = 0
    p: str | None = entries[anchor - 1].get("hash") if anchor > 0 else None
    for e in entries[anchor:]:
        if e.get("prev_hash") != p:
            gaps += 1
        p = e.get("hash")
    ctx.memo[key] = {
        "entries": len(entries),
        "breaks": breaks,
        "last_break_seq": (
            entries[last_break].get("seq") if last_break >= 0 else None
        ),
        "last_reseed_seq": (
            entries[last_reseed].get("seq") if last_reseed >= 0 else None
        ),
        "anchor_seq": (
            entries[anchor].get("seq") if anchor < len(entries) else None
        ),
        "gaps_after_anchor": gaps,
    }
    value = ctx.memo[key]
    return dict(value) if isinstance(value, dict) else {}


def halim_probe(ctx: InspectionContext) -> str:
    """Return ok | asleep (expected post-market) | down (degraded)."""
    key = "halim_probe"
    if key in ctx.memo:
        value = ctx.memo[key]
        return value if isinstance(value, str) else "down"
    try:
        req = urllib.request.Request(
            f"{ctx.halim_url}/v1/complete",
            data=json.dumps(
                {
                    "prompt": 'Reply with exactly: {"ok":1}',
                    "purpose": "health_probe",
                    "priority": "low",
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            res: dict[str, Any] = json.loads(resp.read().decode())
        if res.get("ok"):
            ctx.memo[key] = "ok"
        elif res.get("reason") == "system_asleep":
            ctx.memo[key] = "asleep"
        else:
            ctx.memo[key] = "down"
    except urllib.error.HTTPError as exc:
        try:
            body: dict[str, Any] = json.loads(exc.read().decode(errors="replace"))
        except (json.JSONDecodeError, ValueError):
            ctx.memo[key] = "down"
        else:
            ctx.memo[key] = (
                "asleep" if body.get("reason") == "system_asleep" else "down"
            )
    except Exception:  # noqa: BLE001
        ctx.memo[key] = "down"
    value = ctx.memo[key]
    return value if isinstance(value, str) else "down"
```

Update `__init__.py`:

```python
"""hanoon_prime.inspection — the Inside Man: joint-by-joint verification."""

from .checks import (
    FAIL,
    OK,
    UNVERIFIABLE,
    WARN,
    CheckResult,
    CheckSpec,
    MANIFEST_STATUS,
)
from .ctx import InspectionContext

__all__ = [
    "FAIL",
    "OK",
    "UNVERIFIABLE",
    "WARN",
    "CheckResult",
    "CheckSpec",
    "InspectionContext",
    "MANIFEST_STATUS",
]
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_inspection_ctx.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

Run: `git add src/hanoon_prime/inspection/ctx.py src/hanoon_prime/inspection/probe.py src/hanoon_prime/inspection/__init__.py tests/test_inspection_ctx.py && git commit -m "feat(inspection): context + live probes"`
Re-stage if pre-commit reformats.

---

### Task 3: processes + identity joint

**Files:**
- Create: `src/hanoon_prime/inspection/system.py`
- Test: `tests/test_inspection_system.py`

**Interfaces:**
- Consumes: `ChecksResult/CheckSpec/OK/FAIL/UNVERIFIABLE` from `..checks`; `InspectionContext`; `probe.health`.
- Produces: check functions `bot_alive`, `halim_alive`, `monitor_alive`, `cloudflared_alive`, `gateway_watchdog_alive`, `single_bot`, `bot_from_trusted_checkout` — each `(ctx: InspectionContext) -> CheckResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inspection_system.py
""">Processes + identity joint checks."""
import os

from hanoon_prime.inspection import system
from hanoon_prime.inspection.checks import FAIL, OK, UNVERIFIABLE
from hanoon_prime.inspection.ctx import InspectionContext


def _ctx(tmp_path) -> InspectionContext:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.pid_dir.mkdir(parents=True, exist_ok=True)
    return ctx


def test_service_ok_when_pidfile_alive(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    ctx.pid_file("hanoon_prime").write_text(str(os.getpid()))
    r = system.bot_alive(ctx)
    assert r.status == OK
    assert r.evidence.get("pid") == os.getpid()


def test_service_fail_on_stale_pidfile(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    ctx.pid_file("hanoon_prime").write_text("999999")
    assert system.bot_alive(ctx).status == FAIL


def test_service_pgrep_fallback_when_no_pidfile(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(system, "_pgrep", lambda p: [1234])
    assert system.cloudflared_alive(ctx).status == OK


def test_service_fail_when_no_pidfile_and_down(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(system, "_pgrep", lambda p: [])
    assert system.cloudflared_alive(ctx).status == FAIL


def test_single_bot_double_is_fail(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(system, "_pgrep", lambda p: [1, 2])
    assert system.single_bot(ctx).status == FAIL


def test_single_bot_ok(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(system, "_pgrep", lambda p: [123])
    assert system.single_bot(ctx).status == OK


def test_bot_identity_fail_on_stray_cmdline(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    ctx.pid_file("hanoon_prime").write_text(str(os.getpid()))
    monkeypatch.setattr(system, "_cmdline", lambda pid: "/some/other/script.py")
    assert system.bot_from_trusted_checkout(ctx).status == FAIL


def test_bot_identity_ok(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    ctx.pid_file("hanoon_prime").write_text(str(os.getpid()))
    monkeypatch.setattr(
        system, "_cmdline", lambda pid: "/x/.venv/bin/python -m hanoon_prime.cli"
    )
    assert system.bot_from_trusted_checkout(ctx).status == OK


def test_identity_unverifiable_when_bot_down(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    assert system.bot_from_trusted_checkout(ctx).status == UNVERIFIABLE
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_inspection_system.py -v`
Expected: FAIL — `ModuleNotFoundError: hanoon_prime.inspection.system`.

- [ ] **Step 3: Write the implementation**

```python
# src/hanoon_prime/inspection/system.py
"""processes + identity joint checks."""

from __future__ import annotations

import os
import subprocess

from .checks import FAIL, OK, UNVERIFIABLE, CheckResult
from .ctx import InspectionContext


def _pidread(
    ctx: InspectionContext, service: str
) -> tuple[int | None, bool, str]:
    """(pid, alive, why). why explains a missing/dead pidfile."""
    pf = ctx.pid_file(service)
    if not pf.exists():
        return None, False, "no pidfile"
    try:
        pid = int(pf.read_text().strip() or "0")
    except (OSError, ValueError):
        return None, False, "unreadable pidfile"
    if pid <= 0:
        return None, False, "empty pidfile"
    try:
        os.kill(pid, 0)
        return pid, True, ""
    except ProcessLookupError:
        return pid, False, "stale pidfile (pid dead)"
    except PermissionError:
        return pid, True, ""


def _pgrep(pattern: str) -> list[int] | None:
    """Pids matching pattern; None on tooling failure."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return []
    return [int(p) for p in out.stdout.split() if p.isdigit()]


def _cmdline(pid: int) -> str:
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _service(ctx: InspectionContext, service: str, pattern: str) -> CheckResult:
    pid, alive, why = _pidread(ctx, service)
    if alive:
        return CheckResult("processes", f"{service}_alive", OK, evidence={"pid": pid})
    if pid is None:
        procs = _pgrep(pattern)
        if procs:
            return CheckResult(
                "processes",
                f"{service}_alive",
                OK,
                detail="alive without pidfile",
                evidence={"pids": procs},
            )
    return CheckResult("processes", f"{service}_alive", FAIL, detail=why or "pid dead")


def bot_alive(ctx: InspectionContext) -> CheckResult:
    return _service(ctx, "hanoon_prime", r"hanoon_prime\.cli")


def halim_alive(ctx: InspectionContext) -> CheckResult:
    return _service(ctx, "halim_serve", r"halim/serve\.py")


def monitor_alive(ctx: InspectionContext) -> CheckResult:
    return _service(ctx, "production_monitor", r"production_monitor\.py --daemon")


def cloudflared_alive(ctx: InspectionContext) -> CheckResult:
    return _service(ctx, "cloudflared", r"cloudflared")


def gateway_watchdog_alive(ctx: InspectionContext) -> CheckResult:
    return _service(ctx, "ib_gateway_watchdog", r"ib_gateway_watchdog")


def single_bot(ctx: InspectionContext) -> CheckResult:
    procs = _pgrep(r"hanoon_prime\.cli")
    if procs is None:
        return CheckResult("processes", "single_bot", UNVERIFIABLE, detail="pgrep failed")
    if len(procs) > 1:
        return CheckResult(
            "processes", "single_bot", FAIL, detail="double bot process", evidence={"pids": procs}
        )
    if len(procs) == 1:
        return CheckResult("processes", "single_bot", OK, evidence={"pid": procs[0]})
    return CheckResult("processes", "single_bot", OK, detail="no bot pidfile fallback")


def bot_from_trusted_checkout(ctx: InspectionContext) -> CheckResult:
    pid, alive, why = _pidread(ctx, "hanoon_prime")
    if not alive:
        return CheckResult("identity", "bot_from_trusted_checkout", UNVERIFIABLE, detail="bot not running")
    cmd = _cmdline(pid)
    if "hanoon_prime.cli" not in cmd:
        return CheckResult(
            "identity", "bot_from_trusted_checkout", FAIL, detail="stray/wrong bot cmdline", evidence={"cmdline": cmd}
        )
    return CheckResult("identity", "bot_from_trusted_checkout", OK, evidence={"cmdline": cmd})
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_inspection_system.py -v`
Expected: PASS (9 passed). Also run `.venv/bin/python -m mypy src/hanoon_prime/inspection` — expected: no new errors.

- [ ] **Step 5: Commit**

Run: `git add src/hanoon_prime/inspection/system.py tests/test_inspection_system.py && git commit -m "feat(inspection): processes + identity checks"`
Re-stage if pre-commit reformats.

---

### Task 4: telemetry / pipeline / session / notify joint

**Files:**
- Create: `src/hanoon_prime/inspection/runtime.py`
- Test: `tests/test_inspection_runtime.py`

**Interfaces:**
- Consumes: `._telegram._get_token/_get_chat_id` (already typed `-> Optional[str]`), `probe` helpers, `runtime_state`, `ledger`.
- Produces: `health_ok`, `snapshot_fresh`, `positions_surface`, `heartbeat_fresh`, `cycle_flows_when_active`, `sleep_is_expected`, `state_matches_clock`, `positions_reconciled`, `telegram_configured`, `send_healthy` — each `(ctx) -> CheckResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inspection_runtime.py
"""telemetry / pipeline / session / notify joint checks."""
import datetime

from hanoon_prime.inspection import runtime
from hanoon_prime.inspection.checks import FAIL, OK, WARN
from hanoon_prime.inspection.ctx import InspectionContext


def _now_t() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _healthy_memo(ctx: InspectionContext, *, session: str = "post_market", active: bool = False) -> InspectionContext:
    ctx.memo["health"] = {
        "status": "ok", "connected": True, "session": session,
        "session_active": active, "position_count": 0, "positions": [],
    }
    ctx.memo["snapshot"] = {"health": {"status": "ok"}}
    return ctx


def test_health_ok_pass(tmp_path) -> None:
    assert runtime.health_ok(_healthy_memo(InspectionContext(base_dir=tmp_path))).status == OK


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
    assert runtime.heartbeat_fresh(ctx).status == OK


def test_cycle_only_expected_when_active(tmp_path) -> None:
    ctx = _healthy_memo(InspectionContext(base_dir=tmp_path), active=True)
    assert runtime.cycle_flows_when_active(ctx).status == FAIL  # active but no cycles
    log = ctx.log_path
    log.parent.mkdir(parents=True)
    log.write_text(f"{_now_t()}.000 INFO ib_cycle CYCLE bars=1 open=0 d=0 x=0\n")
    assert runtime.cycle_flows_when_active(ctx).status == OK


def test_sleep_is_expected(tmp_path) -> None:
    ctx = _healthy_memo(InspectionContext(base_dir=tmp_path), active=False)
    assert runtime.sleep_is_expected(ctx).status == WARN  # inactive, no marker
    log = ctx.log_path
    log.parent.mkdir(parents=True)
    log.write_text(
        f"{_now_t()}.000 INFO ib_cycle SESSION SLEEP: post_market inactive — whole system idle\n"
    )
    assert runtime.sleep_is_expected(ctx).status == OK


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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_inspection_runtime.py -v`
Expected: FAIL — `ModuleNotFoundError: hanoon_prime.inspection.runtime`.

- [ ] **Step 3: Write the implementation**

```python
# src/hanoon_prime/inspection/runtime.py
"""telemetry, pipeline, session, notify joint checks."""

from __future__ import annotations

import time

from .._telegram import _get_chat_id, _get_token
from .checks import FAIL, OK, UNVERIFIABLE, WARN, CheckResult
from .ctx import InspectionContext
from .probe import (
    CYCLE_MARKER,
    HEARTBEAT_MARKER,
    SLEEP_MARKER,
    health,
    last_line_age,
    ledger,
    runtime_state,
    session_lines,
    snapshot,
)

KNOWN_SESSIONS = {"pre_market", "regular", "post_market", "inactive"}
CYCLE_STALE_SEC = 300.0


def health_ok(ctx: InspectionContext) -> CheckResult:
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult("telemetry", "health_ok", FAIL, detail=str(h["_unreachable"]))
    ok = h.get("status") == "ok" and bool(h.get("connected", False))
    return CheckResult("telemetry", "health_ok", OK if ok else FAIL,
                       detail=f"status={h.get('status')} connected={h.get('connected', False)}")


def snapshot_fresh(ctx: InspectionContext) -> CheckResult:
    s = snapshot(ctx)
    if s.get("_unreachable"):
        return CheckResult("telemetry", "snapshot_fresh", FAIL, detail=str(s["_unreachable"]))
    good = bool(s) and isinstance(s.get("health"), dict)
    return CheckResult("telemetry", "snapshot_fresh", OK if good else FAIL,
                       detail="snapshot empty/unreachable" if not good else "snapshot present")


def positions_surface(ctx: InspectionContext) -> CheckResult:
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult("telemetry", "positions_surface", UNVERIFIABLE, detail=str(h["_unreachable"]))
    if "position_count" in h:
        return CheckResult("telemetry", "positions_surface", OK, evidence={"position_count": h.get("position_count")})
    return CheckResult("telemetry", "positions_surface", WARN, detail="no position_count field in /health")


def heartbeat_fresh(ctx: InspectionContext) -> CheckResult:
    age = last_line_age(ctx, HEARTBEAT_MARKER)
    if age is None:
        return CheckResult("pipeline", "heartbeat_fresh", FAIL, detail="no HEARTBEAT in log")
    return CheckResult("pipeline", "heartbeat_fresh", OK if age <= CYCLE_STALE_SEC else FAIL,
                       detail=f"last HEARTBEAT {age:.0f}s ago")


def cycle_flows_when_active(ctx: InspectionContext) -> CheckResult:
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult("pipeline", "cycle_flows_when_active", UNVERIFIABLE, detail=str(h["_unreachable"]))
    if not h.get("session_active"):
        return CheckResult("pipeline", "cycle_flows_when_active", OK, detail="session inactive — cycles not expected")
    age = last_line_age(ctx, CYCLE_MARKER)
    if age is None:
        return CheckResult("pipeline", "cycle_flows_when_active", FAIL, detail="active session, no CYCLE since start")
    return CheckResult("pipeline", "cycle_flows_when_active", OK if age <= CYCLE_STALE_SEC else FAIL,
                       detail=f"last CYCLE {age:.0f}s ago")


def sleep_is_expected(ctx: InspectionContext) -> CheckResult:
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult("pipeline", "sleep_is_expected", UNVERIFIABLE, detail=str(h["_unreachable"]))
    active = bool(h.get("session_active", False))
    sleep_lines = [ln for ln in session_lines(ctx) if SLEEP_MARKER.search(ln)]
    if not active:
        if sleep_lines:
            return CheckResult("pipeline", "sleep_is_expected", OK, detail=f"asleep: {sleep_lines[-1].strip()[:48]}")
        return CheckResult("pipeline", "sleep_is_expected", WARN, detail="inactive but no SESSION SLEEP marker")
    return CheckResult("pipeline", "sleep_is_expected", OK, detail="session active")


def state_matches_clock(ctx: InspectionContext) -> CheckResult:
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult("session", "state_matches_clock", UNVERIFIABLE, detail=str(h["_unreachable"]))
    raw = str(h.get("session", "")).split(" ")[0].lower()
    if raw in KNOWN_SESSIONS:
        return CheckResult("session", "state_matches_clock", OK, detail=f"session={h.get('session')}")
    return CheckResult("session", "state_matches_clock", WARN, detail=f"unexpected session {h.get('session')!r}")


def positions_reconciled(ctx: InspectionContext) -> CheckResult:
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult("session", "positions_reconciled", UNVERIFIABLE, detail=str(h["_unreachable"]))
    ib_count = int(h.get("position_count", 0) or 0)
    po = runtime_state(ctx).get("brain_state", {}).get("positions_open")
    brain_count = po if isinstance(po, int) else -1
    if ib_count == brain_count:
        return CheckResult("session", "positions_reconciled", OK, evidence={"ib": ib_count, "brain": brain_count})
    if brain_count == 0 and ib_count > 0:
        return CheckResult("session", "positions_reconciled", OK,
                           detail=f"{ib_count} residual IB positions, bot holds none",
                           evidence={"ib": ib_count, "brain": 0})
    return CheckResult("session", "positions_reconciled", WARN,
                       detail=f"IB {ib_count} vs bot {brain_count}", evidence={"ib": ib_count, "brain": brain_count})


def telegram_configured(ctx: InspectionContext) -> CheckResult:
    token, chat = _get_token(), _get_chat_id()
    ok = bool(token and chat)
    return CheckResult("notify", "telegram_configured", OK if ok else WARN,
                       detail="configured" if ok else "token/chat missing")


def send_healthy(ctx: InspectionContext) -> CheckResult:
    notify = ledger(ctx).get("notify")
    last_ok = float(notify.get("last_ok", 0.0)) if isinstance(notify, dict) else 0.0
    fresh = last_ok > 0 and (time.time() - last_ok) < 86400.0
    detail = (
        f"last successful send {int(time.time() - last_ok)}s ago" if last_ok else "no send recorded yet"
    )
    return CheckResult("notify", "send_healthy", OK if fresh else WARN, detail=detail)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_inspection_runtime.py -v`
Expected: PASS (13 passed). Run `.venv/bin/python -m mypy src/hanoon_prime/inspection` — no new errors.

- [ ] **Step 5: Commit**

Run: `git add src/hanoon_prime/inspection/runtime.py tests/test_inspection_runtime.py && git commit -m "feat(inspection): telemetry/pipeline/session/notify checks"`
Re-stage if pre-commit reformats.

---

### Task 5: safety + purity joints

**Files:**
- Create: `src/hanoon_prime/inspection/safety.py`
- Create: `src/hanoon_prime/inspection/purity.py`
- Test: `tests/test_inspection_safety_purity.py`

**Interfaces:**
- Consumes: `probe` markers + `session_lines`/`health`/`runtime_state`/`juli_state`/`journal_tail`.
- Produces: safety checks `no_netting_guard`, `no_traceback`, `no_safety_halt`, `no_learn_blocked`, `policy_flags`, `drawdown_bound`; purity checks `no_test_episodes`, `weights_finite_in_band`, `brain_fields_bounded`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inspection_safety_purity.py
"""safety + purity joint checks."""
from hanoon_prime.inspection import purity, safety
from hanoon_prime.inspection.checks import FAIL, OK, WARN
from hanoon_prime.inspection.ctx import InspectionContext


def _log(ctx: InspectionContext, lines: list[str]) -> None:
    ctx.log_path.parent.mkdir(parents=True)
    ctx.log_path.write_text("".join(lines))


def test_guard_trigger_fails(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    _log(ctx, ["12:00:00.000 INFO ib_cycle NETTING GUARD: blocked reversal\n"])
    assert safety.no_netting_guard(ctx).status == FAIL
    _log(ctx, ["12:00:00.000 INFO ib_cycle HEARTBEAT open=0\n"])
    assert safety.no_netting_guard(ctx).status == OK


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
    ctx.memo["runtime_state"] = {"brain_state": {"policy_state": {"enabled": False, "authorized": True}}}
    assert safety.policy_flags(ctx).status == WARN


def test_drawdown_ok_when_equity_zero(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["runtime_state"] = {"brain_state": {"policy_state": {"equity": 0.0, "daily_pnl": -100.0}}}
    assert safety.drawdown_bound(ctx).status == OK  # rule re-arms after equity sync


def test_drawdown_within_and_break(tmp_path) -> None:
    base = {"brain_state": {"policy_state": {"equity": 10000.0, "daily_pnl": -80.0}}}
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["runtime_state"] = base
    assert safety.drawdown_bound(ctx).status == OK  # -0.8%
    ctx.memo["runtime_state"] = {"brain_state": {"policy_state": {"equity": 10000.0, "daily_pnl": -200.0}}}
    assert safety.drawdown_bound(ctx).status == FAIL  # -2%


def test_no_test_episodes(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["juli_state"] = {"episodes": [{"ticker": "NVDA", "vector": []}, {"ticker": "TEST", "vector": []}]}
    assert purity.no_test_episodes(ctx).status == FAIL
    ctx.memo["juli_state"] = {"episodes": [{"ticker": "NVDA", "vector": []}]}
    assert purity.no_test_episodes(ctx).status == OK


def test_weights_bounds(tmp_path) -> None:
    ok_weights = {f"w{i}": 0.1 for i in range(20)}
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["juli_state"] = {"weights": ok_weights}
    assert purity.weights_finite_in_band(ctx).status == OK
    ctx.memo["juli_state"] = {"weights": {"w0": 5.0, **{f"w{i}": 0.1 for i in range(1, 20)}}}
    assert purity.weights_finite_in_band(ctx).status == FAIL  # out of [-2,2]
    ctx.memo["juli_state"] = {"weights": {"w0": float("nan"), **{f"w{i}": 0.1 for i in range(1, 20)}}}
    assert purity.weights_finite_in_band(ctx).status == FAIL  # NaN
    ctx.memo["juli_state"] = {"weights": {"w0": 0.1}}
    assert purity.weights_finite_in_band(ctx).status == FAIL  # sparse


def test_brain_fields_bounded(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    good = {"brain_state": {"threshold": 0.58, "pred_error": 0.4, "risk_ceiling": 1.0, "positions_open": 2}}
    ctx.memo["runtime_state"] = good
    assert purity.brain_fields_bounded(ctx).status == OK
    ctx.memo["runtime_state"] = {"brain_state": {"threshold": 0.9, "pred_error": 1.5, "risk_ceiling": -1.0, "positions_open": -3}}
    assert purity.brain_fields_bounded(ctx).status == FAIL
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_inspection_safety_purity.py -v`
Expected: FAIL — `ModuleNotFoundError: hanoon_prime.inspection.safety`.

- [ ] **Step 3: Write the implementation**

```python
# src/hanoon_prime/inspection/safety.py
"""safety joint checks."""

from __future__ import annotations

from .checks import FAIL, OK, UNVERIFIABLE, WARN, CheckResult
from .ctx import InspectionContext
from .probe import (
    GUARD_MARKER,
    LEARN_BLOCKED_MARKER,
    SAFETY_HALT_MARKER,
    TRACE_MARKER,
    health,
    runtime_state,
    session_lines,
)


def _counts(ctx: InspectionContext) -> dict[str, int]:
    lines = session_lines(ctx)
    return {
        "guard": sum(1 for ln in lines if GUARD_MARKER.search(ln)),
        "tracebacks": sum(1 for ln in lines if TRACE_MARKER.search(ln)),
        "safety_halt": sum(1 for ln in lines if SAFETY_HALT_MARKER.search(ln)),
        "learn_blocked": sum(1 for ln in lines if LEARN_BLOCKED_MARKER.search(ln)),
    }


def _zero_check(ctx: InspectionContext, name: str, value: int, label: str) -> CheckResult:
    if value:
        return CheckResult("safety", name, FAIL, detail=f"{value} {label}(s) since last start")
    return CheckResult("safety", name, OK, detail="none")


def no_netting_guard(ctx: InspectionContext) -> CheckResult:
    return _zero_check(ctx, "no_netting_guard", _counts(ctx)["guard"], "NETTING GUARD")


def no_traceback(ctx: InspectionContext) -> CheckResult:
    return _zero_check(ctx, "no_traceback", _counts(ctx)["tracebacks"], "Traceback")


def no_safety_halt(ctx: InspectionContext) -> CheckResult:
    return _zero_check(ctx, "no_safety_halt", _counts(ctx)["safety_halt"], "SAFETY HALT")


def no_learn_blocked(ctx: InspectionContext) -> CheckResult:
    return _zero_check(ctx, "no_learn_blocked", _counts(ctx)["learn_blocked"], "LEARN BLOCKED")


def policy_flags(ctx: InspectionContext) -> CheckResult:
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult("safety", "policy_flags", UNVERIFIABLE, detail=str(h["_unreachable"]))
    pol = runtime_state(ctx).get("brain_state", {}).get("policy_state", {})
    if not isinstance(pol, dict):
        return CheckResult("safety", "policy_flags", UNVERIFIABLE, detail="policy_state missing")
    enabled = bool(pol.get("enabled", False))
    authorized = bool(pol.get("authorized", True))
    if not h.get("session_active"):
        return CheckResult("safety", "policy_flags", OK, detail="session inactive — idle expected")
    if not enabled:
        return CheckResult("safety", "policy_flags", WARN, detail="safety net disabled while session active")
    if not authorized:
        return CheckResult("safety", "policy_flags", WARN, detail="bot not authorized")
    return CheckResult("safety", "policy_flags", OK)


def drawdown_bound(ctx: InspectionContext) -> CheckResult:
    pol = runtime_state(ctx).get("brain_state", {}).get("policy_state", {})
    if not isinstance(pol, dict):
        return CheckResult("safety", "drawdown_bound", UNVERIFIABLE, detail="policy_state missing")
    equity = float(pol.get("equity", 0.0) or 0.0)
    if equity <= 0:
        return CheckResult("safety", "drawdown_bound", OK, detail="equity 0 — rule armed after first equity sync")
    dpnl = float(pol.get("daily_pnl", 0.0) or 0.0)
    pct = -100.0 * dpnl / equity
    ok = pct <= 1.0
    detail = f"daily_pnl {dpnl:.2f}" + (f" = -{pct:.2f}% (floor -1.0%)" if not ok else f" within floor (-{pct:.2f}%)")
    return CheckResult("safety", "drawdown_bound", OK if ok else FAIL, detail=detail)
```

```python
# src/hanoon_prime/inspection/purity.py
"""purity joint checks: hermetic learning, weights, brain fields."""

from __future__ import annotations

from .checks import FAIL, OK, CheckResult
from .ctx import InspectionContext
from .probe import journal_tail, juli_state, runtime_state

POLLUTED = {"T", "TEST"}


def no_test_episodes(ctx: InspectionContext) -> CheckResult:
    js = juli_state(ctx)
    eps = [ep for ep in (js.get("episodes") or []) if isinstance(ep, dict)]
    bad = [ep.get("ticker") for ep in eps if ep.get("ticker") in POLLUTED]
    if bad:
        return CheckResult("purity", "no_test_episodes", FAIL, detail=f"{len(bad)} test episode(s)", evidence={"tickers": bad[:5]})
    rows = journal_tail(ctx, 400)
    bad_rows = [r.get("ticker") for r in rows if r.get("ticker") in POLLUTED]
    if bad_rows:
        return CheckResult("purity", "no_test_episodes", FAIL, detail=f"{len(bad_rows)} journal test row(s)", evidence={"tickers": bad_rows[:5]})
    return CheckResult("purity", "no_test_episodes", OK)


def weights_finite_in_band(ctx: InspectionContext) -> CheckResult:
    w = juli_state(ctx).get("weights")
    if not isinstance(w, dict):
        return CheckResult("purity", "weights_finite_in_band", FAIL, detail="weights missing/untyped")
    vals = [v for v in w.values() if isinstance(v, (int, float))]
    if len(vals) < 10:
        return CheckResult("purity", "weights_finite_in_band", FAIL, detail=f"sparse weights ({len(vals)})")
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        return CheckResult("purity", "weights_finite_in_band", FAIL, detail="NaN/inf weight")
    if not all(-2.0 <= v <= 2.0 for v in vals):
        return CheckResult("purity", "weights_finite_in_band", FAIL, detail="weight outside [-2,2]")
    return CheckResult("purity", "weights_finite_in_band", OK, evidence={"n": len(vals)})


def brain_fields_bounded(ctx: InspectionContext) -> CheckResult:
    bs = runtime_state(ctx).get("brain_state", {})
    if not isinstance(bs, dict):
        return CheckResult("purity", "brain_fields_bounded", FAIL, detail="brain_state missing")
    issues: list[str] = []
    t = bs.get("threshold")
    if not (isinstance(t, (int, float)) and 0.45 <= t <= 0.70):
        issues.append(f"threshold={t!r}")
    pe = bs.get("pred_error")
    if pe is not None and not (isinstance(pe, (int, float)) and 0.0 <= pe <= 1.0):
        issues.append(f"pred_error={pe!r}")
    rc = bs.get("risk_ceiling")
    if not (isinstance(rc, (int, float)) and rc > 0):
        issues.append(f"risk_ceiling={rc!r}")
    po = bs.get("positions_open")
    if not (isinstance(po, int) and po >= 0):
        issues.append(f"positions_open={po!r}")
    if issues:
        return CheckResult("purity", "brain_fields_bounded", FAIL, detail="; ".join(issues))
    return CheckResult("purity", "brain_fields_bounded", OK)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_inspection_safety_purity.py -v`
Expected: PASS (9 passed). Run `.venv/bin/python -m mypy src/hanoon_prime/inspection` — no new errors.

- [ ] **Step 5: Commit**

Run: `git add src/hanoon_prime/inspection/safety.py src/hanoon_prime/inspection/purity.py tests/test_inspection_safety_purity.py && git commit -m "feat(inspection): safety + purity checks"`
Re-stage if pre-commit reformats.

---

### Task 6: memory + execution oracle + halim joints

**Files:**
- Create: `src/hanoon_prime/inspection/journals.py`
- Create: `src/hanoon_prime/inspection/oracle.py`
- Create: `src/hanoon_prime/inspection/halim.py`
- Test: `tests/test_inspection_memory_oracle.py`

**Interfaces:**
- Consumes: `Journal` (from `..memory`), `probe.journal/journal_tail/journal_chain_state/halim_probe/health/runtime_state`.
- Produces: memory checks `journal_grows`, `seq_forward`, `verdicts_valid`, `chain_intact_from_anchor`; oracle checks `enters_minted`, `closes_reconciled`, `equity_synced`; halim check `halim_state_matches_clock`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inspection_memory_oracle.py
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
    jp.write_text(json.dumps({"seq": 0, "event": "verdict", "action": "FOO", "score": 0.5, "ticker": "NVDA"}) + "\n")
    ctx = InspectionContext(base_dir=tmp_path)
    assert journals.verdicts_valid(ctx).status == FAIL


def test_verdicts_valid_nan_score(tmp_path) -> None:
    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    jp.write_text(json.dumps({"seq": 0, "event": "verdict", "action": "HOLD", "score": float("nan"), "ticker": "NVDA"}) + "\n")
    ctx = InspectionContext(base_dir=tmp_path)
    assert journals.verdicts_valid(ctx).status == FAIL


def test_seq_forward_gap_is_warn(tmp_path) -> None:
    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    jp.write_text(
        json.dumps({"seq": 0, "event": "verdict", "action": "HOLD", "score": 0.5, "ticker": "NVDA"}) + "\n"
        + json.dumps({"seq": 5, "event": "verdict", "action": "HOLD", "score": 0.5, "ticker": "NVDA"}) + "\n"
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
    j.append({"event": "verdict", "action": "ENTER", "score": 0.6, "ticker": "ACCL", "direction": 1})
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["health"] = {"positions": [], "position_count": 0}
    assert oracle.enters_minted(ctx).status == WARN  # unminted ENTER
    j.append({"event": "verdict", "action": "ENTER", "score": 0.6, "ticker": "NVDA", "direction": 1})
    ctx.memo["health"] = {"positions": ["NVDA"], "position_count": 1}
    assert oracle.enters_minted(ctx).status == OK


def test_oracle_closes_reconciled(tmp_path) -> None:
    from hanoon_prime.memory import Journal
    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    j = Journal(jp)
    j.append({"event": "position_closed", "ticker": "NVDA", "pnl": 0.0, "entry_price": 10.0, "shares": 1.0})
    ctx = InspectionContext(base_dir=tmp_path)
    assert oracle.closes_reconciled(ctx).status == OK
    j.append({"event": "position_closed", "ticker": "NVDA"})
    assert oracle.closes_reconciled(ctx).status == WARN


def test_oracle_equity_synced(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    ctx.memo["runtime_state"] = {"brain_state": {"policy_state": {"equity_synced": False, "equity": 0.0}}}
    assert oracle.equity_synced(ctx).status == WARN
    ctx.memo["runtime_state"] = {"brain_state": {"policy_state": {"equity_synced": True, "equity": 100.0}}}
    assert oracle.equity_synced(ctx).status == OK


def test_halim_state_matches_clock(tmp_path, monkeypatch) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    monkeypatch.setattr(halim, "halim_probe", lambda c: "ok")
    assert halim.halim_state_matches_clock(ctx).status == OK
    monkeypatch.setattr(halim, "halim_probe", lambda c: "asleep")
    assert halim.halim_state_matches_clock(ctx).status == OK
    monkeypatch.setattr(halim, "halim_probe", lambda c: "down")
    assert halim.halim_state_matches_clock(ctx).status == FAIL
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_inspection_memory_oracle.py -v`
Expected: FAIL — `ModuleNotFoundError: hanoon_prime.inspection.journals`.

- [ ] **Step 3: Write the implementation**

```python
# src/hanoon_prime/inspection/journals.py
"""memory joint checks — journal growth, seq, verdict validity, chain anchor."""

from __future__ import annotations

from .checks import FAIL, OK, WARN, VALID_ACTIONS, CheckResult
from .ctx import InspectionContext
from .probe import journal, journal_chain_state, journal_tail


def journal_grows(ctx: InspectionContext) -> CheckResult:
    cur = journal(ctx).count()
    prev = ctx.prev_journal_count
    if prev is None:
        return CheckResult("memory", "journal_grows", OK, evidence={"count": cur})
    ok = cur >= prev
    return CheckResult("memory", "journal_grows", OK if ok else FAIL, detail=f"count {cur} vs prev {prev}")


def seq_forward(ctx: InspectionContext) -> CheckResult:
    rows = journal_tail(ctx, 200)
    gaps = 0
    prev_seq: int | None = None
    for r in rows:
        s = r.get("seq")
        if isinstance(s, int):
            if prev_seq is not None and s != prev_seq + 1:
                gaps += 1
            prev_seq = s
    return CheckResult("memory", "seq_forward", OK if gaps == 0 else WARN,
                       detail=f"{gaps} seq gap(s) in last {len(rows)}")


def verdicts_valid(ctx: InspectionContext) -> CheckResult:
    rows = journal_tail(ctx, 60)
    sample = [r for r in rows if r.get("event") == "verdict"]
    bad_actions = [r for r in sample if str(r.get("action", "")).upper() not in VALID_ACTIONS]
    nan_scores = [
        r for r in sample
        if not isinstance(r.get("score"), (int, float)) or r.get("score") != r.get("score")
    ]
    if bad_actions or nan_scores:
        return CheckResult("memory", "verdicts_valid", FAIL,
                           detail=f"{len(bad_actions)} bad action, {len(nan_scores)} NaN score")
    return CheckResult("memory", "verdicts_valid", OK, evidence={"sample": len(sample)})


def chain_intact_from_anchor(ctx: InspectionContext) -> CheckResult:
    cs = journal_chain_state(ctx)
    ok = bool(cs.get("gaps_after_anchor") == 0)
    return CheckResult("memory", "chain_intact_from_anchor", OK if ok else FAIL,
                       detail=f"anchor_seq={cs.get('anchor_seq')} gaps={cs.get('gaps_after_anchor')}",
                       evidence=cs)
```

```python
# src/hanoon_prime/inspection/oracle.py
"""execution oracle joint checks — verdict->fill reconciliation."""

from __future__ import annotations

from .checks import OK, UNVERIFIABLE, WARN, CheckResult
from .ctx import InspectionContext
from .probe import health, journal_tail, runtime_state


def _events(ctx: InspectionContext, event: str) -> list[dict]:
    return [r for r in journal_tail(ctx, 200) if r.get("event") == event]


def enters_minted(ctx: InspectionContext) -> CheckResult:
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult("execution_oracle", "enters_minted", UNVERIFIABLE, detail=str(h["_unreachable"]))
    positions = set(h.get("positions") or [])
    enters = [r for r in _events(ctx, "verdict") if r.get("action") == "ENTER"]
    closed = {r.get("ticker") for r in _events(ctx, "position_closed")}
    unmatched = [
        r.get("ticker") for r in enters
        if r.get("ticker") not in positions and r.get("ticker") not in closed
    ]
    if not unmatched:
        return CheckResult("execution_oracle", "enters_minted", OK,
                           evidence={"enters": len(enters), "unmatched": 0})
    return CheckResult("execution_oracle", "enters_minted", WARN,
                       detail=f"{len(unmatched)} ENTER without fill or IB position",
                       evidence={"unmatched": unmatched[:8]})


def closes_reconciled(ctx: InspectionContext) -> CheckResult:
    closed = _events(ctx, "position_closed")
    missing = sum(1 for r in closed if not all(k in r for k in ("pnl", "entry_price", "shares")))
    if not missing:
        return CheckResult("execution_oracle", "closes_reconciled", OK, evidence={"closes": len(closed)})
    return CheckResult("execution_oracle", "closes_reconciled", WARN,
                       detail=f"{missing} close(s) missing pnl/entry_price/shares")


def equity_synced(ctx: InspectionContext) -> CheckResult:
    pol = runtime_state(ctx).get("brain_state", {}).get("policy_state", {})
    if not isinstance(pol, dict):
        return CheckResult("execution_oracle", "equity_synced", UNVERIFIABLE, detail="policy_state missing")
    synced = bool(pol.get("equity_synced", False))
    equity = float(pol.get("equity", 0.0) or 0.0)
    if synced and equity > 0:
        return CheckResult("execution_oracle", "equity_synced", OK, detail=f"equity {equity:.2f}")
    if not synced:
        return CheckResult("execution_oracle", "equity_synced", WARN, detail="post-restart equity not yet synced")
    return CheckResult("execution_oracle", "equity_synced", WARN, detail="synced but equity zero")
```

```python
# src/hanoon_prime/inspection/halim.py
"""halim joint check."""

from __future__ import annotations

from .checks import FAIL, OK, CheckResult
from .ctx import InspectionContext
from .probe import halim_probe


def halim_state_matches_clock(ctx: InspectionContext) -> CheckResult:
    st = halim_probe(ctx)
    if st == "down":
        return CheckResult("halim", "halim_state_matches_clock", FAIL, detail="down (degraded)")
    suffix = " (expected post-market)" if st == "asleep" else ""
    return CheckResult("halim", "halim_state_matches_clock", OK, detail=f"state={st}{suffix}")
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_inspection_memory_oracle.py -v`
Expected: PASS (11 passed). Run `.venv/bin/python -m mypy src/hanoon_prime/inspection` — no new errors.

- [ ] **Step 5: Commit**

Run: `git add src/hanoon_prime/inspection/journals.py src/hanoon_prime/inspection/oracle.py src/hanoon_prime/inspection/halim.py tests/test_inspection_memory_oracle.py && git commit -m "feat(inspection): memory/oracle/halim checks"`
Re-stage if pre-commit reformats.

---

### Task 7: joint registry + manifest + CLI

**Files:**
- Create: `src/hanoon_prime/inspection/joints.py`
- Create: `src/hanoon_prime/inspection/__main__.py`
- Modify: `src/hanoon_prime/inspection/__init__.py` (add exports)
- Test: `tests/test_inspection_joints.py` (registry + manifest + CLI)

**Interfaces:**
- Consumes: all check functions from Tasks 3–6.
- Produces: `JOINT_ORDER: list[str]`, `SPECS: tuple[CheckSpec, ...]`, `HARD_KEYS: frozenset[tuple[str, str]]`, `Manifest(ts, git_head, pid, results)` with properties `status`, `hard_fails`, `anomalies`, and `run_all(ctx) -> Manifest`. CLI: `python3 -m hanoon_prime.inspection manifest [--json]` (exit 0/2/4 per severity).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inspection_joints.py
"""joint registry, manifest aggregation, CLI smoke."""
import datetime

from hanoon_prime.inspection import joints
from hanoon_prime.inspection.checks import FAIL, OK
from hanoon_prime.inspection.ctx import InspectionContext


def test_spec_registry_no_duplicates() -> None:
    names = [(s.joint, s.name) for s in joints.SPECS]
    assert len(names) == len(set(names))
    assert len(joints.HARD_KEYS) == sum(1 for s in joints.SPECS if s.hard)
    assert len(joints.SPECS) >= 30


def test_full_manifest_empty_stack(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    m = joints.run_all(ctx)
    assert m.pid > 0
    assert len(m.results) == len(joints.SPECS)
    assert m.status == FAIL  # health down / process down => hard fail
    assert any(r.joint == "telemetry" and r.name == "health_ok" and r.status == FAIL for r in m.results)


def test_full_manifest_healthy_mock_stack(tmp_path, monkeypatch) -> None:
    import datetime as dt

    from hanoon_prime.memory import Journal
    from hanoon_prime.inspection import halim as halim_mod
    from hanoon_prime.inspection import runtime

    ctx = InspectionContext(base_dir=tmp_path, prev_journal_count=1)
    ctx.memo["health"] = {
        "status": "ok", "connected": True, "session": "post_market",
        "session_active": False, "position_count": 0, "positions": [],
    }
    ctx.memo["snapshot"] = {"health": {}}
    ctx.memo["runtime_state"] = {
        "brain_state": {
            "positions_open": 0,
            "threshold": 0.58, "pred_error": 0.4, "risk_ceiling": 1.0,
            "policy_state": {"equity": 100.0, "equity_synced": True,
                             "enabled": False, "authorized": True, "daily_pnl": 0.0},
        }
    }
    ctx.memo["juli_state"] = {
        "weights": {f"w{i}": 0.1 for i in range(20)},
        "episodes": [{"ticker": "NVDA", "vector": []}],
    }
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
    monkeypatch.setattr(joints.system, "_pidread", lambda c, s: (1234, True, ""))
    monkeypatch.setattr(joints.system, "_pgrep", lambda p: [1234])
    monkeypatch.setattr(joints.system, "_cmdline", lambda pid: "/x/.venv/bin/python -m hanoon_prime.cli")
    monkeypatch.setattr(joints.halim, "halim_probe", lambda c: "asleep")
    monkeypatch.setattr(joints.runtime, "_get_token", lambda: "t")
    monkeypatch.setattr(joints.runtime, "_get_chat_id", lambda: "c")
    m = joints.run_all(ctx)
    bad = [f"{r.joint}.{r.name}:{r.status}:{r.detail}" for r in m.results if r.status != OK]
    assert m.status == OK, bad


def test_manifest_hard_and_anomaly_partition(tmp_path) -> None:
    ctx = InspectionContext(base_dir=tmp_path)
    m = joints.run_all(ctx)
    assert all((r.joint, r.name) in joints.HARD_KEYS for r in m.hard_fails)
    assert all((r.joint, r.name) in {(s.joint, s.name) for s in joints.SPECS if s.report} for r in m.anomalies)


def test_cli_manifest_json(tmp_path, monkeypatch, capsys) -> None:
    from hanoon_prime.inspection import __main__ as cli
    monkeypatch.setattr(cli, "_ctx", lambda: InspectionContext(base_dir=tmp_path))
    rc = cli.main(["manifest", "--json"])
    out = capsys.readouterr().out
    assert '"status"' in out
    assert rc in (0, 2, 4)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_inspection_joints.py -v`
Expected: FAIL — `ModuleNotFoundError: hanoon_prime.inspection.joints`.

- [ ] **Step 3: Write the implementation**

```python
# src/hanoon_prime/inspection/joints.py
"""Joint registry — run_all produces the full manifest."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .checks import FAIL, OK, WARN, CheckResult, CheckSpec, MANIFEST_STATUS, run_check
from .ctx import InspectionContext
from .halim import halim_state_matches_clock
from .journals import chain_intact_from_anchor, journal_grows, seq_forward, verdicts_valid
from .oracle import closes_reconciled, enters_minted, equity_synced
from .purity import brain_fields_bounded, no_test_episodes, weights_finite_in_band
from .runtime import (
    cycle_flows_when_active,
    health_ok,
    heartbeat_fresh,
    positions_reconciled,
    positions_surface,
    send_healthy,
    sleep_is_expected,
    snapshot_fresh,
    state_matches_clock,
    telegram_configured,
)
from .safety import (
    drawdown_bound,
    no_learn_blocked,
    no_netting_guard,
    no_safety_halt,
    no_traceback,
    policy_flags,
)
from .system import (
    bot_alive,
    bot_from_trusted_checkout,
    cloudflared_alive,
    gateway_watchdog_alive,
    halim_alive,
    monitor_alive,
    single_bot,
)

JOINT_ORDER = [
    "processes", "identity", "telemetry", "pipeline", "session",
    "safety", "memory", "purity", "execution_oracle", "halim", "notify",
]

SPECS: tuple[CheckSpec, ...] = (
    CheckSpec("processes", "bot_alive", bot_alive, report=True),
    CheckSpec("processes", "halim_alive", halim_alive, report=True),
    CheckSpec("processes", "monitor_alive", monitor_alive, report=True),
    CheckSpec("processes", "cloudflared_alive", cloudflared_alive, report=True),
    CheckSpec("processes", "gateway_watchdog_alive", gateway_watchdog_alive, report=True),
    CheckSpec("processes", "single_bot", single_bot, report=True),
    CheckSpec("identity", "bot_from_trusted_checkout", bot_from_trusted_checkout, hard=True),
    CheckSpec("telemetry", "health_ok", health_ok, hard=True),
    CheckSpec("telemetry", "snapshot_fresh", snapshot_fresh, hard=True),
    CheckSpec("telemetry", "positions_surface", positions_surface, report=True),
    CheckSpec("pipeline", "heartbeat_fresh", heartbeat_fresh, hard=True),
    CheckSpec("pipeline", "cycle_flows_when_active", cycle_flows_when_active, hard=True),
    CheckSpec("pipeline", "sleep_is_expected", sleep_is_expected, report=True),
    CheckSpec("session", "state_matches_clock", state_matches_clock, report=True),
    CheckSpec("session", "positions_reconciled", positions_reconciled, report=True),
    CheckSpec("safety", "no_netting_guard", no_netting_guard, hard=True),
    CheckSpec("safety", "no_traceback", no_traceback, hard=True),
    CheckSpec("safety", "no_safety_halt", no_safety_halt, hard=True),
    CheckSpec("safety", "no_learn_blocked", no_learn_blocked, report=True),
    CheckSpec("safety", "policy_flags", policy_flags, report=True),
    CheckSpec("safety", "drawdown_bound", drawdown_bound, hard=True),
    CheckSpec("memory", "journal_grows", journal_grows, hard=True),
    CheckSpec("memory", "seq_forward", seq_forward, report=True),
    CheckSpec("memory", "verdicts_valid", verdicts_valid, hard=True),
    CheckSpec("memory", "chain_intact_from_anchor", chain_intact_from_anchor, hard=True),
    CheckSpec("purity", "no_test_episodes", no_test_episodes, hard=True),
    CheckSpec("purity", "weights_finite_in_band", weights_finite_in_band, hard=True),
    CheckSpec("purity", "brain_fields_bounded", brain_fields_bounded, hard=True),
    CheckSpec("execution_oracle", "enters_minted", enters_minted, report=True),
    CheckSpec("execution_oracle", "closes_reconciled", closes_reconciled, report=True),
    CheckSpec("execution_oracle", "equity_synced", equity_synced, report=True),
    CheckSpec("halim", "halim_state_matches_clock", halim_state_matches_clock, report=True),
    CheckSpec("notify", "telegram_configured", telegram_configured, report=True),
    CheckSpec("notify", "send_healthy", send_healthy, report=True),
)

HARD_KEYS: frozenset[tuple[str, str]] = frozenset((s.joint, s.name) for s in SPECS if s.hard)
REPORT_KEYS: frozenset[tuple[str, str]] = frozenset((s.joint, s.name) for s in SPECS if s.report)


@dataclass(frozen=True)
class Manifest:
    """One probe tick: full check results plus provenance."""

    ts: float
    git_head: str | None
    pid: int
    results: tuple[CheckResult, ...]

    @property
    def status(self) -> str:
        return MANIFEST_STATUS(list(self.results), HARD_KEYS)

    @property
    def hard_fails(self) -> list[CheckResult]:
        return [r for r in self.results if (r.joint, r.name) in HARD_KEYS and r.status == FAIL]

    @property
    def anomalies(self) -> list[CheckResult]:
        return [r for r in self.results if (r.joint, r.name) in REPORT_KEYS and r.status in (WARN, FAIL)]


def run_all(ctx: InspectionContext) -> Manifest:
    results = tuple(run_check(s, ctx) for s in SPECS)
    return Manifest(ts=time.time(), git_head=ctx.git_head(), pid=os.getpid(), results=results)
```

```python
# src/hanoon_prime/inspection/__main__.py
"""Inside Man command line.

Usage:
    python3 -m hanoon_prime.inspection manifest [--json]
    python3 -m hanoon_prime.inspection reanchor [--note TEXT]   (Task 9)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .ctx import InspectionContext
from .joints import JOINT_ORDER, run_all

ROOT = Path(__file__).resolve().parents[2]


def _ctx() -> InspectionContext:
    return InspectionContext(base_dir=ROOT)


def _print_manifest(args: argparse.Namespace) -> int:
    m = run_all(_ctx())
    if args.json:
        payload = asdict(m)
        payload["status"] = m.status
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"INSIDE-MAN MANIFEST  ts={m.ts:.0f}  head={m.git_head}")
        print(f"status: {m.status}\n")
        for joint in JOINT_ORDER:
            for r in [x for x in m.results if x.joint == joint]:
                print(f"  [{r.status:<11}] {r.joint}.{r.name}  {r.detail}")
        print()
        print("HARD FAILS:", [f"{r.joint}.{r.name}" for r in m.hard_fails] or "none")
        print("ANOMALIES :", [f"{r.joint}.{r.name}" for r in m.anomalies] or "none")
    if m.status == OK:
        return 0
    if m.hard_fails:
        return 2
    if m.anomalies:
        return 4
    return 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inside Man facility inspection")
    sub = ap.add_subparsers(dest="command", required=True)
    pm = sub.add_parser("manifest", help="print the full check manifest")
    pm.add_argument("--json", action="store_true")
    pm.set_defaults(func=_print_manifest)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
```

Update `__init__.py`:

```python
"""hanoon_prime.inspection — the Inside Man: joint-by-joint verification."""

from .checks import (
    FAIL,
    OK,
    UNVERIFIABLE,
    WARN,
    CheckResult,
    CheckSpec,
    MANIFEST_STATUS,
)
from .ctx import InspectionContext
from .joints import JOINT_ORDER, Manifest, SPECS, run_all

__all__ = [
    "FAIL",
    "OK",
    "UNVERIFIABLE",
    "WARN",
    "CheckResult",
    "CheckSpec",
    "InspectionContext",
    "JOINT_ORDER",
    "MANIFEST_STATUS",
    "Manifest",
    "SPECS",
    "run_all",
]
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_inspection_joints.py -v`
Expected: PASS (5 passed). Run `.venv/bin/python -m mypy src/hanoon_prime/inspection` — no new errors.

- [ ] **Step 5: Commit**

Run: `git add src/hanoon_prime/inspection/joints.py src/hanoon_prime/inspection/__main__.py src/hanoon_prime/inspection/__init__.py tests/test_inspection_joints.py && git commit -m "feat(inspection): joint registry + manifest + CLI"`
Re-stage if pre-commit reformats.

---

### Task 8: daily digest + gated auto-heal

**Files:**
- Create: `src/hanoon_prime/inspection/digest.py`
- Create: `src/hanoon_prime/inspection/heal.py`
- Modify: `src/hanoon_prime/inspection/__main__.py` (add `heal [--dry-run]` subcommand)
- Modify: `src/hanoon_prime/inspection/ctx.py` (add script path fields)
- Test: `tests/test_inspection_digest_heal.py`

**Interfaces:**
- digest: `build_digest(manifest) -> str`, `digest_send(ctx, manifest) -> tuple[bool, str]` (once per local day, ledger key `digest.latest_day`), `_chunks(text, size=4000)`.
- heal: `heal_enabled(ctx)`, `_ledger_day_count(ctx)`, `candidates(ctx, manifest) -> list[str]`, `record_heal(ctx, name)`, `heal(ctx, manifest, dry_run=False) -> list[str]`. Cabinet (max 2/day): `halim_alive` (→ `scripts/halim_start.sh`), `cloudflared_alive` (→ launch_detached with `/opt/homebrew/bin/cloudflared tunnel --config ~/.cloudflared/config.yml run`), `gateway_watchdog_alive` (→ launch_detached `scripts/ib_gateway_watchdog.py`). Bot is **not** healable.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inspection_digest_heal.py
"""digest emit + gated heal."""

from hanoon_prime.inspection import digest, heal, joints
from hanoon_prime.inspection.checks import FAIL, OK, CheckResult
from hanoon_prime.inspection.ctx import InspectionContext


def _manifest(*results: CheckResult) -> joints.Manifest:
    return joints.Manifest(ts=1_700_000_000.0, git_head="deadbeef", pid=1, results=tuple(results))


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
        CheckResult("telemetry", "health_ok", OK),
        CheckResult("halim", "halim_state_matches_clock", FAIL, detail="down"),
    )
    ok, msg = digest.digest_send(ctx, m)
    assert ok and msg == "sent"
    assert len(sent) == 1
    assert "status: FAIL" in sent[0]
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_inspection_digest_heal.py -v`
Expected: FAIL — `ModuleNotFoundError: hanoon_prime.inspection.digest`.

- [ ] **Step 3: Write the implementation**

Extend `ctx.py` with script path fields (defaults below). **First read** `src/hanoon_prime/_telegram.py` at implementation time and adapt `_notify` to the real `send` signature/return.

```python
# src/hanoon_prime/inspection/digest.py
"""Daily Telegram digest — one per local day, chunked to send limit."""

from __future__ import annotations

import time
from typing import Callable

from .._telegram import send
from .ctx import InspectionContext
from .joints import JOINT_ORDER, Manifest
from .checks import OK

SEV = {"OK": 0, "WARN": 1, "FAIL": 2, "UNVERIFIABLE": 3}


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def build_digest(manifest: Manifest) -> str:
    today = _day(manifest.ts)
    lines = [f"INSIDE-MAN digest {today}", f"status: {manifest.status}"]
    for joint in JOINT_ORDER:
        rows = [r for r in manifest.results if r.joint == joint]
        if not rows:
            continue
        worst = max(rows, key=lambda r: SEV.get(r.status, 3))
        if worst.status == OK:
            lines.append(f"  {joint}: ok")
        else:
            details = "; ".join(f"{r.name}={r.status}:{r.detail[:40]}" for r in rows if r.status != OK)
            lines.append(f"  {joint}: {details}")
    return "\n".join(lines)


def _chunks(text: str, size: int = 4000) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


def _notify(text: str, ctx: InspectionContext) -> bool:
    try:
        return send(text, ctx)  # adapt to real signature at implementation time
    except Exception:
        return False


def digest_send(ctx: InspectionContext, manifest: Manifest) -> tuple[bool, str]:
    ledger = ctx.ledger()
    today = _day(manifest.ts)
    if ledger.get("digest", {}).get("latest_day") == today:
        return False, "already sent today"
    text = build_digest(manifest)
    ok = all(_notify(part, ctx) for part in _chunks(text))
    if ok:
        ledger.setdefault("digest", {})["latest_day"] = today
        ctx.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.ledger_path.write_text(__import__("json").dumps(ledger, sort_keys=True, indent=2))
    return ok, "sent" if ok else "send failed"
```

```python
# src/hanoon_prime/inspection/heal.py
"""Gated auto-heal — mechanical, non-trading services only."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .ctx import InspectionContext
from .joints import Manifest

MAX_PER_DAY = 2

HEALABLE = ("halim_alive", "cloudflared_alive", "gateway_watchdog_alive")


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _today() -> str:
    return _day(time.time())


def heal_enabled(ctx: InspectionContext) -> bool:
    return ctx.heal_enabled


def _ledger_day_count(ctx: InspectionContext) -> int:
    h = ctx.ledger().get("heals", {})
    return h.get("count", 0) if h.get("today") == _today() else 0


def _save(ctx: InspectionContext, ledger: dict) -> None:
    ctx.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.ledger_path.write_text(json.dumps(ledger, sort_keys=True, indent=2))


def record_heal(ctx: InspectionContext, name: str) -> None:
    ledger = ctx.ledger()
    h = ledger.setdefault("heals", {})
    if h.get("today") != _today():
        h.update(today=_today(), count=0)
    h["count"] = h.get("count", 0) + 1
    h.setdefault("events", []).append({"name": name, "ts": time.time()})
    _save(ctx, ledger)


def candidates(ctx: InspectionContext, manifest: Manifest) -> list[str]:
    failing = {r.name for r in manifest.results if r.status == "FAIL"}
    return [n for n in HEALABLE if n in failing]


def _commands_for(name: str, ctx: InspectionContext) -> list[str]:
    home = str(Path.home())
    if name == "halim_alive":
        return [str(ctx.halim_start_script)]
    if name == "cloudflared_alive":
        return [sys.executable, str(ctx.launch_detached_script),
                "/opt/homebrew/bin/cloudflared", "tunnel",
                "--config", f"{home}/.cloudflared/config.yml", "run"]
    if name == "gateway_watchdog_alive":
        return [sys.executable, str(ctx.launch_detached_script), str(ctx.watchdog_script)]
    raise ValueError(name)


def _run(cmd: list[str], dry_run: bool) -> str:
    if dry_run:
        return "would run: " + " ".join(cmd)
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return f"ran: {' '.join(cmd)} rc={cp.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"failed: {exc}"


def heal(ctx: InspectionContext, manifest: Manifest, dry_run: bool = False) -> list[str]:
    if not heal_enabled(ctx):
        return ["heal disabled (HANOON_HEAL=0)"]
    if _ledger_day_count(ctx) >= MAX_PER_DAY:
        return [f"heal budget exhausted ({MAX_PER_DAY}/day)"]
    out: list[str] = []
    for name in candidates(ctx, manifest):
        out.append(f"{name}: {_run(_commands_for(name, ctx), dry_run)}")
        if not dry_run:
            record_heal(ctx, name)
    return out or ["nothing to heal"]
```

Extend `ctx.py` fields:

```python
    halim_start_script: Path = dataclasses.field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "scripts" / "halim_start.sh")
    launch_detached_script: Path = dataclasses.field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "scripts" / "launch_detached.py")
    watchdog_script: Path = dataclasses.field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "scripts" / "ib_gateway_watchdog.py")
```

Extend `__main__.py` with the `heal` subcommand (add to the existing file from Task 7):

```python
def _print_heal(args: argparse.Namespace) -> int:
    from .heal import heal as heal_actions
    manifest = run_all(_ctx())
    for line in heal_actions(_ctx(), manifest, dry_run=args.dry_run):
        print(line)
    return 0


# inside main(argv): after the manifest subcommand
    ph = sub.add_parser("heal", help="gated auto-heal of mechanical services")
    ph.add_argument("--dry-run", action="store_true", help="show candidate actions, take none")
    ph.set_defaults(func=_print_heal)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_inspection_digest_heal.py -v`
Expected: PASS (6 passed). Run `.venv/bin/python -m mypy src/hanoon_prime/inspection` — no new errors.

- [ ] **Step 5: Commit**

Run: `git add src/hanoon_prime/inspection/digest.py src/hanoon_prime/inspection/heal.py tests/test_inspection_digest_heal.py`
Re-stage if pre-commit reformats. Commit `feat(inspection): digest + gated auto-heal`.

---

### Task 9: journal chain re-anchor boot step

**Files:**
- Create: `src/hanoon_prime/inspection/reanchor.py`
- Modify: `scripts/start.command` (re-anchor between stop and bot start)
- Modify: `src/hanoon_prime/inspection/__main__.py` (add `reanchor` subcommand)
- Test: `tests/test_inspection_reanchor.py`

**Interfaces:**
- `reanchor_if_broken(ctx) -> tuple[bool, str]`: if `Journal(ctx.journal_path).verify_chain()` reports a break, append `{"event": "chain_reseed", "ts": ..., "reason": "hash_break"}` and return `(True, ...)`; else `(False, "chain intact")`. The `chain_reseed` record becomes the new anchor (memory's `_seed` sliding-window). Returns `(False, "chain intact")` on a fresh/empty journal too.
- `__main__` subcommand `reanchor` prints the outcome and exits 0.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inspection_reanchor.py
"""journal chain re-anchor boot step."""
import json

from hanoon_prime.inspection import reanchor
from hanoon_prime.inspection.ctx import InspectionContext
from hanoon_prime.memory import Journal


def _write_journal(tmp_path, rows: list[dict]) -> None:
    jp = tmp_path / "runtime" / "journal_live.jsonl"
    jp.parent.mkdir(parents=True)
    jp.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _good_ctx(tmp_path, n: int = 3) -> InspectionContext:
    rows = [{"event": "verdict", "action": "HOLD", "score": 0.5, "ticker": "NVDA"} for _ in range(n)]
    _write_journal(tmp_path, rows)
    return InspectionContext(base_dir=tmp_path)


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


def test_cli_reanchor(tmp_path, monkeypatch, capsys) -> None:
    from hanoon_prime.inspection import __main__ as cli
    monkeypatch.setattr(cli, "_ctx", lambda: InspectionContext(base_dir=tmp_path))
    rc = cli.main(["reanchor"])
    assert rc == 0
    assert "intact" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_inspection_reanchor.py -v`
Expected: FAIL — `ModuleNotFoundError: hanoon_prime.inspection.reanchor`.

- [ ] **Step 3: Write the implementation**

```python
# src/hanoon_prime/inspection/reanchor.py
"""Re-anchor the journal chain after an authorized stop (start.command step)."""

from __future__ import annotations

import time

from ..memory import Journal
from .ctx import InspectionContext


def reanchor_if_broken(ctx: InspectionContext) -> tuple[bool, str]:
    j = Journal(ctx.journal_path)
    try:
        broken = j.verify_chain()
    except Exception as exc:  # never raise through the boot step
        return False, f"verify failed: {exc}"
    if broken:
        j.append({"event": "chain_reseed", "ts": time.time(), "reason": "hash_break"})
        return True, "chain re-anchored (hash_break)"
    return False, "chain intact"
```

Extend `__main__.py`:

```python
def _print_reanchor(args: argparse.Namespace) -> int:
    from .reanchor import reanchor_if_broken
    changed, outcome = reanchor_if_broken(_ctx())
    print(outcome)
    return 0


# inside main(argv): after the heal subcommand
    pr = sub.add_parser("reanchor", help="re-anchor journal chain if broken (boot step)")
    pr.set_defaults(func=_print_reanchor)
```

`scripts/start.command` — insert after the bot **stop** step and before the bot **start** step:

```bash
echo "== re-anchoring journal chain =="
.venv/bin/python -m hanoon_prime.inspection reanchor >> logs/reanchor.log 2>&1 || echo "warn: re-anchor step failed (rc $?)"
```

(The chain may only be re-anchored during stop→start, per the spec: never while the bot is live.)

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_inspection_reanchor.py -v`
Expected: PASS (3 passed). Run `.venv/bin/python -m mypy src/hanoon_prime/inspection` — no new errors.

- [ ] **Step 5: Commit**

Run: `git add src/hanoon_prime/inspection/reanchor.py scripts/start.command tests/test_inspection_reanchor.py`
Re-stage if pre-commit reformats. Commit `feat(inspection): journal chain re-anchor boot step`.

---

### Task 10: production_monitor.py refactor

**Files:**
- Modify: `scripts/production_monitor.py`
- Test: `tests/test_monitor_refactor.py`

**Goal:** Replace the monitor's bespoke per-check code with `hanoon_prime.inspection.run_all` while preserving its contract: `_collect()` HALIM metrics feed, `_halim_probe()` ok/asleep/down, ledger/streak/gate1, per-day dedupe (`_alert(day, key, text)`), and **exit codes 0 (PASS) / 2 (hard FAIL, streak reset) / 3 (HALIM down) / 4 (anomalies)**.

Extract a pure decision function (importable by tests via `importlib`):

```python
def _decide(manifest: Manifest) -> int:
    """0 PASS. 2 hard FAIL. 3 HALIM down. 4 FAIL-grade anomalies."""
    if any(r.joint == "halim" and r.status == "FAIL" for r in manifest.results):
        return 3
    if manifest.hard_fails:
        return 2
    if any(r.status == "FAIL" for r in manifest.anomalies):
        return 4
    return 0
```

Exit 4 only on **FAIL**-grade anomalies. WARN-grade anomalies (pre-first-sync `equity_synced`, unminted `enters_minted`, a quiet `send_healthy`) are recorded in the ledger and shown in the next digest but do **not** page — this preserves the old v2 behavior that never alerted on pre-sync/idle baselines.

Loop body becomes: `m = run_all(_ctx())` → `rc = _decide(m)` → on non-zero, build alert text from `m.hard_fails`/`m.anomalies` names `+ ":" + detail`, `_alert(day, key, text)`, update streak/ledger, `sys.exit(rc)`; on 0, scheduled `send_healthy` (freshness-gated). A "down" HALIM mapping to exit 3 preserves the old telemetry-behavior where HALIM death dominates.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_monitor_refactor.py
"""production_monitor decision contract after Inspection refactor."""
import importlib.util
from pathlib import Path

from hanoon_prime.inspection.checks import FAIL, OK, WARN, CheckResult
from hanoon_prime.inspection.joints import Manifest


def _load_monitor():
    p = Path(__file__).resolve().parents[1] / "scripts" / "production_monitor.py"
    spec = importlib.util.spec_from_file_location("monitor_refactor_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mf(*results: CheckResult) -> Manifest:
    return Manifest(ts=1.0, git_head="h", pid=1, results=tuple(results))


def test_decide_pass() -> None:
    m = _load_monitor()
    assert m._decide(_mf(CheckResult("telemetry", "health_ok", OK))) == 0


def test_decide_halim_dominates() -> None:
    m = _load_monitor()
    results = [
        CheckResult("halim", "halim_state_matches_clock", FAIL, detail="down"),
        CheckResult("telemetry", "health_ok", FAIL, detail="unreachable"),
    ]
    assert m._decide(_mf(*results)) == 3


def test_decide_hard_fail() -> None:
    m = _load_monitor()
    results = [
        CheckResult("telemetry", "health_ok", FAIL, detail="unreachable"),
        CheckResult("halim", "halim_state_matches_clock", OK, detail="state=ok"),
    ]
    assert m._decide(_mf(*results)) == 2


def test_decide_anomaly_only() -> None:
    m = _load_monitor()
    results = [
        CheckResult("processes", "bot_alive", FAIL, detail="not running"),
        CheckResult("halim", "halim_state_matches_clock", OK, detail="state=asleep"),
        CheckResult("identity", "bot_from_trusted_checkout", OK),
    ]
    assert m._decide(_mf(*results)) == 4


def test_decide_warn_is_pass() -> None:
    m = _load_monitor()
    results = [
        CheckResult("execution_oracle", "enters_minted", WARN, detail="1 ENTER without fill"),
        CheckResult("halim", "halim_state_matches_clock", OK, detail="state=asleep"),
        CheckResult("identity", "bot_from_trusted_checkout", OK),
    ]
    assert m._decide(_mf(*results)) == 0  # WARN-grade anomalies do not page
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/test_monitor_refactor.py -v`
Expected: FAIL — monitor has no `_decide` (AttributeError).

- [ ] **Step 3: Write the implementation**

Refactor `scripts/production_monitor.py`:

1. Add imports: `from hanoon_prime.inspection.ctx import InspectionContext`, `from hanoon_prime.inspection.joints import Manifest, run_all` (module already bootstraps `sys.path` with the repo `src`).
2. Add `_ctx()` building `InspectionContext(base_dir=ROOT, telemetry_url=TELEMETRY_URL, halim_url=HALIM_URL)` with `heal_enabled=False` (the daemon reports; it never acts on its own).
3. Add `_decide(manifest)` exactly as above.
4. Replace the bespoke per-check block in the polling loop with:

```python
m = run_all(_ctx())
rc = _decide(m)
if rc == 3:
    _alert(_today(), "halim_down", "INSIDE-MAN: HALIM down " + next(r.detail for r in m.results if r.joint == "halim" and r.status == "FAIL"))
    _gate_or_exit(3)
elif rc == 2:
    fails = " | ".join(f"{r.joint}.{r.name}: {r.detail}" for r in m.hard_fails)
    _alert(_today(), "hard_fail", "INSIDE-MAN hard: " + fails)
    _gate_or_exit(2)
elif rc == 4:
    anom = " | ".join(f"{r.joint}.{r.name}: {r.detail}" for r in m.anomalies if r.status == "FAIL")
    _alert(_today(), "anomalies", "INSIDE-MAN anomalies: " + anom)
    _gate_or_exit(4)
else:
    _healthy_path(m)
```

(`_gate_or_exit` and `_healthy_path` are **adapters to the monitor's existing helper names** — map them to the actual functions already in the file (gate1/streak/ledger bookkeeping and `_collect()`/`send_healthy` freshness path). WARN anomalies reach `_healthy_path` and are folded into the digest text so they are recorded without paging.)

5. Keep `_collect()` and the graceful `SIGINT`/`SIGTERM` handler as-is. Do not introduce `print()`.

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_monitor_refactor.py -v` and the pre-existing monitor tests (e.g. `tests/test_production_monitor*.py`) — all green. Run `.venv/bin/python -m mypy src/hanoon_prime` (monitor lives outside `src`; run `python -m py_compile scripts/production_monitor.py`).

- [ ] **Step 5: Commit**

Run: `git add scripts/production_monitor.py tests/test_monitor_refactor.py`
Re-stage if pre-commit reformats. Commit `refactor(monitor): derive alerts from inspection run_all`.

---

### Task 11: docs, full verification, rollout smoke

**Files:**
- Modify: `PRODUCTION_READINESS.md`

- [ ] **Step 1: Document the Inside Man**

Add a "Inside Man — facility inspection" section under Gate 3:
- `python3 -m hanoon_prime.inspection manifest [--json]` (or `--json | jq '.'`) — all joints, one command.
- Exit codes 0/2/3/4 and severity semantics (OK/WARN/FAIL/UNVERIFIABLE; manifest FAIL = any hard FAIL).
- Heal cabinet table: `halim_alive` → `halim_start.sh`, `cloudflared_alive` → `cloudflared tunnel` (launch_detached), `gateway_watchdog_alive` → `ib_gateway_watchdog.py`; gated to 2/day, default-on via `HANOON_HEAL` (0 disables), all actions logged to the ledger (`runtime/inspection_ledger.json`).
- Digest: one Telegram message/day via `digest_send`; freshness-gated healthy heartbeat.
- Boot re-anchor: `start.command` runs `reanchor` between stop and start; chain history before an intentional purge is expected to show breaks, and `chain_intact_from_anchor` verifies clean from the last `chain_reseed`/purge anchor.
- Note: `runtime/inspection_ledger.json` and `logs/reanchor.log` are runtime artifacts (gitignored).

- [ ] **Step 2: Full verification**

Run: `.venv/bin/python -m pytest -q` (all suites) and `.venv/bin/python -m mypy src/hanoon_prime`. Then `pre-commit run --all-files` (re-stage reformats; do not stage `.planning/`).

- [ ] **Step 3: Commit + push**

Run: `git add PRODUCTION_READINESS.md` then commit `docs: Inside Man inspection guide`. Push **both** remotes: `git push origin && git push origin-sajib`.

- [ ] **Step 4: Lifecycle rollout smoke (user, at a market-safe time)**

1. `sh scripts/start.command` (bot, HALIM, monitor, watchdog all come up; re-anchor runs in the quiet window).
2. `python3 -m hanoon_prime.inspection manifest --json` — expected first-tick read:
   - Overnight/post-market: `halim_state_matches_clock` OK (`asleep — expected post-market`).
   - `chain_intact_from_anchor` OK, anchored at the purge break (seq 94214) with 0 gaps after.
   - `equity_synced` WARN until the first IB equity sync after restart.
   - `send_healthy` WARN until the first send of the day.
   - Everything else OK.
3. `python3 -m hanoon_prime.inspection heal --dry-run` prints no candidates when the stack is healthy.
4. Confirm the daily digest arrives once and the daemon keeps publishing HEARTBEAT lines to `logs/ib_cycle.log`.

---

## Definition of Done

- [ ] **Task 1** — `checks.py` model + harness + `MANIFEST_STATUS` (tests: `test_inspection_checks.py`).
- [ ] **Task 2** — `ctx.py` (`InspectionContext`, memo, ledger, paths, env, `git_head`, `session_lines`) + `probe.py` + `__init__` re-exports (tests: `test_inspection_ctx.py`).
- [ ] **Task 3** — `system.py`: processes + identity checks (tests: `test_inspection_system.py`).
- [ ] **Task 4** — `runtime.py`: telemetry + pipeline + session + notify checks (tests: `test_inspection_runtime.py`).
- [ ] **Task 5** — `safety.py` + `purity.py` (tests: `test_inspection_safety_purity.py`).
- [ ] **Task 6** — `journals.py` + `oracle.py` + `halim.py` (tests: `test_inspection_memory_oracle.py`).
- [ ] **Task 7** — `joints.py` registry + `Manifest` + `run_all` + CLI `manifest [--json]` (tests: `test_inspection_joints.py`).
- [ ] **Task 8** — `digest.py` + `heal.py` + CLI `heal [--dry-run]` (tests: `test_inspection_digest_heal.py`).
- [ ] **Task 9** — `reanchor.py` + `start.command` boot step + CLI `reanchor` (tests: `test_inspection_reanchor.py`).
- [ ] **Task 10** — `production_monitor.py` refactor to `run_all` + `_decide` (tests: `test_monitor_refactor.py` + existing monitor tests).
- [ ] **Task 11** — `PRODUCTION_READINESS.md` Inside Man section; full suite + mypy + pre-commit green; both remotes pushed; lifecycle smoke read confirmed.

**Exit criteria:** `python3 -m pytest -q` and `.venv/bin/python -m mypy src/hanoon_prime` both succeed; a `manifest --json` reads `"status": "OK"` (or a WARN baseline: `equity_synced`/`send_healthy` pre-first-sync) on a live stack; the digest fires once/day; heal budget and dry-run behave as specified; monitor exits 0 on a healthy tick and 3 on HALIM-down.