"""Live pokers: health, snapshot, state, and the log surface.

Each function takes an InspectionContext and is memoized per instance so a
tick touches each source at most once. Journal-chain and halim probes live
in probe_chain.py / probe_halim.py and are re-exported here to keep this
file under the 200-line src limit.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .ctx import InspectionContext
from .probe_chain import journal as journal
from .probe_chain import journal_chain_state as journal_chain_state
from .probe_chain import journal_tail as journal_tail
from .probe_halim import halim_probe as halim_probe

START_MARKER = re.compile(r"ib_adapter\s+Starting \(seed=")
HEARTBEAT_MARKER = re.compile(r"ib_cycle\s+HEARTBEAT")
CYCLE_MARKER = re.compile(r"ib_cycle\s+CYCLE ")
CYCLE_BARS_RE = re.compile(r"ib_cycle\s+CYCLE\s+bars=(\d+)")
# Live bot logs verdict summaries via the `juli` logger as `EVAL TICKER:ACTION...`
# (NOT `ib_cycle EVAL`); match the token generically so a logger rename can't
# silently blind the veto guard again.
EVAL_MARKER = re.compile(r"EVAL\s+[A-Z]")
SLEEP_MARKER = re.compile(r"SESSION SLEEP")
GUARD_MARKER = re.compile(r"NETTING GUARD")
TRACE_MARKER = re.compile(r"Traceback \(most recent call last\)")
SAFETY_HALT_MARKER = re.compile(r"SAFETY HALT")
LEARN_BLOCKED_MARKER = re.compile(r"LEARN BLOCKED")
_TS_RE = re.compile(r"(\d\d):(\d\d):(\d\d)\.\d+")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _http_get(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:  # instrument failure ≠ verdict
        return {"_unreachable": str(exc)}
    return data if isinstance(data, dict) else {}


def _surface(
    ctx: InspectionContext, key: str, path: Path | None = None, url: str | None = None
) -> dict[str, Any]:
    """Fetch JSON from path or url once per tick; {} on miss."""
    if key in ctx.memo:
        value = ctx.memo[key]
        return value if isinstance(value, dict) else {}
    if path is not None:
        value = _read_json(path)
    else:
        value = _http_get(url if url is not None else "")
    ctx.memo[key] = value
    return value if isinstance(value, dict) else {}


def _log_lines(ctx: InspectionContext) -> list[str]:
    """Full log file text, memoized."""
    if "log_lines" not in ctx.memo:
        try:
            with open(ctx.log_path, encoding="utf-8", errors="ignore") as fh:
                ctx.memo["log_lines"] = fh.readlines()
        except OSError:
            ctx.memo["log_lines"] = []
    value = ctx.memo["log_lines"]
    return list(value) if isinstance(value, list) else []


def health(ctx: InspectionContext) -> dict[str, Any]:
    """Live telemetry /health payload."""
    return _surface(ctx, "health", url=f"{ctx.telemetry_url}/health")


def snapshot(ctx: InspectionContext) -> dict[str, Any]:
    """Live telemetry /snapshot payload."""
    return _surface(ctx, "snapshot", url=f"{ctx.telemetry_url}/snapshot")


def account(ctx: InspectionContext) -> dict[str, Any]:
    """Live telemetry account feed (equity, positions_open, summary)."""
    return _surface(ctx, "account", url=f"{ctx.telemetry_url}/account")


def positions(ctx: InspectionContext) -> dict[str, Any]:
    """Live telemetry /positions payload (per-position live marks)."""
    return _surface(ctx, "positions", url=f"{ctx.telemetry_url}/positions")


def recent_cycle_bars(ctx: InspectionContext, window: int = 120) -> list[int]:
    """Closed-bar counts from the most recent CYCLE log lines."""
    counts: list[int] = []
    for line in _log_lines(ctx):
        match = CYCLE_BARS_RE.search(line)
        if match:
            counts.append(int(match.group(1)))
            if len(counts) > window:
                counts.pop(0)
    return counts[-window:]


def runtime_state(ctx: InspectionContext) -> dict[str, Any]:
    """Persisted brain policy state."""
    return _surface(ctx, "runtime_state", path=ctx.state_path)


def juli_state(ctx: InspectionContext) -> dict[str, Any]:
    """Persisted juli learning state."""
    return _surface(ctx, "juli_state", path=ctx.juli_state_path)


def regime_weights(ctx: InspectionContext) -> dict[str, Any]:
    """Persisted juli per-regime weight vectors."""
    return _surface(ctx, "regime_weights", path=ctx.regime_weights_path)


def ledger(ctx: InspectionContext) -> dict[str, Any]:
    """Guardian production-state ledger."""
    return _surface(ctx, "ledger", path=ctx.state_file)


def session_lines(ctx: InspectionContext) -> list[str]:
    """Every log line after the LAST 'ib_adapter Starting' marker."""
    key = "session_lines"
    if key in ctx.memo:
        value = ctx.memo[key]
        return list(value) if isinstance(value, list) else []
    session: list[str] = []
    for line in _log_lines(ctx):
        if START_MARKER.search(line):
            session = []
        else:
            session.append(line)
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


def last_line_age(ctx: InspectionContext, marker: re.Pattern[str]) -> float | None:
    """Age in seconds of the most recent log line matching marker."""
    latest: float | None = None
    for line in _log_lines(ctx):
        if marker.search(line):
            t = line_ts(line)
            if t is not None:
                latest = t
    return (time.time() - latest) if latest is not None else None
