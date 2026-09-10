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


def _unreachable(ctx: InspectionContext, joint: str, name: str) -> CheckResult | None:
    """Flag a probe instrument failure as UNVERIFIABLE."""
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult(joint, name, UNVERIFIABLE, detail=str(h["_unreachable"]))
    return None


def health_ok(ctx: InspectionContext) -> CheckResult:
    """Telemetry reports the bot connected and healthy."""
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult("telemetry", "health_ok", FAIL, detail="health unreachable")
    ok = h.get("status") == "ok" and bool(h.get("connected", False))
    return CheckResult(
        "telemetry", "health_ok", OK if ok else FAIL, detail=f"status={h.get('status')}"
    )


def snapshot_fresh(ctx: InspectionContext) -> CheckResult:
    """Telemetry snapshot endpoint serves a health block."""
    s = snapshot(ctx)
    if s.get("_unreachable"):
        return CheckResult(
            "telemetry", "snapshot_fresh", FAIL, detail="snapshot unreachable"
        )
    good = bool(s) and isinstance(s.get("health"), dict)
    return CheckResult(
        "telemetry",
        "snapshot_fresh",
        OK if good else FAIL,
        detail="ok" if good else "empty",
    )


def positions_surface(ctx: InspectionContext) -> CheckResult:
    """/health exposes position_count for reconciliation."""
    bad = _unreachable(ctx, "telemetry", "positions_surface")
    if bad:
        return bad
    h = health(ctx)
    if "position_count" in h:
        return CheckResult(
            "telemetry",
            "positions_surface",
            OK,
            evidence={"position_count": h.get("position_count")},
        )
    return CheckResult(
        "telemetry", "positions_surface", WARN, detail="no position_count"
    )


def heartbeat_fresh(ctx: InspectionContext) -> CheckResult:
    """ib_cycle published a HEARTBEAT recently."""
    age = last_line_age(ctx, HEARTBEAT_MARKER)
    if age is None:
        return CheckResult("pipeline", "heartbeat_fresh", FAIL, detail="no HEARTBEAT")
    ok = age <= CYCLE_STALE_SEC
    return CheckResult(
        "pipeline", "heartbeat_fresh", OK if ok else FAIL, detail=f"{age:.0f}s ago"
    )


def cycle_flows_when_active(ctx: InspectionContext) -> CheckResult:
    """An active session must be producing CYCLE lines."""
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult(
            "pipeline", "cycle_flows_when_active", FAIL, detail="health unreachable"
        )
    if not h.get("session_active"):
        return CheckResult("pipeline", "cycle_flows_when_active", OK, detail="idle")
    age = last_line_age(ctx, CYCLE_MARKER)
    if age is None:
        return CheckResult(
            "pipeline", "cycle_flows_when_active", FAIL, detail="no CYCLE"
        )
    ok = age <= CYCLE_STALE_SEC
    return CheckResult(
        "pipeline",
        "cycle_flows_when_active",
        OK if ok else FAIL,
        detail=f"{age:.0f}s ago",
    )


def sleep_is_expected(ctx: InspectionContext) -> CheckResult:
    """Idle sessions must declare SESSION SLEEP."""
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult(
            "pipeline", "sleep_is_expected", FAIL, detail="health unreachable"
        )
    active = bool(h.get("session_active", False))
    if active:
        return CheckResult("pipeline", "sleep_is_expected", OK, detail="session active")
    sleep_lines = [ln for ln in session_lines(ctx) if SLEEP_MARKER.search(ln)]
    return CheckResult(
        "pipeline",
        "sleep_is_expected",
        OK if sleep_lines else WARN,
        detail="asleep" if sleep_lines else "inactive, no marker",
    )


def state_matches_clock(ctx: InspectionContext) -> CheckResult:
    """Telemetry session label is one we recognize."""
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult(
            "session", "state_matches_clock", FAIL, detail="health unreachable"
        )
    raw = str(h.get("session", "")).split(" ")[0].lower()
    ok = raw in KNOWN_SESSIONS
    return CheckResult(
        "session",
        "state_matches_clock",
        OK if ok else WARN,
        detail=f"session={h.get('session')}",
    )


def positions_reconciled(ctx: InspectionContext) -> CheckResult:
    """IB position_count agrees with the brain's positions_open while active."""
    h = health(ctx)
    if h.get("_unreachable"):
        return CheckResult(
            "session", "positions_reconciled", FAIL, detail="health unreachable"
        )
    ib_count = int(h.get("position_count", 0) or 0)
    po = runtime_state(ctx).get("brain_state", {}).get("positions_open")
    brain_count = po if isinstance(po, int) else -1
    if ib_count == brain_count:
        return CheckResult(
            "session",
            "positions_reconciled",
            OK,
            evidence={"ib": ib_count, "brain": brain_count},
        )
    if not h.get("session_active"):
        return CheckResult(
            "session",
            "positions_reconciled",
            OK,
            detail=f"{ib_count} residual positions while session inactive",
        )
    return CheckResult(
        "session",
        "positions_reconciled",
        FAIL,
        detail=f"IB {ib_count} vs bot {brain_count} while session active",
        evidence={"ib": ib_count, "brain": brain_count},
    )


def telegram_configured(ctx: InspectionContext) -> CheckResult:
    """Telegram token and chat id are present."""
    token, chat = _get_token(), _get_chat_id()
    ok = bool(token and chat)
    return CheckResult(
        "notify",
        "telegram_configured",
        OK if ok else WARN,
        detail="configured" if ok else "missing",
    )


def send_healthy(ctx: InspectionContext) -> CheckResult:
    """A healthy heartbeat reached Telegram within 24h."""
    notify = ledger(ctx).get("notify")
    last_ok = float(notify.get("last_ok", 0.0)) if isinstance(notify, dict) else 0.0
    fresh = last_ok > 0 and (time.time() - last_ok) < 86400.0
    detail = f"sent {int(time.time() - last_ok)}s ago" if last_ok else "no send yet"
    return CheckResult("notify", "send_healthy", OK if fresh else WARN, detail=detail)
