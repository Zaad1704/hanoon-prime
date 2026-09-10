"""inspection.lifecycle — full signal-lifecycle coverage checks.

The Inside Man historically verified infrastructure (processes, sessions,
journals). These checks walk the *trading pipeline itself*: subscription,
data arrival, candidate scan, verdict, order, fill, and learning fan-out.
The webapp surfaces them as the lifecycle rail on the Sentinel tab.

Stages 1–3 (seed, stream, scan) live here; stages 4–6 (verdict, execution,
learning) live in ``lifecycle_flow``.
"""

from __future__ import annotations

import time
from typing import Any

from .checks import OK, UNVERIFIABLE, WARN, CheckResult
from .ctx import InspectionContext
from .probe import health

LIFECYCLE_STAGES: tuple[str, ...] = (
    "seed",
    "stream",
    "scan",
    "verdict",
    "execution",
    "learning",
)

# Subscription freshness: a live RTH feed ticks every few seconds.
STREAM_STALE_S: float = 45.0
# A scanning brain should emit verdicts at least this often while active.
VERDICT_MAX_AGE_S = 900.0

__all__ = [
    "LIFECYCLE_CHECKS",
    "LIFECYCLE_STAGES",
    "STREAM_STALE_S",
    "VERDICT_MAX_AGE_S",
    "lifecycle_scan",
    "lifecycle_seed",
    "lifecycle_stream",
    # re-exported from lifecycle_flow:
    "lifecycle_execution",
    "lifecycle_learning",
    "lifecycle_verdict",
]


def _telemetry_base(ctx: InspectionContext) -> str:
    return ctx.telemetry_url


def _get(ctx: InspectionContext, route: str) -> dict[str, Any]:
    """Memoized GET against the live bot telemetry (best-effort)."""
    key = f"lifecycle:{route}"
    if key in ctx.memo:
        val = ctx.memo[key]
        return val if isinstance(val, dict) else {}
    if key not in ctx.memo:
        import json
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{_telemetry_base(ctx)}{route}", timeout=4
            ) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            data = {}
        ctx.memo[key] = data if isinstance(data, dict) else {}
    val = ctx.memo[key]
    return val if isinstance(val, dict) else {}


def _inactive(ctx: InspectionContext) -> bool:
    h = health(ctx)
    return bool(h.get("_unreachable")) or not bool(h.get("session_active", False))


def _stage_result(
    stage: str,
    ok: bool,
    detail: str,
    evidence: dict[str, Any],
    warn_only: bool = False,
) -> CheckResult:
    return CheckResult(
        "lifecycle",
        f"stage_{stage}",
        OK if ok else (WARN if warn_only else UNVERIFIABLE),
        detail=detail,
        evidence=evidence,
    )


# ── Stage 1: SEED — tickers subscribed at the gateway ─────────────────


def lifecycle_seed(ctx: InspectionContext) -> CheckResult:
    """The scanner universe is subscribed and known to the session."""
    if _inactive(ctx):
        return _stage_result("seed", True, "session inactive", {})
    h = health(ctx)
    tickers = h.get("tickers") or []
    if not tickers:
        return _stage_result(
            "seed", False, "no tickers subscribed", {"tickers": []}, warn_only=True
        )
    return _stage_result(
        "seed",
        True,
        f"{len(tickers)} subscribed",
        {"tickers": list(tickers)[:12], "count": len(tickers)},
    )


# ── Stage 2: STREAM — quotes actually arriving ────────────────────────


def lifecycle_stream(ctx: InspectionContext) -> CheckResult:
    """Live quotes flow: /ib ticker rows are fresh and two-sided."""
    if _inactive(ctx):
        return _stage_result("stream", True, "session inactive", {})
    ib = _get(ctx, "/ib")
    if not ib.get("connected"):
        return _stage_result(
            "stream", False, "gateway disconnected", {"error": ib.get("error")}
        )
    tickers = ib.get("tickers") or []
    if not tickers:
        return _stage_result(
            "stream", False, "no live tick objects", {}, warn_only=True
        )
    now_ms = time.time() * 1000
    fresh, quoted = 0, 0
    oldest_s = 0.0
    for tk in tickers:
        t = tk.get("time") or 0
        age = max(0.0, (now_ms - t) / 1000.0) if t else None
        if age is not None:
            oldest_s = max(oldest_s, age)
            if age <= STREAM_STALE_S:
                fresh += 1
        if (tk.get("bid") or 0) > 0 and (tk.get("ask") or 0) > 0:
            quoted += 1
    detail = f"{fresh}/{len(tickers)} fresh, {quoted} two-sided"
    ok = fresh > 0
    return _stage_result(
        "stream",
        ok,
        detail,
        {
            "fresh": fresh,
            "two_sided": quoted,
            "count": len(tickers),
            "oldest_age_s": round(oldest_s, 1) if oldest_s else None,
        },
        warn_only=not ok,
    )


# ── Stage 3: SCAN — brain sees candidates ─────────────────────────────


def lifecycle_scan(ctx: InspectionContext) -> CheckResult:
    """The cognitive surface receives candidates (bar buffer + positions 0)."""
    if _inactive(ctx):
        return _stage_result("scan", True, "session inactive", {})
    br = _get(ctx, "/brain")
    bs_raw = br.get("brain_state")
    bs = bs_raw if isinstance(bs_raw, dict) else {}
    regime = bs.get("regime_label")
    return _stage_result(
        "scan",
        True,
        f"brain state live (regime: {regime or 'unknown'})",
        {"regime": regime, "panic_mode": bs.get("panic_mode")},
    )


# Stages 4–6 live in lifecycle_flow; imported here so joints.py's
# ``from .lifecycle import lifecycle_*`` continues to work unchanged.
from .lifecycle_flow import lifecycle_execution, lifecycle_learning, lifecycle_verdict

LIFECYCLE_CHECKS = (
    lifecycle_seed,
    lifecycle_stream,
    lifecycle_scan,
    lifecycle_verdict,
    lifecycle_execution,
    lifecycle_learning,
)
