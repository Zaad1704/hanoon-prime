"""probe_halim — HALIM liveness probe.

Split from probe.py to respect the 200-line src-file rule. probe.py
re-exports halim_probe.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .ctx import InspectionContext


def _extract_reason(text: str) -> str | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    reason = data.get("reason")
    return reason if isinstance(reason, str) else None


def _probe_once(ctx: InspectionContext) -> str:
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
    data: Any = {}
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        reason = _extract_reason(exc.read().decode(errors="replace"))
        return "asleep" if reason == "system_asleep" else "down"
    except Exception:  # instrument failure ≠ verdict
        return "down"
    res = data if isinstance(data, dict) else {}
    if res.get("ok"):
        return "ok"
    if res.get("reason") == "system_asleep":
        return "asleep"
    return "down"


def halim_probe(ctx: InspectionContext) -> str:
    """Return ok | asleep (expected post-market) | down (degraded)."""
    key = "halim_probe"
    if key in ctx.memo:
        value = ctx.memo[key]
        return value if isinstance(value, str) else "down"
    ctx.memo[key] = _probe_once(ctx)
    value = ctx.memo[key]
    return value if isinstance(value, str) else "down"
