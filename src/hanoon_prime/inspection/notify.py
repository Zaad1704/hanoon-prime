"""Inspection alerting — Telegram hard-fail / critical-WARN notifications.

Called by the telemetry server's /inspection handler so Inside Man findings
surface in the user's Telegram chat alongside the webapp. Deduped so the bot
doesn't spam on every 30 s manifest refresh.
"""

from __future__ import annotations

from typing import Any

from .._telegram import send
from .checks import FAIL, WARN

_last_sig: str = ""


def manifest_notify(manifest: dict[str, Any]) -> None:
    """Send Inside Man hard-fails + critical WARNs to Telegram (deduped).

    Fires only when the set of failing/anomalous checks changes.
    """
    global _last_sig
    results = manifest.get("results", []) if isinstance(manifest, dict) else []
    if not isinstance(results, list):
        return
    critical = [
        r for r in results if isinstance(r, dict) and r.get("status") in (FAIL, WARN)
    ]
    if not critical:
        _last_sig = ""
        return
    sig = "|".join(f"{r.get('joint', '?')}.{r.get('name', '?')}" for r in critical)
    if sig == _last_sig:
        return
    _last_sig = sig
    lines: list[str] = []
    for r in critical:
        emoji = "❌" if r.get("status") == FAIL else "⚠️"
        joint = r.get("joint", "?")
        name = r.get("name", "?")
        detail = str(r.get("detail", ""))[:120]
        lines.append(f"{emoji} {joint}.{name}: {detail}")
    send("🚨 Inside Man alert\n" + "\n".join(lines))


__all__ = ["manifest_notify"]
