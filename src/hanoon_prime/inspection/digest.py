"""Daily Telegram digest — one per local day, chunked to send limit."""

from __future__ import annotations

import json
import time

from .._telegram import send
from .checks import OK
from .ctx import InspectionContext
from .joints import JOINT_ORDER, Manifest

SEV = {"OK": 0, "WARN": 1, "FAIL": 2, "UNVERIFIABLE": 3}


def _day(ts: float) -> str:
    """Local calendar day for a timestamp."""
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def build_digest(manifest: Manifest) -> str:
    """Human-readable daily summary; ok joints are folded to one line."""
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
            details = "; ".join(
                f"{r.name}={r.status}:{r.detail[:40]}" for r in rows if r.status != OK
            )
            lines.append(f"  {joint}: {details}")
    return "\n".join(lines)


def _chunks(text: str, size: int = 4000) -> list[str]:
    """Split text into telegram-sized pieces."""
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


def _notify(text: str, ctx: InspectionContext) -> bool:
    """Send one chunk via the real telegram adapter."""
    try:
        return send(text)
    except Exception:
        return False


def digest_send(ctx: InspectionContext, manifest: Manifest) -> tuple[bool, str]:
    """Emit the digest at most once per local day."""
    ledger = ctx.ledger()
    today = _day(manifest.ts)
    if ledger.get("digest", {}).get("latest_day") == today:
        return False, "already sent today"
    text = build_digest(manifest)
    ok = all(_notify(part, ctx) for part in _chunks(text))
    if ok:
        ledger.setdefault("digest", {})["latest_day"] = today
        ctx.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.ledger_path.write_text(json.dumps(ledger, sort_keys=True, indent=2))
    return ok, "sent" if ok else "send failed"
