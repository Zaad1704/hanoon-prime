"""Gated auto-heal — mechanical, non-trading services only."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .ctx import InspectionContext
from .joints import Manifest

MAX_PER_DAY = 2

HEALABLE = ("halim_alive", "cloudflared_alive", "gateway_watchdog_alive")


def _day(ts: float) -> str:
    """Local calendar day for a timestamp."""
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _today() -> str:
    """Local calendar day for now."""
    return _day(time.time())


def heal_enabled(ctx: InspectionContext) -> bool:
    """Healing is enabled for this context."""
    return ctx.heal_enabled


def _ledger_day_count(ctx: InspectionContext) -> int:
    """Heal actions recorded so far today, 0 cross-day."""
    h = ctx.ledger().get("heals", {})
    if not isinstance(h, dict):
        return 0
    count = h.get("count", 0)
    if not isinstance(count, int):
        return 0
    return count if h.get("today") == _today() else 0


def _save(ctx: InspectionContext, ledger: dict[str, Any]) -> None:
    """Persist the shared ledger."""
    ctx.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.ledger_path.write_text(json.dumps(ledger, sort_keys=True, indent=2))


def record_heal(ctx: InspectionContext, name: str) -> None:
    """Record one heal action in the daily budget."""
    ledger = ctx.ledger()
    h = ledger.setdefault("heals", {})
    if h.get("today") != _today():
        h.update(today=_today(), count=0)
    h["count"] = h.get("count", 0) + 1
    h.setdefault("events", []).append({"name": name, "ts": time.time()})
    _save(ctx, ledger)


def candidates(ctx: InspectionContext, manifest: Manifest) -> list[str]:
    """Healable services currently reporting FAIL."""
    failing = {r.name for r in manifest.results if r.status == "FAIL"}
    return [n for n in HEALABLE if n in failing]


def _commands_for(name: str, ctx: InspectionContext) -> list[str]:
    """Launch command for a healable service."""
    home = str(Path.home())
    if name == "halim_alive":
        return [str(ctx.halim_start_script)]
    if name == "cloudflared_alive":
        return [
            sys.executable,
            str(ctx.launch_detached_script),
            "/opt/homebrew/bin/cloudflared",
            "tunnel",
            "--config",
            f"{home}/.cloudflared/config.yml",
            "run",
        ]
    if name == "gateway_watchdog_alive":
        return [
            sys.executable,
            str(ctx.launch_detached_script),
            str(ctx.watchdog_script),
        ]
    raise ValueError(name)


def _run(cmd: list[str], dry_run: bool) -> str:
    """Execute (or preview) a heal command."""
    if dry_run:
        return "would run: " + " ".join(cmd)
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return f"ran: {' '.join(cmd)} rc={cp.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"failed: {exc}"


def heal(
    ctx: InspectionContext, manifest: Manifest, dry_run: bool = False
) -> list[str]:
    """Apply gated, budgeted heals for failed mechanical services."""
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
