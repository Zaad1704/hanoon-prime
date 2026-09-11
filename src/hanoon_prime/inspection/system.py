"""processes + identity joint checks."""

from __future__ import annotations

import os
import subprocess

from .checks import FAIL, OK, UNVERIFIABLE, CheckResult
from .ctx import InspectionContext


def _pidread(ctx: InspectionContext, service: str) -> tuple[int | None, bool, str]:
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
    """Main trading bot process is alive."""
    return _service(ctx, "hanoon_prime", r"hanoon_prime\.cli")


def halim_alive(ctx: InspectionContext) -> CheckResult:
    """HALIM serve process is alive."""
    return _service(ctx, "halim_serve", r"halim/serve\.py")


def monitor_alive(ctx: InspectionContext) -> CheckResult:
    """Guardian monitor process is alive."""
    return _service(ctx, "overnight_monitor", r"overnight_monitor\.py")


def cloudflared_alive(ctx: InspectionContext) -> CheckResult:
    """Cloudflared tunnel process is alive."""
    return _service(ctx, "cloudflared", r"cloudflared")


def gateway_watchdog_alive(ctx: InspectionContext) -> CheckResult:
    """IB gateway watchdog process is alive."""
    return _service(ctx, "ib_gateway_watchdog", r"ib_gateway_watchdog")


def single_bot(ctx: InspectionContext) -> CheckResult:
    """Exactly one bot process, never a double."""
    procs = _pgrep(r"hanoon_prime\.cli")
    if procs is None:
        return CheckResult(
            "processes", "single_bot", UNVERIFIABLE, detail="pgrep failed"
        )
    if len(procs) > 1:
        return CheckResult(
            "processes",
            "single_bot",
            FAIL,
            detail="double bot process",
            evidence={"pids": procs},
        )
    if len(procs) == 1:
        return CheckResult("processes", "single_bot", OK, evidence={"pid": procs[0]})
    return CheckResult("processes", "single_bot", OK, detail="no bot pidfile fallback")


def bot_from_trusted_checkout(ctx: InspectionContext) -> CheckResult:
    """Bot runs the hanoon_prime.cli module from this checkout."""
    pid, alive, _ = _pidread(ctx, "hanoon_prime")
    if not alive or pid is None:
        return CheckResult(
            "identity",
            "bot_from_trusted_checkout",
            UNVERIFIABLE,
            detail="bot not running",
        )
    cmd = _cmdline(pid)
    if "hanoon_prime.cli" not in cmd:
        return CheckResult(
            "identity",
            "bot_from_trusted_checkout",
            FAIL,
            detail="stray/wrong bot cmdline",
            evidence={"cmdline": cmd},
        )
    return CheckResult(
        "identity", "bot_from_trusted_checkout", OK, evidence={"cmdline": cmd}
    )
