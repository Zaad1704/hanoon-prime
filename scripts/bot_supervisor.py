#!/usr/bin/env python3
"""bot_supervisor — lightweight process supervisor for hanoon_prime.cli.

Launches the trading bot as a subprocess and optionally restarts it on
unexpected exit.  Auto-restart is OFF by default; pass ``--restart`` (or
set ``BOT_AUTO_RESTART=1`` in the environment) to enable it.

Usage:
    python3 scripts/bot_supervisor.py                       # no auto-restart
    python3 scripts/bot_supervisor.py --restart            # auto-restart on crash
    python3 scripts/bot_supervisor.py --restart --max-restarts 5 --backoff 10
    BOT_AUTO_RESTART=1 python3 scripts/bot_supervisor.py --restart

The supervisor writes its own PID to runtime/pids/bot_supervisor.pid and
the bot's PID to runtime/pids/hanoon_prime.pid so stop.command works
without modification.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
PID_DIR = ROOT / "runtime" / "pids"
BOT_LOG = LOG_DIR / "hanoon_prime.log"
SUPERVISOR_PID = PID_DIR / "bot_supervisor.pid"
BOT_PIDFILE = PID_DIR / "hanoon_prime.pid"

LOG_DIR.mkdir(parents=True, exist_ok=True)
PID_DIR.mkdir(parents=True, exist_ok=True)


class Supervisor:
    """Wrap a subprocess with optional crash-restart logic."""

    def __init__(
        self,
        auto_restart: bool,
        max_restarts: int = 10,
        backoff: float = 5.0,
        max_backoff: float = 300.0,
    ) -> None:
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.base_backoff = backoff
        self.max_backoff = max_backoff
        self._restart_count = 0
        self._child: subprocess.Popen | None = None
        self._shutdown = False

    # ── process management ──────────────────────────────────────────

    def _resolve_python(self) -> str:
        return os.environ.get("PYTHON_BIN") or sys.executable

    def _start_child(self) -> subprocess.Popen:
        if self._child is not None and self._child.poll() is None:
            raise RuntimeError("child already running")
        python = self._resolve_python()
        log_fd = open(BOT_LOG, "ab", buffering=0)
        proc = subprocess.Popen(
            [python, "-m", "hanoon_prime.cli"],
            cwd=str(ROOT),
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        BOT_PIDFILE.write_text(str(proc.pid))
        log(f"Bot started (PID {proc.pid})")
        return proc

    def _stop_child(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        log(f"Sending SIGTERM to bot (PID {proc.pid})...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(10):
            if proc.poll() is not None:
                return
            time.sleep(1)
        log("Bot ignored SIGTERM — SIGKILL")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    # ── main loop ───────────────────────────────────────────────────

    def run(self) -> int:
        self._register_signals()
        self._child = self._start_child()
        last_exit = 0.0
        while True:
            if self._child is None:
                break
            rc = self._child.poll()
            if rc is None:
                try:
                    self._child.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    continue
                continue
            # Child exited
            elapsed = time.monotonic() - last_exit
            log(f"Bot exited (rc={rc}) after {elapsed:.1f}s")
            BOT_PIDFILE.unlink(missing_ok=True)
            self._child = None
            last_exit = time.monotonic()

            if self._shutdown:
                log("Shutdown requested — not restarting")
                return rc if rc is not None else 1

            if not self.auto_restart:
                log("Auto-restart DISABLED — supervisor exiting")
                return rc if rc is not None else 1

            if self._restart_count >= self.max_restarts:
                log(f"Max restarts ({self.max_restarts}) reached — giving up")
                return rc if rc is not None else 1

            backoff = min(
                self.base_backoff * (2**self._restart_count),
                self.max_backoff,
            )
            self._restart_count += 1
            log(
                f"Restart #{self._restart_count} in {backoff:.0f}s "
                f"(backoff: {backoff:.0f}s)"
            )
            time.sleep(backoff)
            self._child = self._start_child()

        return 0

    # ── signal handling ─────────────────────────────────────────────

    def _register_signals(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        log(f"Received {sig_name} — initiating shutdown")
        self._shutdown = True
        if self._child is not None and self._child.poll() is None:
            self._stop_child(self._child)
        self._cleanup()
        sys.exit(0)

    def _cleanup(self) -> None:
        SUPERVISOR_PID.unlink(missing_ok=True)
        BOT_PIDFILE.unlink(missing_ok=True)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [supervisor] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="HANOON Prime bot supervisor")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Enable auto-restart on crash (default: OFF)",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=10,
        help="Maximum restart attempts before giving up (default: 10)",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=5.0,
        help="Base backoff in seconds between restarts (default: 5)",
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=300.0,
        help="Maximum backoff in seconds (default: 300)",
    )

    args = parser.parse_args()

    # Env var override: BOT_AUTO_RESTART=1 enables restart
    auto_restart = args.restart or os.environ.get("BOT_AUTO_RESTART", "") == "1"

    SUPERVISOR_PID.write_text(str(os.getpid()))

    if auto_restart:
        log(
            f"Auto-restart: ENABLED "
            f"(max={args.max_restarts}, backoff={args.backoff}s, "
            f"max_backoff={args.max_backoff}s)"
        )
    else:
        log(
            "Auto-restart: DISABLED (pass --restart or set BOT_AUTO_RESTART=1 to enable)"
        )

    sup = Supervisor(
        auto_restart=auto_restart,
        max_restarts=args.max_restarts,
        backoff=args.backoff,
        max_backoff=args.max_backoff,
    )
    try:
        return sup.run()
    finally:
        sup._cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
