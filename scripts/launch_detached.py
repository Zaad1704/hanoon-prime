#!/usr/bin/env python3
"""launch_detached.py — detach a command via double-fork + setsid.

Usage: launch_detached.py <pidfile> <logfile> <cmd> [args...]

Why this exists (2026-08-15):
  • `nohup ... &` from a script gets killed when the launching shell's
    process group is reaped — the daemon must escape to its own session.
  • `launchctl submit` (launchd) survives that, BUT macOS TCC blocks
    launchd-spawned processes from reading ~/Downloads — and this whole
    project (and the HALIM repo) lives under ~/Downloads. So launchd is
    unusable here.
  • Double-fork + setsid creates a new session with no controlling
    terminal. The process keeps the TCC grants inherited from the caller
    (the terminal app), survives shell exit, and writes its real PID to
    <pidfile> so stop scripts can signal it.

The command inherits the caller's environment and working directory.
stdout/stderr/stdin are redirected to <logfile>.
"""

import os
import sys


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    pidfile = os.path.abspath(sys.argv[1])
    logfile = os.path.abspath(sys.argv[2])
    cmd = sys.argv[3:]
    if not cmd:
        print("launch_detached: empty command", file=sys.stderr)
        return 2

    # Fork 1 — detach from the launching shell.
    pid1 = os.fork()
    if pid1 > 0:
        # Original process: wait briefly to ensure the grandchild's pidfile
        # is written (so callers can read it synchronously), then exit.
        try:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if os.path.exists(pidfile):
                    break
                time.sleep(0.05)
        except Exception:
            pass
        return 0

    # Child 1: new session (no controlling terminal), then fork again so the
    # grandchild is not a session leader (cannot reacquire a tty).
    try:
        os.setsid()
    except Exception:
        pass
    pid2 = os.fork()
    if pid2 > 0:
        # Child 1: record the grandchild's PID, then exit.
        try:
            with open(pidfile, "w") as f:
                f.write(str(pid2))
        except Exception:
            pass
        os._exit(0)

    # Grandchild: the actual daemon. Redirect std fds to the log, exec the
    # command in-place (PID preserved).
    try:
        fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 0)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        if fd > 2:
            os.close(fd)
    except Exception:
        pass
    try:
        os.execvp(cmd[0], cmd)
    except Exception as exc:
        with open(logfile, "a") as f:
            f.write(f"launch_detached: exec failed: {exc}\n")
        os._exit(127)


if __name__ == "__main__":
    import time  # noqa: PLC0415 — imported here so fork timing is precise

    sys.exit(main())
