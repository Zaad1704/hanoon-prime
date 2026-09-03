#!/usr/bin/env python3
"""IB Gateway watchdog — monitors IB connection and ensures it stays active.

Checks IB Gateway port every 30 seconds. Logs connectivity status.
If IB goes down, alerts via log so the bot can reconnect.
"""
from __future__ import annotations

import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "ib_gateway_watchdog.log"

IB_HOST = "127.0.0.1"
IB_PORT = 4002  # paper trading port
CHECK_INTERVAL = 30  # seconds between checks


def _log(msg: str) -> None:
    """Log to file and stdout."""
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC | {msg}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
    print(line)


def _port_open(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    """Check if IB Gateway port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "connected"
    except OSError as e:
        return False, str(e)


def _bot_running() -> bool:
    """Check if hanoon_prime bot is running."""
    import subprocess

    try:
        r = subprocess.run(
            ["pgrep", "-f", "hanoon_prime.cli"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return r.returncode == 0
    except Exception:
        return False


def main() -> int:
    """Main watchdog loop."""
    _log(f"IB Gateway watchdog started ({IB_HOST}:{IB_PORT})")

    port_was_up = True
    down_since: float | None = None

    while True:
        try:
            bot_active = _bot_running()

            if not bot_active:
                time.sleep(CHECK_INTERVAL * 2)  # Bot not running, check less often
                continue

            is_up, msg = _port_open(IB_HOST, IB_PORT)

            if is_up:
                if not port_was_up and down_since:
                    duration = time.time() - down_since
                    _log(f"IB Gateway port UP after {duration:.0f}s")
                port_was_up = True
                down_since = None
            else:
                if port_was_down:
                    down_since = time.time()
                    _log(f"IB Gateway port DOWN: {msg}")
                port_was_up = False

        except Exception as e:
            _log(f"Watchdog error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
