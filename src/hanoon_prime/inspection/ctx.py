"""hanoon_prime.inspection.ctx — where everything lives.

Resolves repo paths, pidfiles, git HEAD, and the runtime surfaces the checks
poke. Cheap reads are memoized per InspectionContext instance so one probe
tick touches each source (log, json files, journal, HTTP) only once.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InspectionContext:
    """Paths, urls, and cached probes for one manifest tick."""

    base_dir: Path
    telemetry_url: str = "http://127.0.0.1:8080"
    halim_url: str = "http://127.0.0.1:8765"
    prev_journal_count: int | None = None
    heal_enabled: bool = True
    halim_start_script: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "scripts"
        / "halim_start.sh"
    )
    launch_detached_script: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "scripts"
        / "launch_detached.py"
    )
    watchdog_script: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "scripts"
        / "ib_gateway_watchdog.py"
    )
    memo: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.base_dir = self.base_dir.resolve()

    @property
    def runtime(self) -> Path:
        """Repo runtime state dir (gitignored)."""
        return self.base_dir / "runtime"

    @property
    def pid_dir(self) -> Path:
        """PID file directory."""
        return self.runtime / "pids"

    @property
    def logs_dir(self) -> Path:
        """Log output directory."""
        return self.base_dir / "logs"

    @property
    def log_path(self) -> Path:
        """Main bot log."""
        return self.logs_dir / "hanoon_prime.log"

    @property
    def state_path(self) -> Path:
        """Brain policy state dump."""
        return self.runtime / "state.json"

    @property
    def juli_state_path(self) -> Path:
        """Juli learning state dump."""
        return self.runtime / "juli_state.json"

    @property
    def regime_weights_path(self) -> Path:
        """Juli per-regime weight vectors."""
        return self.runtime / "juli_regime_weights.json"

    @property
    def realized_path(self) -> Path:
        """Juli realized-PnL history."""
        return self.runtime / "juli_realized.json"

    @property
    def journal_path(self) -> Path:
        """Append-only journal."""
        return self.runtime / "journal_live.jsonl"

    @property
    def state_file(self) -> Path:
        """Guardian production-state ledger."""
        return self.base_dir / "scripts" / "production_state.json"

    @property
    def ledger_path(self) -> Path:
        """Shared production-state ledger (inspection + guardian)."""
        return self.base_dir / "scripts" / "production_state.json"

    def ledger(self) -> dict[str, Any]:
        """Shared production-state ledger, read fresh each call."""
        try:
            raw = self.ledger_path.read_text()
        except FileNotFoundError:
            raw = ""
        data: dict[str, Any] = {}
        if raw.strip():
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                data = loaded
        return data

    @property
    def venv_python(self) -> Path:
        """Venv interpreter used to launch daemons."""
        return self.base_dir / ".venv" / "bin" / "python"

    @property
    def launch_detached(self) -> Path:
        """True-detach launcher for background services."""
        return self.base_dir / "scripts" / "launch_detached.py"

    @property
    def halim_start(self) -> Path:
        """Idempotent HALIM serve starter."""
        return self.base_dir / "scripts" / "halim_start.sh"

    @property
    def halim_log_path(self) -> Path:
        """HALIM serve log."""
        return self.logs_dir / "halim_serve.log"

    def pid_file(self, service: str) -> Path:
        """PID file for a named service."""
        return self.pid_dir / f"{service}.pid"

    def git_head(self) -> str | None:
        """Repo HEAD (cached). None when git is unavailable."""
        key = "git_head"
        if key not in self.memo:
            try:
                out = subprocess.run(
                    ["git", "-C", str(self.base_dir), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.memo[key] = out.stdout.strip() if out.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                self.memo[key] = None
        value = self.memo[key]
        return value if isinstance(value, str) else None
