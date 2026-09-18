"""monitor.vitals_log — rotating CSV store of per-cycle pipeline vitals."""

from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_ROWS_PER_FILE: int = 86_400  # ~24h at the 1 Hz cycle cadence
DEFAULT_MAX_FILES: int = 7  # one week of daily files retained

_COLUMNS: tuple[str, ...] = (
    "ts",
    "market_open",
    "session",
    "session_active",
    "ib_connected",
    "decision_count",
    "feed_age",
    "stall_cycles",
    "healthy",
)


def _flatten(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Reduce a PipelineMonitor snapshot to the fixed vitals row columns."""
    vit = snapshot.get("vitals", {})
    return {
        "ts": round(float(vit.get("ts") or time.time()), 3),
        "market_open": bool(vit.get("market_open")),
        "session": str(vit.get("session") or ""),
        "session_active": bool(vit.get("session_active")),
        "ib_connected": bool(vit.get("ib_connected")),
        "decision_count": int(vit.get("decision_count") or 0),
        "feed_age": round(float(vit.get("feed_age") or 0.0), 1),
        "stall_cycles": int(snapshot.get("stall_cycles") or 0),
        "healthy": bool(snapshot.get("healthy")),
    }


def _coerce(row: dict[str, Any]) -> dict[str, Any]:
    """Parse one raw CSV row into typed vitals values."""
    try:
        return {
            "ts": float(row.get("ts") or 0.0),
            "market_open": bool(row.get("market_open") == "True"),
            "session": str(row.get("session") or ""),
            "session_active": bool(row.get("session_active") == "True"),
            "ib_connected": bool(row.get("ib_connected") == "True"),
            "decision_count": int(row.get("decision_count") or 0),
            "feed_age": round(float(row.get("feed_age") or 0.0), 1),
            "stall_cycles": int(row.get("stall_cycles") or 0),
            "healthy": bool(row.get("healthy") == "True"),
        }
    except (TypeError, ValueError):
        return {}


class VitalsLog:
    """Append-only rotating CSV store of per-cycle pipeline vitals.

    Writes happen once per main cycle (1 Hz) from the cycle thread; reads
    (tail/since) are served on demand to the telemetry API. A single lock
    serializes writers and readers so record() never races tail().
    """

    def __init__(
        self,
        base: Path,
        rows_per_file: int = DEFAULT_ROWS_PER_FILE,
        max_files: int = DEFAULT_MAX_FILES,
    ) -> None:
        self._base = Path(base)
        self._rows_per_file = rows_per_file
        self._max_files = max_files
        self._lock = threading.Lock()
        self._rows_in_file: int = 0
        self._current: Path | None = None
        self._base.mkdir(parents=True, exist_ok=True)
        self._seq: int = 0

    def _new_file(self) -> Path:
        """Allocate a unique timestamped file (sub-second-safe rotation)."""
        self._seq += 1
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{self._seq:03d}"
        return self._base / f"vitals_{stamp}.csv"

    def _rotate_if_needed(self) -> Path:
        """Open today's file or rotate to the next window when full."""
        if self._current is not None and self._rows_in_file < self._rows_per_file:
            return self._current
        self._current = self._new_file()
        with self._current.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=list(_COLUMNS)).writeheader()
        self._rows_in_file = 0
        self._prune()
        return self._current

    def record(self, snapshot: dict[str, Any]) -> None:
        """Persist one pipeline snapshot as a CSV row (thread-safe)."""
        with self._lock:
            path = self._rotate_if_needed()
            try:
                with path.open("a", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=list(_COLUMNS)).writerow(
                        _flatten(snapshot)
                    )
                self._rows_in_file += 1
            except OSError as exc:
                log.warning("vitals write failed: %s", exc)

    def _prune(self) -> None:
        """Delete oldest files beyond max_files (oldest→newest by name)."""
        try:
            files = sorted(self._base.glob("vitals_*.csv"))
        except OSError as exc:
            log.debug("vitals prune glob failed: %s", exc)
            return
        for old in files[: -self._max_files]:
            try:
                old.unlink()
            except OSError as exc:
                log.debug("vitals prune unlink %s failed: %s", old.name, exc)

    def _files(self) -> list[Path]:
        """Retained vitals files, oldest→newest by timestamped name."""
        return sorted(self._base.glob("vitals_*.csv"))

    def _read(self, path: Path) -> list[dict[str, Any]]:
        """Read one CSV file into typed rows (best-effort, non-fatal)."""
        try:
            with path.open(newline="") as fh:
                return [r for row in csv.DictReader(fh) if (r := _coerce(row))]
        except (OSError, csv.Error) as exc:
            log.debug("vitals read %s failed: %s", path.name, exc)
            return []

    def tail(self, n: int = 120) -> list[dict[str, Any]]:
        """Return the n most recent rows (oldest→newest within the tail)."""
        with self._lock:
            rows: list[dict[str, Any]] = []
            for path in reversed(self._files()):
                rows = self._read(path) + rows
                if len(rows) >= n:
                    break
            if not rows:
                return []
            return rows[-n:]

    def _rows_from(self, path: Path, since_ts: float) -> list[dict[str, Any]]:
        """Rows from one file with ts >= since_ts (helper keeps nesting ≤3)."""
        out: list[dict[str, Any]] = []
        for row in self._read(path):
            if row["ts"] >= since_ts:
                out.append(row)
        return out

    def since(self, ts: float) -> list[dict[str, Any]]:
        """Return rows with ts >= ts across retained files (chronological)."""
        with self._lock:
            out: list[dict[str, Any]] = []
            for path in self._files():
                out.extend(self._rows_from(path, ts))
            return out
