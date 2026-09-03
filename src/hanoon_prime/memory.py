"""hanoon_prime.memory — immutable hash-chained trade journal.

R7: Entries are appended and can NEVER be deleted, updated, or overwritten.
Every entry includes the SHA-256 hash of the previous entry — a
lightweight hash chain so tampering is detectable.

Appends are O(1): the entry count and last hash are cached in memory
and seeded from the file tail once at startup, so a growing journal
never forces full-file re-reads on every write.

Renamed from journal.py to complete the neuro-morphic naming.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

_TAIL_BYTES = 1 << 20  # read at most 1MB from the end for tail operations


class Journal:
    """Append-only trade journal with hash-chaining for tamper detection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count, self._last_hash_val = self._seed()

    def _seed(self) -> tuple[int, str | None]:
        """Recover entry count + last hash from the file tail (O(1))."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0, None
        with open(self.path, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - _TAIL_BYTES))
            data = f.read().decode("utf-8", "replace").rstrip("\n")
        try:
            entry = json.loads(data.split("\n")[-1])
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0, None
        seq = int(entry.get("seq", -1))
        return seq + 1, str(entry.get("hash")) if entry.get("hash") else None

    def append(self, entry: dict[str, Any]) -> None:
        """Append a single entry. Never deletes or updates existing entries."""
        stamped = {
            "ts": time.time(),
            "seq": self._count,
            "prev_hash": self._last_hash_val,
            **entry,
        }
        stamped["hash"] = self._hash_entry(stamped)
        with open(self.path, "a") as f:
            f.write(json.dumps(stamped, sort_keys=True) + "\n")
        self._count += 1
        self._last_hash_val = stamped["hash"]

    @staticmethod
    def _hash_entry(entry: dict[str, Any]) -> str:
        """Hash everything except the hash field itself."""
        to_hash = {k: v for k, v in entry.items() if k != "hash"}
        raw = json.dumps(to_hash, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def count(self) -> int:
        """Number of entries — O(1), cached in memory."""
        return self._count

    def entries(self) -> list[dict[str, Any]]:
        """Read all entries. Returns empty list if file doesn't exist."""
        if not self.path.exists():
            return []
        result: list[dict[str, Any]] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    result.append(json.loads(line))
        return result

    def tail(self, n: int) -> list[dict[str, Any]]:
        """Read the last n entries (newest last) — reads only the file tail."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        with open(self.path, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - _TAIL_BYTES))
            data = f.read().decode("utf-8", "replace")
        lines = [ln for ln in data.split("\n") if ln.strip()]
        parsed = [json.loads(ln) for ln in lines[-n:]]
        return parsed

    def verify_chain(self) -> bool:
        """Verify hash chain integrity. Returns True if intact."""
        entries = self.entries()
        prev_hash: str | None = None
        for entry in entries:
            if entry.get("prev_hash") != prev_hash:
                return False
            expected = self._hash_entry(
                {k: v for k, v in entry.items() if k != "hash"}
                | {"prev_hash": prev_hash}
            )
            if entry.get("hash") != expected:
                return False
            prev_hash = entry["hash"]
        return True
