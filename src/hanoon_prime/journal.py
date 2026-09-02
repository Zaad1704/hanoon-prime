"""hanoon_prime.journal — immutable append-only trade log.

R7: Entries are appended and can NEVER be deleted, updated, or overwritten.
Every entry is JSON + SHA256 hash of the previous entry — a lightweight
Merkle chain so tampering is detectable.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class Journal:
    """Append-only trade journal with hash-chaining for tamper detection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict[str, Any]) -> None:
        """Append a single entry. Never deletes or updates existing entries."""
        stamped = {
            "ts": time.time(),
            "seq": self._next_seq(),
            "prev_hash": self._last_hash(),
            **entry,
        }
        stamped["hash"] = self._hash_entry(stamped)

        with open(self.path, "a") as f:
            f.write(json.dumps(stamped, sort_keys=True) + "\n")

    def _next_seq(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path) as f:
            lines = f.readlines()
        return len(lines)

    def _last_hash(self) -> str | None:
        if not self.path.exists():
            return None
        with open(self.path) as f:
            lines = f.readlines()
        if not lines:
            return None
        try:
            return json.loads(lines[-1])["hash"]
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def _hash_entry(entry: dict) -> str:
        """Hash everything except the hash field itself."""
        to_hash = {k: v for k, v in entry.items() if k != "hash"}
        raw = json.dumps(to_hash, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def entries(self) -> list[dict[str, Any]]:
        """Read all entries. Returns empty list if file doesn't exist."""
        if not self.path.exists():
            return []
        result = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    result.append(json.loads(line))
        return result

    def verify_chain(self) -> bool:
        """Verify hash chain integrity. Returns True if intact."""
        entries = self.entries()
        prev_hash = None
        for i, entry in enumerate(entries):
            if entry.get("prev_hash") != prev_hash:
                return False
            expected = self._hash_entry({k: v for k, v in entry.items() if k != "hash"} | {"prev_hash": prev_hash})
            if entry.get("hash") != expected:
                return False
            prev_hash = entry["hash"]
        return True
