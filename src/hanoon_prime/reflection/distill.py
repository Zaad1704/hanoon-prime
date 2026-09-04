"""reflection.distill — Memory distillation.

Periodically distills live memory into a verified digest.
If live state fails validation, rolls back to last verified digest.

Source: rebuild's distill.py (simplified).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_DIGEST_PATH = Path("runtime/memory_digest.json")
_MAX_HISTORY: int = 5


class DistillationEngine:
    """Memory distillation — prevent corruption."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._digests: list[dict] = []
        self._load()

    def distill(self, memory: Any) -> None:
        """Create a verified digest of current memory state."""
        snap = memory.snapshot() if hasattr(memory, "snapshot") else {}
        snap["_ts"] = time.time()
        self._digests.append(snap)
        if len(self._digests) > _MAX_HISTORY:
            self._digests = self._digests[-_MAX_HISTORY:]
        self._save()

    def guard(self, memory: Any) -> bool:
        """Validate live memory against last digest. Returns True if OK."""
        if not self._digests:
            return True
        last = self._digests[-1]
        snap = memory.snapshot() if hasattr(memory, "snapshot") else {}
        # Check for suspicious changes
        old_wr = last.get("win_rate", 0.5)
        new_wr = snap.get("win_rate", 0.5)
        if abs(new_wr - old_wr) > 0.3 and snap.get("total_trades", 0) > 10:
            log.warning(
                "Memory WR jumped from %.2f to %.2f — possible corruption",
                old_wr,
                new_wr,
            )
            return False
        return True

    def rollback(self, memory: Any) -> bool:
        """Rollback to last verified digest."""
        if not self._digests:
            return False
        last = self._digests[-1]
        if hasattr(memory, "set_weights") and "weights" in last:
            memory.set_weights(last["weights"])
            log.info("Memory rolled back to digest from %.0f", last.get("_ts", 0))
            return True
        return False

    def _save(self) -> None:
        """Auto-generated docstring."""
        try:
            _DIGEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _DIGEST_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._digests, default=str))
            tmp.replace(_DIGEST_PATH)
        except Exception as exc:
            log.debug("Digest save failed: %s", exc)

    def _load(self) -> None:
        """Auto-generated docstring."""
        if not _DIGEST_PATH.exists():
            return
        try:
            self._digests = json.loads(_DIGEST_PATH.read_text())
        except Exception as exc:
            log.debug("Digest load failed: %s", exc)
