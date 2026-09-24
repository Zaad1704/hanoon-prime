"""brain.policy.safety_store — halt/latch persistence for SafetyProducer.

Split out of brain/policy/safety.py so that module stays under the R3
200-line contract limit. The state-file path still lives in safety.py as
``_SAFETY_STATE_FILE`` (tests patch it there); these helpers take the path
as an argument so the patched value is honored.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def persist_safety_state(
    path: Path,
    *,
    halted: bool,
    latched: bool,
    pause_reason: str,
    kill_reason: str,
    halt_started_at: float | None,
) -> None:
    """Persist halt/latch across restarts (atomic tmp-file + replace).

    A reboot must never silently clear a safety halt or kill latch.
    Best-effort: a failed write logs but never breaks the policy.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "halted": halted,
            "latched": latched,
            "pause_reason": pause_reason,
            "kill_reason": kill_reason,
            "halt_started_at": halt_started_at,
            "saved_at": time.time(),
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except Exception as exc:
        log.warning("Safety state persist failed: %s", exc)


def restore_safety_state(path: Path) -> dict[str, Any]:
    """Read persisted halt/latch state; {} when absent or corrupt."""
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("Safety state read failed: %s", exc)
        return {}
    try:
        data = json.loads(raw)
    except Exception as exc:
        log.warning("Safety state corrupt, starting clean: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def apply_restored_state(producer: Any, data: dict[str, Any]) -> None:
    """Apply a parsed state dict onto a SafetyProducer (init path)."""
    producer.halted = bool(data.get("halted", False))
    producer.latched = bool(data.get("latched", False))
    reason = data.get("pause_reason")
    producer.pause_reason = str(reason) if reason else ""
    kreason = data.get("kill_reason")
    producer._kill_reason = str(kreason) if kreason else ""
    started = data.get("halt_started_at")
    try:
        producer._halt_started_at = float(started) if started is not None else None
    except (TypeError, ValueError):
        producer._halt_started_at = None
    if producer.halted or producer.latched:
        log.warning(
            "SAFETY: restored persisted state halted=%s latched=%s " "reason=%r",
            producer.halted,
            producer.latched,
            producer.pause_reason or producer._kill_reason,
        )


__all__ = [
    "persist_safety_state",
    "restore_safety_state",
    "apply_restored_state",
]
