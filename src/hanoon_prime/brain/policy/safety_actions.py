"""brain.policy.safety_actions — halt/kill side effects for SafetyProducer.

Split out of brain/policy/safety.py so that module stays under the R3
200-line contract limit. These are the exact bodies moved out of
``SafetyProducer._halt`` / ``SafetyProducer._kill`` — behavior is
unchanged; the producer keeps thin wrappers so its public interface
(and private method names) are untouched.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


def trip_halt(producer: Any, reason: str) -> tuple[bool, str]:
    """Trip the halt once: flag, journal, notify, persist."""
    producer.halted = True
    producer.pause_reason = reason
    producer._halt_started_at = time.time()
    log.critical("SAFETY HALT: %s", reason)
    if producer._journal is not None:
        try:
            producer._journal.append(
                {"event": "halt", "reason": reason, "ts": time.time()}
            )
        except Exception as exc:
            log.warning("Safety journal failed: %s", exc)
    try:
        producer._notify(reason)
    except Exception as exc:
        log.warning("Safety notify failed: %s", exc)
    producer._persist_state()
    return False, reason


def latch_kill(producer: Any, reason: str) -> tuple[bool, str]:
    """Latch the kill switch: halt + block entries until manual re-arm.

    The kill switch is LATCHED: only ``release_kill`` (operator re-arm)
    clears it — resume() intentionally does not. Runs the order-cancel
    hook once so working orders are pulled before anything else.
    """
    producer.halted = True
    producer.latched = True
    producer.pause_reason = reason
    producer._kill_reason = reason
    log.critical("KILL SWITCH LATCHED: %s (manual re-arm required)", reason)
    if producer._journal is not None:
        try:
            producer._journal.append(
                {
                    "event": "kill",
                    "reason": reason,
                    "latched": True,
                    "ts": time.time(),
                }
            )
        except Exception as exc:
            log.warning("Kill journal failed: %s", exc)
    try:
        if producer._on_kill is not None:
            producer._on_kill(reason)
    except Exception as exc:
        log.warning("Kill hook failed: %s", exc)
    try:
        producer._notify(f"KILL SWITCH: {reason} — manual re-arm required")
    except Exception as exc:
        log.warning("Kill notify failed: %s", exc)
    producer._persist_state()
    return False, reason


__all__ = ["trip_halt", "latch_kill"]
