"""brain.policy.safety — live halt + pause policy (slow-cortex owned).

Safety nets are portfolio-level and low-cadence, so the slow cortex feeds
them from the account feed and publishes ``authorized`` in policy_state.
Halt behavior is preserved from the old ib_cycle gate: block NEW entries,
never stop the bot, notify + journal. The sim-path pause in
Hippocampus.check_entry_allowed stays untouched.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ..._telegram import safety_halt as _telegram_halt
from ...immune import (
    CONSECUTIVE_LOSSES_PAUSE,
    DAILY_LOSS_LIMIT,
    KILL_DAILY_LOSS_LIMIT,
    MAX_CONCURRENT_POSITIONS,
)

log = logging.getLogger(__name__)


class SafetyProducer:
    """Feed-run halt/pause policy; publishes authorization each cycle."""

    def __init__(
        self,
        journal: Any = None,
        notify: Callable[[str], None] | None = None,
        enabled: bool = False,
    ) -> None:
        """Start disabled (deactivated until the webapp enables), unhalted."""
        self.enabled: bool = enabled
        self.halted: bool = False
        self.pause_reason: str = ""
        self.latched: bool = False  # kill switch: latched = manual re-arm required
        self._kill_reason: str = ""
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._position_count: int = 0
        self._journal = journal
        self._notify: Callable[[str], None] = notify or _telegram_halt
        self._on_kill: Callable[[str], None] | None = None

    def attach_journal(self, journal: Any) -> None:
        """Wire the bot journal for halt events."""
        self._journal = journal

    def on_kill(self, hook: Callable[[str], None] | None) -> None:
        """Wire a hook run once when the kill switch latches (cancel orders)."""
        self._on_kill = hook

    def set_enabled(self, enabled: bool) -> None:
        """Toggle the safety system (webapp command)."""
        self.enabled = bool(enabled)
        if not self.enabled:
            self.resume()

    def begin_call(self) -> None:
        """Start one slow-cortex policy pulse (keep a live halt's reason)."""
        if not self.halted:
            self.pause_reason = ""

    def on_daily_pnl(self, pnl: float) -> None:
        """Feed IB day P&L."""
        self._daily_pnl = float(pnl)

    def on_consecutive_losses(self, n: int) -> None:
        """Feed current loss streak."""
        self._consecutive_losses = int(n)

    def on_position_count(self, n: int) -> None:
        """Feed current open position count."""
        self._position_count = int(n)

    def authorized(self) -> tuple[bool, str]:
        """(True, "") to allow entries; (False, reason) to veto."""
        if self.latched:
            return False, self._kill_reason or "kill_switch_latched"
        if not self.enabled:
            return True, ""
        if self._daily_pnl < -KILL_DAILY_LOSS_LIMIT:
            return self._kill("kill_daily_loss_limit")
        if self.halted:
            return False, self.pause_reason
        if self._daily_pnl < -DAILY_LOSS_LIMIT:
            return self._halt("daily_loss_limit")
        if self._consecutive_losses >= CONSECUTIVE_LOSSES_PAUSE:
            return self._halt("consecutive_losses")
        if self._position_count > MAX_CONCURRENT_POSITIONS:
            return self._halt("too_many_positions")
        return True, ""

    def release_kill(self) -> None:
        """Manual re-arm: clear the latch (operator command, not auto-resume)."""
        self.latched = False
        self._kill_reason = ""
        self.halted = False
        self.pause_reason = ""
        log.warning("SAFETY: kill switch re-armed by operator")

    def resume(self) -> None:
        """Clear a halt (webapp resume command). Latch stays latched."""
        if self.halted:
            log.info("SAFETY: halt cleared by resume")
        self.halted = False
        self.pause_reason = ""

    def _halt(self, reason: str) -> tuple[bool, str]:
        """Trip the halt once: flag, journal, notify."""
        self.halted = True
        self.pause_reason = reason
        log.critical("SAFETY HALT: %s", reason)
        if self._journal is not None:
            try:
                self._journal.append(
                    {"event": "halt", "reason": reason, "ts": time.time()}
                )
            except Exception as exc:
                log.warning("Safety journal failed: %s", exc)
        try:
            self._notify(reason)
        except Exception as exc:
            log.warning("Safety notify failed: %s", exc)
        return False, reason

    def _kill(self, reason: str) -> tuple[bool, str]:
        """Latch the kill switch: halt + block entries until manual re-arm.

        The kill switch is LATCHED: only ``release_kill`` (operator re-arm)
        clears it — resume() intentionally does not. Runs the order-cancel
        hook once so working orders are pulled before anything else.
        """
        self.halted = True
        self.latched = True
        self.pause_reason = reason
        self._kill_reason = reason
        log.critical("KILL SWITCH LATCHED: %s (manual re-arm required)", reason)
        if self._journal is not None:
            try:
                self._journal.append(
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
            if self._on_kill is not None:
                self._on_kill(reason)
        except Exception as exc:
            log.warning("Kill hook failed: %s", exc)
        try:
            self._notify(f"KILL SWITCH: {reason} — manual re-arm required")
        except Exception as exc:
            log.warning("Kill notify failed: %s", exc)
        return False, reason


__all__ = ["SafetyProducer"]
