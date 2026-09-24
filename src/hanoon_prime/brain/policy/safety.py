"""brain.policy.safety — live halt + pause policy (slow-cortex owned).

Persistence in safety_store.py, halt/kill side effects in safety_actions.py
(R3: under 200 lines). _SAFETY_STATE_FILE stays here — tests patch that path.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from ..._telegram import safety_halt as _telegram_halt
from ...immune import (
    CONSECUTIVE_LOSSES_PAUSE,
    DAILY_LOSS_LIMIT,
    KILL_DAILY_LOSS_LIMIT,
    MAX_CONCURRENT_POSITIONS,
    PAUSE_DURATION_MIN,
    TRAINING_KILL_BYPASS,
)
from .safety_actions import latch_kill, trip_halt
from .safety_store import (
    apply_restored_state,
    persist_safety_state,
    restore_safety_state,
)

log = logging.getLogger(__name__)

# Halt/latch state persists here (atomic tmp-file + replace) so a restart can
# never silently clear a halt or latch. Tests patch this path — the store
# helpers take it as an argument so the patched value is honored.
_SAFETY_STATE_FILE = (
    Path(__file__).resolve().parents[4] / "runtime" / "safety_state.json"
)

# Documented "auto-resume after 60 min pause" for ordinary halts.
_PAUSE_SECS = PAUSE_DURATION_MIN * 60

_BYPASS_BANNER = (
    "!!! TRAINING_KILL_BYPASS IS ACTIVE — the $500 kill switch is BYPASSED for paper "
    "training. All other halts still apply when safety is enabled. NEVER run live money "
    "with this on: set TRAINING_KILL_BYPASS = False in immune.py."
)


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
        self._halt_started_at: float | None = None
        # Named training override (immune.py): loud at startup, exposed in
        # safety status; the fail-closed paper-only guard lives in immune.py.
        self.kill_bypass: bool = TRAINING_KILL_BYPASS
        self._bypass_warned = False
        self._restore_state()
        if self.kill_bypass:
            log.critical(_BYPASS_BANNER)

    def attach_journal(self, journal: Any) -> None:
        """Wire the bot journal for halt events."""
        self._journal = journal

    def on_kill(self, hook: Callable[[str], None] | None) -> None:
        """Wire a hook run once when the kill switch latches (cancels working
        orders AND flattens positions — kill means get flat now)."""
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
        self._maybe_auto_resume()
        kill_tripped = self._daily_pnl < -KILL_DAILY_LOSS_LIMIT
        if kill_tripped and self.kill_bypass:
            # User-directed training override: skip ONLY the $500 kill; every
            # other halt below still applies. Warned loudly at startup,
            # throttled to one warning per process here.
            if not self._bypass_warned:
                self._bypass_warned = True
                log.warning(
                    "Kill level breached ($%.0f daily) but "
                    "TRAINING_KILL_BYPASS is ON — kill skipped (paper "
                    "training)",
                    KILL_DAILY_LOSS_LIMIT,
                )
        elif kill_tripped:
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
        self._halt_started_at = None
        log.warning("SAFETY: kill switch re-armed by operator")
        self._persist_state()

    def resume(self) -> None:
        """Clear a halt (webapp resume command). Latch stays latched."""
        if self.halted:
            log.info("SAFETY: halt cleared by resume")
        self.halted = False
        self.pause_reason = ""
        self._halt_started_at = None
        self._persist_state()

    def _maybe_auto_resume(self) -> None:
        """Auto-expire ordinary (non-kill) halts after PAUSE_DURATION_MIN."""
        if not self.halted or self.latched or self._halt_started_at is None:
            return
        if time.time() - self._halt_started_at >= _PAUSE_SECS:
            log.warning(
                "SAFETY: auto-resume after %d-min pause (was: %s)",
                PAUSE_DURATION_MIN,
                self.pause_reason,
            )
            self.resume()

    def _persist_state(self) -> None:
        """Persist halt/latch across restarts (see safety_store)."""
        persist_safety_state(
            _SAFETY_STATE_FILE,
            halted=self.halted,
            latched=self.latched,
            pause_reason=self.pause_reason,
            kill_reason=self._kill_reason,
            halt_started_at=self._halt_started_at,
        )

    def _restore_state(self) -> None:
        """Restore halt/latch persisted by a previous run (called at init)."""
        data = restore_safety_state(_SAFETY_STATE_FILE)
        if data:
            apply_restored_state(self, data)

    def _halt(self, reason: str) -> tuple[bool, str]:
        """Trip the halt once: flag, journal, notify (see safety_actions)."""
        return trip_halt(self, reason)

    def _kill(self, reason: str) -> tuple[bool, str]:
        """Latch the kill switch (see safety_actions)."""
        return latch_kill(self, reason)


__all__ = ["SafetyProducer"]
