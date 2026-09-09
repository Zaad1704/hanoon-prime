"""monitor.sleep_manager — Market session awareness.

Decides whether the bot should be ACTIVE (trading) or SLEEPING (idle).
Honors market sessions (pre_market/RTH/post_market/overnight), weekends,
and holidays.

Session ids MATCH TradingConfig keys exactly — pre_market / rth /
post_market / overnight. No capitalized "RTH", no legacy "pre_post":
an id mismatch silently short-circuits the gate (is_session_active
falls back to True), so the contract lives in SESSION_IDS below.

Uses zoneinfo for proper US/Eastern timezone (handles DST automatically).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Canonical session ids — must equal TradingConfig.session_* suffixes.
SESSION_IDS = frozenset({"pre_market", "rth", "post_market", "overnight"})

# Clock-level default activity: post_market/overnight remain OFF even
# with config enabled; only pre_market + rth trade by default.
_CLOCK_ACTIVE = {
    "pre_market": True,
    "rth": True,
    "post_market": False,
    "overnight": False,
}

_PRE_START = (4, 0)
_RTH_START = (9, 30)
_RTH_END = (16, 0)
_POST_END = (20, 0)


class SessionEnabled(Protocol):
    """Anything exposing the TradingConfig session policy."""

    def is_session_active(self, session: str) -> bool:
        """Return True when the named session is enabled in config."""
        ...


def _classify_session(now_et: datetime) -> str:
    """Map an ET-aware datetime to a canonical session id."""
    minute_of_day = now_et.hour * 60 + now_et.minute

    def inside(start: tuple[int, int], end: tuple[int, int]) -> bool:
        """True when the current minute-of-day falls in [start, end)."""
        return (start[0] * 60 + start[1]) <= minute_of_day < (end[0] * 60 + end[1])

    if inside(_PRE_START, _RTH_START):
        return "pre_market"
    if inside(_RTH_START, _RTH_END):
        return "rth"
    if inside(_RTH_END, _POST_END):
        return "post_market"
    return "overnight"


@dataclass
class SleepState:
    active: bool = True
    session: str = "rth"
    reason: str = ""


class SleepManager:
    """Market session awareness."""

    def __init__(self) -> None:
        """Initialize with no forced override."""
        self._force_active: bool = False

    def _et_now(self, now: datetime | None) -> datetime:
        """Resolve the reference clock (injectable for tests)."""
        if now is not None:
            return now
        return datetime.now(timezone.utc).astimezone(_ET)

    def minutes_to_close(self) -> float:
        """Minutes until RTH close (4:00 PM ET). Negative if past close."""
        now = datetime.now(timezone.utc).astimezone(_ET)
        close = now.replace(
            hour=_RTH_END[0], minute=_RTH_END[1], second=0, microsecond=0
        )
        return (close - now).total_seconds() / 60.0

    def is_eod_window(self, minutes: float = 5.0) -> bool:
        """True if within `minutes` of RTH close on a weekday."""
        now = datetime.now(timezone.utc).astimezone(_ET)
        if now.weekday() >= 5:
            return False
        remaining = self.minutes_to_close()
        return 0 < remaining <= minutes

    def get_state(
        self, ib_connected: bool = True, now: datetime | None = None
    ) -> SleepState:
        """Clock + connectivity gate (session id is always classified)."""
        now_et = self._et_now(now)
        if self._force_active:
            return SleepState(
                active=True, session=_classify_session(now_et), reason="manual"
            )
        if now_et.weekday() >= 5:
            return SleepState(active=False, session="weekend", reason="Weekend")
        session = _classify_session(now_et)
        if not ib_connected:
            return SleepState(active=False, session=session, reason="IB disconnected")
        base = _CLOCK_ACTIVE.get(session, False)
        return SleepState(
            active=base, session=session, reason="Market open" if base else "Off hours"
        )

    def effective_state(
        self,
        enabled: SessionEnabled,
        ib_connected: bool = True,
        now: datetime | None = None,
    ) -> SleepState:
        """System gate: clock window AND the enabled-set must agree.

        This is the single gate the whole stack (bot, telemetry, HALIM)
        reads. Disabling a session via TradingConfig puts the whole
        system to sleep.
        """
        state = self.get_state(ib_connected=ib_connected, now=now)
        if state.active and not enabled.is_session_active(state.session):
            return SleepState(
                active=False, session=state.session, reason="session_disabled"
            )
        return state

    def force_active(self, active: bool) -> None:
        """Override market hours check."""
        self._force_active = active
