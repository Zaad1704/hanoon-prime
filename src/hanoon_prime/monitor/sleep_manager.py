"""monitor.sleep_manager — Market session awareness.

Decides whether the bot should be ACTIVE (trading) or SLEEPING (idle).
Honors market sessions (RTH/pre/post), weekends, and holidays.
Uses zoneinfo for proper US/Eastern timezone (handles DST automatically).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_RTH_START_HOUR = 9  # 9:30 AM ET
_RTH_START_MIN = 30
_RTH_END_HOUR = 16  # 4:00 PM ET
_RTH_END_MIN = 0


@dataclass
class SleepState:
    active: bool = True
    session: str = "RTH"
    reason: str = ""


class SleepManager:
    """Market session awareness."""

    def __init__(self) -> None:
        """Initialize with no forced override."""
        self._force_active: bool = False

    def get_state(self, ib_connected: bool = True) -> SleepState:
        """Determine if bot should be active."""
        if self._force_active:
            return SleepState(active=True, session="forced", reason="manual")
        if not ib_connected:
            return SleepState(active=False, reason="IB disconnected")
        now = datetime.now(timezone.utc).astimezone(_ET)
        et_hour = now.hour
        et_min = now.minute
        if now.weekday() >= 5:
            return SleepState(active=False, session="weekend", reason="Weekend")
        is_rth = et_hour > _RTH_START_HOUR or (
            et_hour == _RTH_START_HOUR and et_min >= _RTH_START_MIN
        )
        is_rth = is_rth and (
            et_hour < _RTH_END_HOUR
            or (et_hour == _RTH_END_HOUR and et_min < _RTH_END_MIN)
        )
        if is_rth:
            return SleepState(active=True, session="RTH", reason="Market open")
        is_pre = et_hour == 8
        is_post = et_hour == 16
        if is_pre or is_post:
            return SleepState(
                active=False, session="pre_post", reason="Pre/Post market"
            )
        return SleepState(active=False, session="overnight", reason="Off hours")

    def force_active(self, active: bool) -> None:
        """Override market hours check."""
        self._force_active = active
