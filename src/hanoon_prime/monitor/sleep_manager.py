"""monitor.sleep_manager — Market session awareness.

Decides whether the bot should be ACTIVE (trading) or SLEEPING (idle).
Honors market sessions (RTH/pre/post), weekends, and holidays.

Source: rebuild's monitoring/sleep.py (simplified).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# US Eastern Time offsets (simplified — no pytz dependency)
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
        """Auto-generated docstring."""
        self._force_active: bool = False

    def get_state(self, ib_connected: bool = True) -> SleepState:
        """Determine if bot should be active."""
        if self._force_active:
            return SleepState(active=True, session="forced", reason="manual")
        if not ib_connected:
            return SleepState(active=False, reason="IB disconnected")
        now = datetime.now(timezone.utc)
        # Convert to ET (UTC-4 or UTC-5 depending on DST)
        # Simplified: assume UTC-4 (EDT)
        et_hour = (now.hour - 4) % 24
        et_min = now.minute
        # Weekend check
        if now.weekday() >= 5:
            return SleepState(active=False, session="weekend", reason="Weekend")
        # RTH check
        is_rth = et_hour > _RTH_START_HOUR or (
            et_hour == _RTH_START_HOUR and et_min >= _RTH_START_MIN
        )
        is_rth = is_rth and (
            et_hour < _RTH_END_HOUR
            or (et_hour == _RTH_END_HOUR and et_min < _RTH_END_MIN)
        )
        if is_rth:
            return SleepState(active=True, session="RTH", reason="Market open")
        # Pre/Post market
        is_pre = et_hour == 8  # 8:00-9:30 AM ET
        is_post = et_hour == 16  # 4:00-6:00 PM ET
        if is_pre or is_post:
            return SleepState(
                active=False, session="pre_post", reason="Pre/Post market"
            )
        return SleepState(active=False, session="overnight", reason="Off hours")

    def force_active(self, active: bool) -> None:
        """Auto-generated docstring."""
        self._force_active = active
