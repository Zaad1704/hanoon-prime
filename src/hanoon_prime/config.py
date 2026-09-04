"""config.py — Shared trading configuration for Juli.

Session toggles and direction mode. Both the telemetry API and
ib_cycle read/write these. Thread-safe via simple attribute access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TradingConfig:
    """Global trading config shared across bot components."""

    # Session toggles (all start enabled)
    session_pre_market: bool = True
    session_rth: bool = True
    session_post_market: bool = True
    session_overnight: bool = True

    # Direction mode: "both", "long_only", "short_only"
    direction_mode: str = "both"

    # EOD flatten
    eod_flatten_enabled: bool = True
    eod_flatten_minutes: float = 5.0  # minutes before close to flatten

    def is_session_active(self, session: str) -> bool:
        """Check if a session is enabled."""
        return getattr(self, f"session_{session}", True)

    def is_direction_allowed(self, side: str) -> bool:
        """Check if a trade side is allowed."""
        if self.direction_mode == "both":
            return True
        if self.direction_mode == "long_only":
            return side.upper() in ("BUY", "LONG")
        if self.direction_mode == "short_only":
            return side.upper() in ("SELL", "SHORT")
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize for telemetry."""
        return {
            "sessions": {
                "pre_market": self.session_pre_market,
                "rth": self.session_rth,
                "post_market": self.session_post_market,
                "overnight": self.session_overnight,
            },
            "direction_mode": self.direction_mode,
            "eod_flatten_enabled": self.eod_flatten_enabled,
            "eod_flatten_minutes": self.eod_flatten_minutes,
        }


# Singleton — import this everywhere
TRADING_CONFIG = TradingConfig()
