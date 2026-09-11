"""brain.policy.trading_policy — decision parts of the trading configuration.

Direction mode, session enablement, and the sub-dollar confidence bar.
The singleton is the SAME object ``config.TRADING_CONFIG`` re-exports, so
telemetry and existing code share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...immune import PENNY_PRICE, PENNY_SCORE_BAR


@dataclass
class TradingConfig:
    """Decision-relevant trading configuration (brain-owned)."""

    # Session toggles (all start enabled).
    session_pre_market: bool = True
    session_rth: bool = True
    session_post_market: bool = True
    session_overnight: bool = True

    # Direction mode: "both", "long_only", "short_only".
    direction_mode: str = "long_only"

    # EOD flatten (user command kept in the orchestration layer).
    eod_flatten_enabled: bool = True
    eod_flatten_minutes: float = 5.0

    # Manual flatten order type: "market" for guaranteed fills,
    # "limit" for price protection (user picks via /flatten API).
    flatten_order_type: str = "market"

    # Horizons — scalp-only by default.
    horizons: set[str] = field(default_factory=lambda: {"scalp"})

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

    def is_penny_bar_cleared(
        self, _ticker: str, price: float, score: float
    ) -> tuple[bool, str]:
        """Raise-the-bar for sub-dollar tickers: (False, reason) unless the
        brain is EXTREMELY sure of a micro-cap scalp (a higher bar the score
        can still clear — not a hard price block)."""
        if price < PENNY_PRICE and abs(score) < PENNY_SCORE_BAR:
            return False, "low_penny_score"
        return True, ""

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
            "flatten_order_type": self.flatten_order_type,
            "horizons": sorted(self.horizons),
        }


# Singleton — import this everywhere (config.py re-exports it).
TRADING_CONFIG = TradingConfig()

__all__ = ["TradingConfig", "TRADING_CONFIG"]
