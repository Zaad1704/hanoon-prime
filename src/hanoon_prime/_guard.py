"""hanoon_prime._guard — netting-reversal guard for long-only mode.

PAPER (netting) accounts: a resting SELL protective leg can fill beyond
the held long and flip the net position negative (an accidental short).
While ``direction_mode == "long_only"`` the bot never opens shorts on
purpose, so any negative tracked position is a reversal that must be
flattened. This module provides the pure IB reads/cancels for that guard.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def short_positions(ib_client: Any, tracked: set[str]) -> dict[str, float]:
    """Return {ticker: abs_shares} for tracked net-short positions."""
    shorts: dict[str, float] = {}
    try:
        for pos in ib_client.positions():
            sym = getattr(getattr(pos, "contract", None), "symbol", "")
            if sym in tracked and pos.position < 0:
                shorts[sym] = abs(float(pos.position))
    except Exception as exc:
        log.debug("short_positions scan failed: %s", exc)
    return shorts


def open_order_for(ib_client: Any, sym: str, action: str) -> bool:
    """True if a non-done order of ``action`` is live for ``sym``."""
    try:
        for t in ib_client.openTrades():
            o = getattr(t, "order", None)
            if not o or t.isDone():
                continue
            if getattr(getattr(t, "contract", None), "symbol", "") != sym:
                continue
            if getattr(o, "action", "") == action:
                return True
    except Exception as exc:
        log.debug("open_order_for %s failed: %s", sym, exc)
    return False


_SELL = "SELL"


def _cancel_matching(ib_client: Any, t: Any, sym: str, action: str) -> int:
    """Cancel one live order; return 1 if it matched and was cancelled."""
    o = getattr(t, "order", None)
    if not o or t.isDone():
        return 0
    if getattr(getattr(t, "contract", None), "symbol", "") != sym:
        return 0
    if getattr(o, "action", "") != action:
        return 0
    try:
        ib_client.cancelOrder(o)
        return 1
    except Exception as exc:
        log.debug("cancel leg %s %s failed: %s", sym, o.action, exc)
        return 0


def cancel_sell_legs(ib_client: Any, sym: str) -> int:
    """Cancel all live SELL orders on ``sym`` (stale stop/target legs)."""
    cancelled = 0
    try:
        for t in list(ib_client.openTrades()):
            cancelled += _cancel_matching(ib_client, t, sym, _SELL)
    except Exception as exc:
        log.debug("cancel_sell_legs %s failed: %s", sym, exc)
    return cancelled


__all__ = ["short_positions", "open_order_for", "cancel_sell_legs"]
