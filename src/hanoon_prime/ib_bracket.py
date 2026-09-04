"""hanoon_prime.ib_bracket — IB bracket order utilities."""
from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger(__name__)


def _is_valid_bracket(t: Any, tracked: set[str]) -> bool:
    """Check if trade has valid bracket info to extract."""
    sym = getattr(getattr(t, "contract", None), "symbol", "")
    if sym not in tracked or not hasattr(t.order, "children"):
        return False
    return bool(t.order.children)


def _extract_bracket_prices(t: Any) -> tuple[float, float] | None:
    """Extract stop and target prices from a trade's bracket orders."""
    sp = [c.auxPrice for c in t.order.children if c.auxPrice is not None]
    tp = [c.lmtPrice for c in t.order.children if c.lmtPrice is not None]
    if not sp or not tp:
        return None
    return (float(max(sp)), float(max(tp)))


def _brackets_from_trades(ib: Any, tracked: set[str], brackets: dict[str, tuple[float, float]]) -> None:
    """Update bracket levels from IB trades."""
    try:
        for t in ib.trades():
            if not _is_valid_bracket(t, tracked):
                continue
            prices = _extract_bracket_prices(t)
            if prices is None:
                continue
            sym = getattr(getattr(t, "contract", None), "symbol", "")
            brackets[sym] = prices
    except Exception as e:
        log.debug("bracket extraction failed: %s", e)


__all__ = ["_brackets_from_trades"]