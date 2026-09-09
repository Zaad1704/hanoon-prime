"""hanoon_prime.ib_order_sweep — resting-order sweep for IB protection.

Validates JULI_* OCA groups, self-heals broken ones, and clears stale
non-OCA resting orders (queued exits, orphaned bracket legs) that would
otherwise fire unintended fills at the next open.
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)

# Minimum target move (fraction of price) so OCA targets never collapse
# onto the stop leg for penny / stale-ATR instruments.
MIN_TARGET_PCT: float = 0.04

# Position side an IB closing action closes (+1 long-exit SELL, -1 short-exit BUY).
_CLOSING_SIDE: dict[str, int] = {"SELL": 1, "BUY": -1}

_ACTIVE_STATUSES = {"PendingSubmit", "PreSubmitted", "Submitted", "Active"}


def _pair_prices(trades: list[Any]) -> tuple[float | None, float | None]:
    """Extract (stop, target) from an STP+LMT pair (None when absent)."""

    def _price(order_type: str) -> float | None:
        for t in trades:
            if getattr(t.order, "orderType", "") != order_type:
                continue
            field = {"STP": "auxPrice", "LMT": "lmtPrice"}[order_type]
            return getattr(t.order, field, None)
        return None

    return _price("STP"), _price("LMT")


def _prices_sensible(action: str, stop: float | None, target: float | None) -> bool:
    """Target must be beyond the stop in the closing direction (no degeneracy).

    A long-exit (SELL) lift requires target > stop; a short-exit (BUY)
    lift requires target < stop. Pairs without prices are accepted.
    """
    if stop is None or target is None:
        return True
    if not (math.isfinite(stop) and math.isfinite(target)):
        return False
    side = _CLOSING_SIDE.get(action, 1)
    return (target - stop) * side > 0


def _is_valid_protection(trades: list[Any]) -> bool:
    """Check if OCA group has exactly STP+LMT pair with sane prices."""
    if len(trades) != 2 or {t.order.orderType for t in trades} != {"STP", "LMT"}:
        return False
    action = trades[0].order.action
    stop, target = _pair_prices(trades)
    return _prices_sensible(action, stop, target)


def sweep_zombies(ib_client: Any, known_pending: set[str] | None = None) -> None:
    """Validate JULI_* OCA groups, fix broken ones, and clear stale orders.

    Cancelled/filled OCA legs staying visible in openTrades() is valid —
    filtering them avoids an infinite SWEEP loop. ``known_pending`` marks
    acquisition parents still awaited.
    """
    try:
        all_trades = ib_client.openTrades()
    except Exception:
        return
    juli = [
        t
        for t in all_trades
        if t.order.ocaGroup
        and t.order.ocaGroup.startswith("JULI_")
        and t.orderStatus.status in _ACTIVE_STATUSES
    ]
    if juli:
        from collections import defaultdict

        groups: dict[str, list[Any]] = defaultdict(list)
        for t in juli:
            groups[t.order.ocaGroup].append(t)
        for grp, trades in groups.items():
            if _is_valid_protection(trades):
                continue
            types = {t.order.orderType for t in trades}
            log.info(
                "SWEEP %s: %d active orders (%s) — fixing",
                grp.replace("JULI_", ""),
                len(trades),
                types,
            )
            _cancel_oca(ib_client, trades)
    _sweep_stale_orders(ib_client, all_trades, known_pending)


def _held_sizes(ib_client: Any) -> dict[str, float] | None:
    """Current |position| by symbol; None when positions are unavailable.

    A None result aborts the stale-order pass, since an empty position
    map would make every queued MKT (incl. fresh full-size exits) stale.
    """
    held: dict[str, float] = {}
    try:
        for pos in ib_client.positions():
            sym = pos.contract.symbol if pos.contract else ""
            if sym:
                held[sym] = held.get(sym, 0.0) + abs(float(pos.position))
    except Exception:
        return None
    return held


def _order_qty(order: Any) -> float:
    try:
        return float(order.totalQuantity or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _sweep_stale_orders(
    ib_client: Any, all_trades: list[Any], known_pending: set[str] | None = None
) -> None:
    """Cancel stale non-OCA resting orders left by earlier cycles.

    A PreSubmitted MKT whose quantity no longer matches the current IB
    position would fire an unintended partial/oversized exit at the next
    open; a BUY limit for a flat symbol is an orphaned bracket leg.
    """
    held = _held_sizes(ib_client)
    if held is None:
        return
    for t in all_trades:
        o = t.order
        if o.ocaGroup or o.orderType not in ("MKT", "LMT"):
            continue
        sym = t.contract.symbol if t.contract else ""
        qty = _order_qty(o)
        if o.orderType == "MKT":
            stale = (
                t.orderStatus.status == "PreSubmitted"
                and abs(held.get(sym, 0.0) - qty) > 0.01
            )
            if stale:
                _cancel_order(ib_client, t, "STALE-MKT")
            continue
        if (
            _CLOSING_SIDE.get(o.action, 1) >= 0
            or t.orderStatus.status not in _ACTIVE_STATUSES
        ):
            continue
        if held.get(sym, 0.0) == 0.0 and (
            known_pending is None or sym not in known_pending
        ):
            _cancel_order(ib_client, t, "STALE-BUY")


def _cancel_order(ib_client: Any, t: Any, tag: str) -> None:
    try:
        ib_client.cancelOrder(t.order)
        sym = t.contract.symbol if t.contract else "?"
        log.info(
            "%s cancel %s %s %s", tag, sym, t.order.orderType, t.order.totalQuantity
        )
    except Exception as exc:
        log.warning("%s cancel failed: %s", tag, exc)


def _get_oca_orders(ib_client: Any, sym: str) -> list[Any]:
    """Get active trades for a JULI_* OCA group (excludes cancelled/filled)."""
    try:
        return [
            t
            for t in ib_client.openTrades()
            if t.order.ocaGroup == f"JULI_{sym}"
            and t.orderStatus.status in _ACTIVE_STATUSES
        ]
    except Exception:
        return []


def _cancel_oca(ib_client: Any, trades: list[Any]) -> None:
    for t in trades:
        try:
            ib_client.cancelOrder(t.order)
        except Exception as e:
            log.debug("cancel skip: %s", e)


__all__ = ["sweep_zombies"]
