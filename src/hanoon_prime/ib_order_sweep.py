"""hanoon_prime.ib_order_sweep — resting-order sweep for IB protection.

Validates JULI_* OCA groups, self-heals broken ones, clears stale
non-OCA resting orders, and cancels orphaned bracket legs.
"""
from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)
MIN_TARGET_PCT: float = 0.04
_CLOSING_SIDE: dict[str, int] = {"SELL": 1, "BUY": -1}
_ACTIVE_STATUSES = {"PendingSubmit", "PreSubmitted", "Submitted", "Active"}


def _pair_prices(trades: list[Any]) -> tuple[float | None, float | None]:
    """Extract (stop, target) from an STP+LMT pair."""

    def _price(order_type: str) -> float | None:
        for t in trades:
            if getattr(t.order, "orderType", "") != order_type:
                continue
            field = {"STP": "auxPrice", "LMT": "lmtPrice"}[order_type]
            return getattr(t.order, field, None)
        return None

    return _price("STP"), _price("LMT")


def _prices_sensible(action: str, stop: float | None, target: float | None) -> bool:
    """Target must be beyond the stop in the closing direction."""
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
    """Validate JULI_* OCA groups, fix broken ones, and clear stale/orphaned orders."""
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
    _sweep_orphan_brackets(ib_client, all_trades, known_pending)


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
    """Cancel stale non-OCA resting orders left by earlier cycles."""
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


def _sweep_orphan_brackets(
    ib_client: Any, all_trades: list[Any], known_pending: set[str] | None = None
) -> None:
    """Cancel bracket children whose parent never filled (penny-stock defense)."""
    held = _held_sizes(ib_client)
    if held is None:
        return
    pending = known_pending or set()
    for t in all_trades:
        o = t.order
        if o.ocaGroup and o.ocaGroup.startswith("JULI_"):
            continue
        if t.orderStatus.status not in _ACTIVE_STATUSES:
            continue
        sym = t.contract.symbol if t.contract else ""
        if not sym or sym in pending:
            continue
        is_leg = o.orderType in ("LMT", "STP", "STPLMT") and (
            o.parentId > 0 or (o.ocaGroup and not o.ocaGroup.startswith("JULI_"))
        )
        if is_leg and held.get(sym, 0.0) == 0.0:
            _cancel_order(ib_client, t, "ORPHAN-BRACKET")


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
