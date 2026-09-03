"""hanoon_prime._protect — Position protection for IB.

Places and validates OCA stop+target legs over existing IB positions.
Self-heals broken protection on every sync cycle.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .ib_compat import ib as _ib
from .immune import ATR_STOP_MULT, ATR_TARGET_MULT

log = logging.getLogger(__name__)


def sweep_zombies(ib_client: Any) -> None:
    """Cancel all JULI_* OCA orders on startup (zombie cleanup)."""
    try:
        zombies = [
            t
            for t in ib_client.openTrades()
            if t.order.ocaGroup and t.order.ocaGroup.startswith("JULI_")
        ]
    except Exception:
        return
    if not zombies:
        return
    log.info("SWEEP: cancelling %d zombie JULI_* orders", len(zombies))
    for t in zombies:
        try:
            ib_client.cancelOrder(t.order)
        except Exception as e:
            log.debug("sweep cancel skip: %s", e)


def _get_oca_orders(ib_client: Any, sym: str) -> list[Any]:
    """Get all open trades for a JULI_* OCA group."""
    try:
        return [t for t in ib_client.openTrades() if t.order.ocaGroup == f"JULI_{sym}"]
    except Exception:
        return []


def _cancel_oca(ib_client: Any, trades: list[Any]) -> None:
    for t in trades:
        try:
            ib_client.cancelOrder(t.order)
        except Exception as e:
            log.debug("cancel skip: %s", e)


def _validate_protection(
    trades: list[Any], expected_qty: int, expected_action: str
) -> bool:
    """Check if OCA pair is correctly composed."""
    if len(trades) != 2:
        return False
    types = {t.order.orderType for t in trades}
    if types != {"STP", "LMT"}:
        return False
    for t in trades:
        if t.order.action != expected_action:
            return False
        if abs(t.order.totalQuantity - expected_qty) > 0.01:
            return False
    return True


def _place_oca(
    ib_client: Any,
    contract: Any,
    action: str,
    qty: int,
    stop: float,
    target: float,
    oca: str,
) -> None:
    """Place OCA STP+LMT pair as position protection."""
    kw = dict(
        action=action,
        totalQuantity=qty,
        tif="GTC",
        ocaGroup=oca,
        ocaType=1,
        transmit=True,
    )
    ib_client.placeOrder(contract, _ib.Order(orderType="STP", auxPrice=stop, **kw))
    ib_client.placeOrder(contract, _ib.Order(orderType="LMT", lmtPrice=target, **kw))


def protect_position(
    ib_client: Any,
    tracked: set[str],
    brackets: dict[str, tuple[float, float]],
    pending: set[str],
    streamer: Any,
) -> None:
    """Validate and fix OCA protection for all tracked positions."""
    from .immune import ATR_STOP_MULT, ATR_TARGET_MULT

    for pos in ib_client.positions():
        sym = pos.contract.symbol if pos.contract else ""
        if sym not in tracked or sym in pending:
            continue
        d = 1 if pos.position > 0 else -1
        expected_qty, expected_action = abs(pos.position), "SELL" if d > 0 else "BUY"
        trades = _get_oca_orders(ib_client, sym)
        if _validate_protection(trades, expected_qty, expected_action):
            continue
        if trades:
            _cancel_oca(ib_client, trades)
            log.info(f"HEAL {sym}: {len(trades)} broken orders cancelled")
        px = streamer.get_last_price(sym)
        atr = streamer.buffer_atr(sym)
        if not px or atr <= 0.0 or math.isnan(atr):
            continue
        try:
            c = ib_client.qualifyContracts(_ib.Stock(sym, "SMART", "USD"))[0]
        except Exception:
            continue
        stop = round(px - d * ATR_STOP_MULT * atr, 2)
        target = round(px + d * ATR_TARGET_MULT * atr, 2)
        oca = f"JULI_{sym}"
        try:
            _place_oca(ib_client, c, expected_action, expected_qty, stop, target, oca)
            brackets[sym] = (stop, target)
            tag = "L" if d > 0 else "S"
            log.info(f"ADOPT {tag} {sym} q={expected_qty} s={stop} t={target}")
        except Exception as e:
            log.warning("ADOPT fail %s: %s", sym, e)
