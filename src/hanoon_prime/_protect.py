"""hanoon_prime._protect — Position protection for IB.

Places and validates OCA stop+target legs over existing IB positions.
Self-heals broken protection on every sync cycle. The resting-order
sweep machinery lives in ``ib_order_sweep`` (R3b file cap).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .ib_compat import ib as _ib
from .ib_order_sweep import (
    MIN_TARGET_PCT,
    _cancel_oca,
    _get_oca_orders,
    _pair_prices,
    _prices_sensible,
    sweep_zombies,
)
from .immune import ALLOW_EXTENDED_HOURS
from .types import BracketOrder

log = logging.getLogger(__name__)


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
    stop, target = _pair_prices(trades)
    if not _prices_sensible(expected_action, stop, target):
        return False
    return True


def _place_oca(ib_client: Any, contract: Any, order: BracketOrder) -> None:
    """Place OCA STP+LMT pair as position protection."""
    if math.isnan(order.stop) or math.isnan(order.target):
        log.warning(
            "OCA skip %s: NaN stop=%.4f target=%.4f",
            order.oca,
            order.stop,
            order.target,
        )
        return
    kw = dict(
        action=order.action,
        totalQuantity=order.qty,
        tif="DAY",
        ocaGroup=order.oca,
        ocaType=1,
        transmit=True,
        outsideRth=ALLOW_EXTENDED_HOURS,
    )
    ib_client.placeOrder(
        contract, _ib.Order(orderType="STP", auxPrice=order.stop, **kw)
    )
    ib_client.placeOrder(
        contract, _ib.Order(orderType="LMT", lmtPrice=order.target, **kw)
    )


def _cancel_flat_legs(ib_client: Any, sym: str, pending: set[str]) -> None:
    """Cancel lingering OCA legs on a now-flat tracked position."""
    if sym in pending:
        return
    trades = _get_oca_orders(ib_client, sym)
    if not trades:
        return
    _cancel_oca(ib_client, trades)
    log.info("CANCEL-LEGS %s: position flat", sym)


def protect_position(
    ib_client: Any,
    tracked: set[str],
    brackets: dict[str, tuple[float, float]],
    pending: set[str],
    streamer: Any,
) -> None:
    """Validate and fix OCA protection for all tracked positions.

    Flat tracked positions get any lingering SELL legs cancelled (their
    stale GTC legs previously over-sold the position on netting accounts
    and flipped it short). Negative (accidental short) positions are left
    alone for the netting guard to flatten.
    """
    live = {}
    for pos in ib_client.positions():
        sym = pos.contract.symbol if pos.contract else ""
        if sym:
            live[sym] = pos
    for sym in tracked:
        pos = live.get(sym)
        if pos is None or abs(int(pos.position)) == 0:
            _cancel_flat_legs(ib_client, sym, pending)
            continue
        if pos.position < 0:
            continue
        if sym in pending:
            continue
        if sym in brackets:
            exp_action = "SELL" if pos.position > 0 else "BUY"
            if _validate_protection(
                _get_oca_orders(ib_client, sym), abs(pos.position), exp_action
            ):
                continue
            # Stale bracket entry — protection died. Clear and re-protect.
            brackets.pop(sym, None)
        _reprotect_position(ib_client, sym, pos, streamer, brackets)


def _atr_levels(px: float, atr: float, d: int) -> tuple[float, float]:
    """Compute ATR stop/target for the position direction.

    Floors at $0.01 and enforces a minimum target move (plus a stop
    safety tick) so degenerate pairs (target == stop) are never created
    for penny stocks or stale/zero ATR.
    """
    from .immune import ATR_STOP_MULT, ATR_TARGET_MULT

    min_reward = max(0.01, round(abs(px) * MIN_TARGET_PCT, 3))
    stop_tick = max(0.005, round(min_reward * 0.5, 3))
    stop = round(px - d * ATR_STOP_MULT * atr, 2)
    target = round(px + d * ATR_TARGET_MULT * atr, 2)
    if d > 0:
        stop = min(stop, round(px - stop_tick, 2))
        target = max(target, round(px + min_reward, 2))
    else:
        stop = max(stop, round(px + stop_tick, 2))
        target = min(target, round(px - min_reward, 2))
    return max(0.01, stop), max(0.01, target)


def _reprotect_position(
    ib_client: Any,
    sym: str,
    pos: Any,
    streamer: Any,
    brackets: dict[str, tuple[float, float]],
) -> None:
    """Heal or adopt OCA protection for one tracked position."""
    d = 1 if pos.position > 0 else -1
    expected_qty, expected_action = abs(pos.position), "SELL" if d > 0 else "BUY"
    trades = _get_oca_orders(ib_client, sym)
    if _validate_protection(trades, expected_qty, expected_action):
        return
    if trades:
        _cancel_oca(ib_client, trades)
        log.info(f"HEAL {sym}: {len(trades)} broken orders cancelled")
    px = streamer.get_last_price(sym)
    atr = streamer.buffer_atr(sym)
    if not px or atr <= 0.0 or math.isnan(atr) or math.isnan(px):
        return
    try:
        c = ib_client.qualifyContracts(_ib.Stock(sym, "SMART", "USD"))[0]
    except Exception as exc:
        log.debug("qualify skip %s: %s", sym, exc)
        return
    stop, target = _atr_levels(px, atr, d)
    try:
        _place_oca(
            ib_client,
            c,
            BracketOrder(expected_action, expected_qty, stop, target, f"JULI_{sym}"),
        )
        brackets[sym] = (stop, target)
        tag = "L" if d > 0 else "S"
        log.info(f"ADOPT {tag} {sym} q={expected_qty} s={stop} t={target}")
    except Exception as e:
        log.warning("ADOPT fail %s: %s", sym, e)


__all__ = ["sweep_zombies", "protect_position"]
