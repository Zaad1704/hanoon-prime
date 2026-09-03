"""hanoon_prime._protect — Position protection for IB.

Places OCA stop+target legs over existing IB positions that lack
bracket protection. Called each sync cycle by ib_executor.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .ib_compat import ib as _ib
from .immune import ATR_STOP_MULT, ATR_TARGET_MULT

log = logging.getLogger(__name__)


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
    kw = dict(action=action, totalQuantity=qty, tif="GTC", ocaGroup=oca, ocaType=1)
    ib_client.placeOrder(
        contract, _ib.Order(orderType="STP", auxPrice=stop, transmit=False, **kw)
    )
    ib_client.placeOrder(contract, _ib.Order(orderType="LMT", lmtPrice=target, **kw))


def protect_position(
    ib_client: Any,
    tracked: set[str],
    brackets: dict[str, tuple[float, float]],
    pending: set[str],
    streamer: Any,
) -> None:
    """Place OCA stop+target legs over unprotected IB positions."""
    try:
        ocas = {t.order.ocaGroup for t in ib_client.openTrades() if t.order.ocaGroup}
    except Exception:
        ocas = set()
    for pos in ib_client.positions():
        sym = pos.contract.symbol if pos.contract else ""
        if sym not in tracked or sym in pending or sym in brackets:
            continue
        if f"JULI_{sym}" in ocas:
            continue
        px = streamer.get_last_price(sym)
        atr = streamer.buffer_atr(sym)
        if not px or atr <= 0.0 or math.isnan(atr):
            continue
        try:
            c = ib_client.qualifyContracts(_ib.Stock(sym, "SMART", "USD"))[0]
        except Exception:
            continue
        d = 1 if pos.position > 0 else -1
        stop = round(px - d * ATR_STOP_MULT * atr, 2)
        target = round(px + d * ATR_TARGET_MULT * atr, 2)
        action = "SELL" if d > 0 else "BUY"
        oca, qty = f"JULI_{sym}", abs(pos.position)
        try:
            _place_oca(ib_client, c, action, qty, stop, target, oca)
            brackets[sym] = (stop, target)
            tag = "L" if d > 0 else "S"
            log.info(f"ADOPT {tag} {sym} q={qty}")
        except Exception as e:
            log.warning(f"ADOPT fail {sym}: {e}")
