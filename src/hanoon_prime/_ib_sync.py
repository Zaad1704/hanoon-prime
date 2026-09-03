"""hanoon_prime._ib_sync — IB state reading helpers.
Pure functions that read state from IB. Used by ib_executor.py
to keep that file under the 200-line R3 limit.
IB is the single source of truth — these functions only READ from IB.
"""
from __future__ import annotations

import time
from typing import Any

from .memory import Journal
from .types import Position


def get_ib_pnl(ib_client: Any, ticker: str, pos: Position) -> float:
    """Get P&L from IB fills (source of truth), not local calculation."""
    pnl = _pnl_from_trade(ib_client, ticker)
    if pnl != 0.0:
        return pnl
    return _pnl_from_fill_price(ib_client, ticker, pos)


def _pnl_from_trade(ib_client: Any, ticker: str) -> float:
    """Extract P&L from IB's completed trade objects."""
    try:
        for t in ib_client.trades():
            sym = getattr(getattr(t, "contract", None), "symbol", "")
            if sym == ticker and t.isDone() and t.pnl:
                return float(t.pnl)
    except Exception:
        pass
    return 0.0


def _get_fill_price(trade: Any) -> float:
    """Extract fill price from an IB trade object."""
    return float(trade.order.auxPrice or trade.order.lmtPrice or 0)


def _pnl_from_fill_price(ib_client: Any, ticker: str, pos: Position) -> float:
    """Fallback: compute P&L from IB's fill prices (still IB data)."""
    try:
        for t in ib_client.trades():
            sym = getattr(getattr(t, "contract", None), "symbol", "")
            if sym == ticker and t.isDone():
                fill = _get_fill_price(t)
                return _calc_pnl_from_fill(fill, pos)
    except Exception:
        pass
    return 0.0


def _calc_pnl_from_fill(fill: float, pos: Position) -> float:
    """Calculate P&L from a fill price."""
    if fill > 0 and pos.entry_price > 0:
        return (fill - pos.entry_price) / pos.entry_price * pos.direction
    return 0.0


def read_ib_orders(ib_client: Any) -> list[dict[str, Any]]:
    """Read OPEN orders from IB for journal carbon copy (done ones excluded)."""
    orders: list[dict[str, Any]] = []
    try:
        for trade in ib_client.trades():
            if trade.isDone():
                continue  # filled/cancelled orders are no longer open
            o = trade.order
            if o and not getattr(o, "parentId", None):
                orders.append(
                    {
                        "orderId": o.orderId,
                        "action": o.action,
                        "totalQuantity": o.totalQuantity,
                        "lmtPrice": getattr(o, "lmtPrice", None),
                        "auxPrice": getattr(o, "auxPrice", None),
                    }
                )
    except Exception:
        pass
    return orders


def read_ib_positions(
    ib_client: Any,
    tracked: set[str],
    brackets: dict[str, tuple[float, float]],
) -> dict[str, Position]:
    """Query IB for live positions in tracked tickers."""
    result: dict[str, Position] = {}
    try:
        for pos in ib_client.positions():
            if pos.contract.symbol not in tracked:
                continue
            sym = pos.contract.symbol
            b = brackets.get(sym, (0.0, 0.0))
            result[sym] = Position(
                ticker=sym,
                entry_idx=-1,
                entry_price=pos.avgCost,
                shares=abs(pos.position),
                direction=1 if pos.position > 0 else -1,
                stop_price=b[0],
                target_price=b[1],
                peak_price=getattr(pos, "marketPrice", 0.0) or 0.0,
                score=0.0,
                atr=0.0,
            )
    except Exception:
        pass
    return result


def read_portfolio(ib_client: Any) -> dict[str, Any]:
    """Read portfolio from IB — market value, unrealized P&L, % change."""
    result: dict[str, Any] = {}
    try:
        for item in ib_client.portfolio():
            sym = item.contract.symbol
            mv = item.marketValue
            result[sym] = {
                "shares": item.position,
                "value": mv,
                "pnl": item.unrealizedPNL,
                "pct": (item.unrealizedPNL / abs(mv) * 100) if mv else 0.0,
            }
    except Exception:
        pass
    return result


def read_account_summary(ib_client: Any) -> dict[str, str]:
    """Read account summary — NetLiq, BuyingPower, CashBalance, etc."""
    result: dict[str, str] = {}
    try:
        for item in ib_client.accountSummary():
            result[item.tag] = item.value
    except Exception:
        pass
    return result


def journal_snapshot(
    journal: Journal,
    ib_client: Any,
    ib_positions: dict[str, Position],
    brackets: dict[str, Any],
) -> None:
    """Carbon-copy journal: snapshot IB state every cycle."""
    ib_orders = read_ib_orders(ib_client)
    portfolio = read_portfolio(ib_client)
    account = read_account_summary(ib_client)
    journal.append(
        {
            "event": "ib_state_snapshot",
            "source": "ib_gateway",
            "positions": {
                s: {"shares": p.shares * p.direction, "avg_cost": p.entry_price}
                for s, p in ib_positions.items()
            },
            "open_orders": ib_orders,
            "brackets": {s: list(v) for s, v in brackets.items()},
            "portfolio": portfolio,
            "account_summary": {
                "net_liq": account.get("NetLiquidation", "?"),
                "buying_power": account.get("BuyingPower", "?"),
                "cash": account.get("CashBalance", "?"),
            },
            "timestamp": time.time(),
        }
    )


def journal_exit(journal: Journal, ticker: str, pnl: float, pos: Position) -> None:
    """Carbon-copy exit to journal (IB fill data)."""
    journal.append(
        {
            "event": "position_closed",
            "source": "ib_gateway",
            "ticker": ticker,
            "pnl": pnl,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "shares": pos.shares,
            "timestamp": time.time(),
        }
    )
