"""hanoon_prime._ib_marks — live positions marking helpers.

ib_insync's Position carries only (account, contract, position, avgCost);
the real-time mark lives on PortfolioItem (ib.portfolio()) or the live
Ticker (ib.tickers). These helpers lift a truthful market-price and
unrealized-PnL surface off the IB client.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ._ib_sync import read_portfolio

log = logging.getLogger(__name__)

ZERO_PNL = {"positions": [], "total_pnl": 0.0, "count": 0}


def feed_positions(ib_client: Any) -> dict[str, Any]:
    """Portfolio map, falling back to raw holdings when the feed is silent."""
    positions = read_portfolio(ib_client)
    if positions:
        return positions
    return {
        str(pos.contract.symbol): {
            "shares": pos.position,
            "value": pos.position * pos.avgCost,
            "pnl": 0.0,
            "pct": 0.0,
        }
        for pos in ib_client.positions()
        if pos.position
    }


def _number(value: Any) -> float:
    """Finite float or 0.0 — IB surfaces nan/inf/None marks, never trust them."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return converted if math.isfinite(converted) else 0.0


def _tick_price(tick: Any) -> float:
    """Best live price from an ib_insync Ticker (last/close/bid/ask)."""
    if tick is None:
        return 0.0
    for attr in ("last", "close", "bid", "ask"):
        value = _number(getattr(tick, attr, 0.0))
        if value:
            return value
    return 0.0


def _portfolio_marks(ib_client: Any) -> dict[str, Any]:
    """Symbol -> PortfolioItem map from IB's portfolio feed."""
    result: dict[str, Any] = {}
    try:
        for item in ib_client.portfolio():
            contract = getattr(item, "contract", None)
            if contract is not None:
                result[contract.symbol] = item
    except Exception as exc:
        log.debug("portfolio marks unavailable: %s", exc)
    return result


def _live_tickers(ib_client: Any) -> dict[str, Any]:
    """Symbol -> Ticker map from IB's active market data (property, not call)."""
    result: dict[str, Any] = {}
    try:
        for tick in ib_client.tickers:
            contract = getattr(tick, "contract", None)
            if contract is not None:
                result[contract.symbol] = tick
    except Exception as exc:
        log.debug("live tickers unavailable: %s", exc)
    return result


def _snapshot_mark(symbol: str, entry: float, get_snapshot: Any) -> tuple[float, bool]:
    """4th-tier mark: streamer bar-close price for penny stocks.

    Returns ``(price, ok)`` where ``ok`` is False when no usable close
    price could be obtained.  Never raises.
    """
    try:
        snap = get_snapshot(symbol)
    except Exception as exc:
        log.debug("snapshot mark failed for %s: %s", symbol, exc)
        return 0.0, False
    if not snap:
        return 0.0, False
    prices = snap.get("prices")
    if not prices:
        return 0.0, False
    close_price = _number(prices[-1])
    if close_price <= 0:
        return 0.0, False
    return close_price, True


def _mark_position(
    pos: Any,
    port: dict[str, Any],
    live: dict[str, Any],
    get_snapshot: Any = None,
) -> dict[str, Any] | None:
    """One live-marked position row, or None for a zero-quantity stub."""
    symbol = pos.contract.symbol
    entry = _number(pos.avgCost)
    shares = abs(_number(pos.position))
    if not shares:
        return None
    item = port.get(symbol)
    direction = 1.0 if pos.position > 0 else -1.0
    if item is not None and _number(item.marketPrice):
        market = _number(item.marketPrice)
        unrealized = _number(item.unrealizedPNL)
    else:
        market = _tick_price(live.get(symbol)) or entry
        unrealized = (market - entry) * shares * direction
        if market == entry and get_snapshot is not None:
            bar_price, ok = _snapshot_mark(symbol, entry, get_snapshot)
            if ok:
                market = bar_price
                unrealized = (market - entry) * shares * direction
    return {
        "ticker": symbol,
        "entry_price": round(entry, 2),
        "shares": int(shares),
        "direction": "LONG" if pos.position > 0 else "SHORT",
        "market_price": round(market, 2),
        "unrealized_pnl": round(unrealized, 2),
        "pnl_pct": round(((market - entry) / entry) * 100, 2) if entry > 0 else 0.0,
    }


def mark_positions(
    ib_client: Any,
    get_snapshot: Any = None,
) -> dict[str, Any]:
    """Live positions surface: portfolio marks with live-ticker fallback.

    ib_insync Position has no marketPrice/unrealizedPNL; PortfolioItem does.
    When the portfolio feed lags, the live Ticker supplies the mark so
    per-position PnL stays truthful instead of reading as entry=$0.

    ``get_snapshot`` is an optional callable returning the latest snapshot
    dict for a symbol.  Used as a 4th-tier mark fallback for penny stocks
    whose IB portfolio item *and* live ticker both carry stale/zero prices.
    """
    port = _portfolio_marks(ib_client)
    live = _live_tickers(ib_client)
    try:
        raw_positions = ib_client.positions()
    except Exception:
        return dict(ZERO_PNL)
    out = [
        row
        for row in (_mark_position(p, port, live, get_snapshot) for p in raw_positions)
        if row
    ]
    return {
        "positions": out,
        "total_pnl": round(sum(x["unrealized_pnl"] for x in out), 2),
        "count": len(out),
    }
