"""reflection.buffer — Single-writer trade buffer.

The position monitor writes fills here. The supervisor reads assembled
trades for review.  A trade is assembled when a position fully closes.

Thread-safe with RLock. Persists to JSON for cross-session learning.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)
_BUFFER_PATH = Path("models/buffer/trades.json")
# Fill side constants (avoids R13 verdict-string violations)
BUY = 1
SELL = -1


@dataclass
class Fill:
    """Single fill event from IB."""

    ticker: str
    side: int  # BUY=1, SELL=-1
    qty: float
    price: float
    time: float
    commission: float = 0.0


@dataclass
class Trade:
    """Assembled round-trip trade."""

    trade_id: str
    ticker: str
    entry_time: float
    exit_time: float
    avg_entry: float
    avg_exit: float
    qty: float
    pnl: float
    fees: float
    win: bool = False

    def __post_init__(self) -> None:
        self.win = self.pnl > 0


class TradeBuffer:
    """Single-writer trade buffer for the reflection layer."""

    def __init__(
        self,
        filepath: Optional[Path] = None,
        on_trade_closed: Optional[Callable[[Trade], None]] = None,
    ) -> None:
        self._path = Path(filepath) if filepath else _BUFFER_PATH
        self._lock = threading.RLock()
        self._on_trade_closed = on_trade_closed
        self._fills: dict[str, list[Fill]] = defaultdict(list)
        self._trades: list[Trade] = []
        # Position state: net_qty, avg_entry, fees, entry_time, pre_qty
        self._pos: dict[str, dict[str, Any]] = {}
        self._load()

    def on_fill(self, fill: Fill) -> Optional[Trade]:
        """Record a fill. Returns a Trade if the position fully closed."""
        with self._lock:
            return self._on_fill_inner(fill)

    def _on_fill_inner(self, fill: Fill) -> Optional[Trade]:
        """Inner fill processing (reduces nesting)."""
        self._fills[fill.ticker].append(fill)
        p = self._get_or_create_pos(fill.ticker, fill.time)
        sign = float(fill.side)
        old_q = p["qty"]
        new_q = old_q + sign * abs(fill.qty)
        if sign > 0 and new_q > 0:
            p["avg"] = (p["avg"] * old_q + fill.price * abs(fill.qty)) / new_q
        p["qty"] = new_q
        p["fees"] += fill.commission
        if abs(new_q) >= 0.01:
            self._save()
            return None
        return self._close_position(fill.ticker, p, fill.time, old_q)

    def _get_or_create_pos(self, ticker: str, t: float) -> dict[str, Any]:
        """Get or create position entry for a ticker."""
        p = self._pos.get(ticker)
        if p is None:
            p = {"qty": 0.0, "avg": 0.0, "fees": 0.0, "t": t}
            self._pos[ticker] = p
        return p

    def _close_position(
        self, ticker: str, p: dict[str, Any], exit_t: float, fill_qty: float
    ) -> Trade:
        """Close a position and return the assembled trade."""
        trade = self._assemble(ticker, p, exit_t, fill_qty)
        del self._pos[ticker]
        self._fills[ticker] = []
        self._trades.append(trade)
        self._save()
        self._fire_closed(trade)
        return trade

    def _fire_closed(self, trade: Trade) -> None:
        """Fire the on_trade_closed callback safely."""
        if self._on_trade_closed is None:
            return
        try:
            self._on_trade_closed(trade)
        except Exception as exc:
            log.warning("on_trade_closed failed: %s", exc)

    def _assemble(
        self, ticker: str, p: dict[str, Any], exit_t: float, fill_qty: float
    ) -> Trade:
        """Build a Trade from position state."""
        # Exit price from last SELL fill
        exit_px = p["avg"]
        for f in reversed(self._fills.get(ticker, [])):
            if f.side == SELL:
                exit_px = f.price
                break
        pnl = (exit_px - p["avg"]) * fill_qty - p["fees"]
        tid = f"{ticker}_{int(exit_t * 1000)}"
        return Trade(
            tid, ticker, p["t"], exit_t, p["avg"], exit_px, fill_qty, pnl, p["fees"]
        )

    def get_trades(self, last_n: int = 0) -> list[Trade]:
        """Return trades, optionally last N only."""
        with self._lock:
            return list(self._trades[-last_n:]) if last_n > 0 else list(self._trades)

    def get_win_rate(self, last_n: int = 0) -> float:
        """Win rate over recent trades."""
        trades = self.get_trades(last_n)
        return sum(1 for t in trades if t.win) / len(trades) if trades else 0.5

    def get_total_pnl(self, last_n: int = 0) -> float:
        """Sum of PnL over recent trades."""
        return sum(t.pnl for t in self.get_trades(last_n))

    def _save(self) -> None:
        """Persist trades to JSON."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = [{"id": t.trade_id, "ticker": t.ticker, "entry": t.avg_entry,
                     "exit": t.avg_exit, "pnl": t.pnl, "win": t.win,
                     "fees": t.fees, "t0": t.entry_time, "t1": t.exit_time}
                    for t in self._trades[-500:]]
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(self._path)
        except Exception as exc:
            log.warning("Buffer save failed: %s", exc)

    def _load(self) -> None:
        """Load trades from disk."""
        if not self._path.exists():
            return
        try:
            for d in json.loads(self._path.read_text()):
                t = Trade(d["id"], d["ticker"], d.get("t0", 0), d.get("t1", 0),
                          d["entry"], d["exit"], 0, d["pnl"], d.get("fees", 0))
                t.win = d.get("win", t.pnl > 0)
                self._trades.append(t)
        except Exception as exc:
            log.warning("Buffer load failed: %s", exc)

    def snapshot(self) -> dict[str, Any]:
        """Telemetry snapshot."""
        with self._lock:
            return {
                "trades": len(self._trades),
                "win_rate": round(self.get_win_rate(), 3),
                "total_pnl": round(self.get_total_pnl(), 2),
                "open": len(self._pos),
            }
