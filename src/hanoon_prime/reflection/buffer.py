"""reflection.buffer — Single-writer trade buffer (RLock, JSON-persisted)."""

from __future__ import annotations

import json
import logging
import threading
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
    ticker: str
    side: int  # BUY=1, SELL=-1
    qty: float
    price: float
    time: float
    commission: float = 0.0


@dataclass
class Trade:
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
        self._trades: list[Trade] = []
        self._pos: dict[str, dict[str, Any]] = {}
        self._load()

    def on_fill(self, fill: Fill) -> Optional[Trade]:
        """Record a fill. Returns a Trade if the position fully closed."""
        with self._lock:
            return self._on_fill_inner(fill)

    def _on_fill_inner(self, fill: Fill) -> Optional[Trade]:
        p = self._get_or_create_pos(fill.ticker, fill.time)
        sv = float(fill.side) * abs(fill.qty)
        old_q = p["qty"]
        p["fees"] += fill.commission
        if old_q != 0 and (old_q > 0) != (sv > 0):
            self._reduce(p, old_q, sv, fill.price)
        else:
            self._add_open(p, sv, fill.price)
        p["qty"] = old_q + sv
        if abs(p["qty"]) >= 0.01:
            self._save()
            return None
        return self._close_position(fill.ticker, p, fill.time)

    def _get_or_create_pos(self, ticker: str, t: float) -> dict[str, Any]:
        p = self._pos.get(ticker)
        if p is None:
            # fmt: off
            p = {
                "qty": 0.0, "avg": 0.0, "fees": 0.0, "t": t, "realized": 0.0,
                "open_q": 0.0, "entry_n": 0.0, "entry_q": 0.0,
                "exit_n": 0.0, "exit_q": 0.0,
            }
            # fmt: on
            self._pos[ticker] = p
        return p

    def _add_open(self, p: dict[str, Any], sv: float, px: float) -> None:
        amt = abs(sv)
        total = p["open_q"] + amt
        p["avg"] = (p["avg"] * p["open_q"] + px * amt) / total
        p["open_q"] = total
        p["entry_n"] += px * amt
        p["entry_q"] += amt

    def _reduce(self, p: dict[str, Any], old_q: float, sv: float, px: float) -> None:
        closed = min(abs(old_q), abs(sv))
        direction = 1.0 if old_q > 0 else -1.0
        p["realized"] += (px - p["avg"]) * closed * direction
        p["exit_n"] += px * closed
        p["exit_q"] += closed
        p["open_q"] -= closed
        if abs(sv) > abs(old_q):  # flipped past flat: residual opens here
            p["avg"] = px
            p["open_q"] = abs(sv) - abs(old_q)

    def _close_position(self, ticker: str, p: dict[str, Any], exit_t: float) -> Trade:
        trade = self._assemble(ticker, p, exit_t)
        del self._pos[ticker]
        self._trades.append(trade)
        self._save()
        self._fire_closed(trade)
        return trade

    def _assemble(self, ticker: str, p: dict[str, Any], exit_t: float) -> Trade:
        avg_entry = p["entry_n"] / p["entry_q"] if p["entry_q"] else 0.0
        avg_exit = p["exit_n"] / p["exit_q"] if p["exit_q"] else avg_entry
        pnl = p["realized"] - p["fees"]
        tid = f"{ticker}_{int(exit_t * 1000)}"
        # fmt: off
        return Trade(
            tid, ticker, p["t"], exit_t, avg_entry, avg_exit,
            p["entry_q"], pnl, p["fees"],
        )
        # fmt: on

    def _fire_closed(self, trade: Trade) -> None:
        if self._on_trade_closed is None:
            return
        try:
            self._on_trade_closed(trade)
        except Exception as exc:
            log.warning("on_trade_closed failed: %s", exc)

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
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # fmt: off
            data = [
                {
                    "id": t.trade_id, "ticker": t.ticker, "entry": t.avg_entry,
                    "exit": t.avg_exit, "pnl": t.pnl, "win": t.win, "fees": t.fees,
                    "t0": t.entry_time, "t1": t.exit_time, "qty": t.qty
                }
                for t in self._trades[-500:]
            ]
            # fmt: on
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(self._path)
        except Exception as exc:
            log.warning("Buffer save failed: %s", exc)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            for d in json.loads(self._path.read_text()):
                # fmt: off
                t = Trade(
                    d["id"], d["ticker"], d.get("t0", 0), d.get("t1", 0),
                    d["entry"], d["exit"], d.get("qty", 0), d["pnl"], d.get("fees", 0)
                )
                # fmt: on
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
