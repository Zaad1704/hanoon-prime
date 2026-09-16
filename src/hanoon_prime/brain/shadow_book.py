"""brain.shadow_book — zero-size paper trial book (data-starvation fix).

Paper positions when the pipeline declines a directional signal for a
non-policy reason, TTL-closed. Feeds ONLY bandit + registry.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .learning_config import (
    SHADOW_BOOK_FILE,
    SHADOW_MAX_HISTORY,
    SHADOW_MAX_OPEN,
    SHADOW_TTL,
)

log = logging.getLogger(__name__)


@dataclass
class ShadowClose:
    """One completed paper trial (stale = refund at entry, skip learning)."""

    ticker: str
    direction: int
    strategy_id: str
    canon: str
    horizon: str
    entry_price: float
    exit_price: float
    age: float
    pnl_pct: float
    won: bool
    stale: bool = False


@dataclass
class ShadowOpen:
    """A proposed paper entry, market-agnostic metadata."""

    ticker: str
    direction: int
    strategy_id: str
    canon: str
    horizon: str


class ShadowBook:
    """Paper-position ledger: one open per ticker, TTL-closed on re-eval."""

    def __init__(
        self,
        path: Path | None = None,
        ttl: float = SHADOW_TTL,
        max_open: int = SHADOW_MAX_OPEN,
    ) -> None:
        self._path = path or SHADOW_BOOK_FILE
        self._ttl = ttl
        self._max_open = max_open
        self._lock = threading.RLock()
        self._open: dict[str, dict[str, Any]] = {}
        self._history: list[ShadowClose] = []
        self._opened = 0
        self._load()

    def open(self, spec: ShadowOpen, price: float, now: float | None = None) -> None:
        """Record a paper entry at the current mark (no-op if already open)."""
        if price <= 0:
            return
        with self._lock:
            if spec.ticker in self._open or len(self._open) >= self._max_open:
                return
            self._open[spec.ticker] = {
                "direction": int(spec.direction),
                "strategy_id": str(spec.strategy_id),
                "canon": str(spec.canon),
                "horizon": str(spec.horizon),
                "entry_price": float(price),
                "entry_epoch": float(now if now is not None else time.time()),
            }
            self._opened += 1
            self._save()

    def sweep_close(
        self, ticker: str, price: float, now: float | None = None
    ) -> ShadowClose | None:
        """Force-close an open paper trial once it passes TTL."""
        if price <= 0:
            return None
        ts = float(now if now is not None else time.time())
        with self._lock:
            pos = self._open.get(ticker)
            if pos is None:
                return None
            if ts - float(pos["entry_epoch"]) < self._ttl:
                return None
            return self._close(ticker, pos, price, ts, stale=False)

    def sweep_expired(self, now: float | None = None) -> list[ShadowClose]:
        """S2: refund every open trial past TTL (stale=True, skip learning)."""
        ts = float(now if now is not None else time.time())
        closes: list[ShadowClose] = []
        with self._lock:
            for ticker, pos in self._open.copy().items():
                if ts - float(pos["entry_epoch"]) < self._ttl:
                    continue
                closes.append(
                    self._close(ticker, pos, float(pos["entry_price"]), ts, True)
                )
        return closes

    def drop(self, ticker: str) -> None:
        """Abandon a paper trial because a real trade covered the signal."""
        with self._lock:
            self._open.pop(ticker, None)

    def _close(
        self,
        ticker: str,
        pos: dict[str, Any],
        price: float,
        ts: float,
        stale: bool = False,
    ) -> ShadowClose:
        entry = float(pos["entry_price"])
        age = ts - float(pos["entry_epoch"])
        pnl_pct = (price - entry) / entry * float(pos["direction"])
        self._open.pop(ticker, None)
        closed = ShadowClose(
            ticker=ticker,
            direction=int(pos["direction"]),
            strategy_id=str(pos["strategy_id"]),
            canon=str(pos["canon"]),
            horizon=str(pos["horizon"]),
            entry_price=entry,
            exit_price=float(price),
            age=age,
            pnl_pct=round(pnl_pct, 6),
            won=pnl_pct > 0,
            stale=stale,
        )
        self._history.append(closed)
        del self._history[:-SHADOW_MAX_HISTORY]
        self._save()
        return closed

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view of the paper book."""
        with self._lock:
            closes = self._history[-SHADOW_MAX_HISTORY:]
            total = len(closes)
            edge = sum(c.pnl_pct for c in closes) / total if total else 0.0
            return {
                "open_count": len(self._open),
                "total_opened": self._opened,
                "closed_trials": total,
                "win_rate": (
                    round(len([c for c in closes if c.won]) / total, 4)
                    if total
                    else 0.0
                ),
                "edge": round(edge, 6),
                "ttl_seconds": self._ttl,
                "open": list(self._open.keys())[:SHADOW_MAX_OPEN],
            }

    def _load(self) -> None:
        """Restore open book + history; corrupt file → fresh book."""
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            self._open = {
                k: v
                for k, v in d.get("open", {}).items()
                if isinstance(v, dict)
                and isinstance(v.get("entry_epoch"), (int, float))
            }
            self._history = [
                ShadowClose(**c) for c in d.get("history", []) if isinstance(c, dict)
            ] or self._history
            self._opened = int(d.get("opened", 0))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("Shadow book load failed (fresh): %s", exc)

    def _save(self) -> None:
        """Persist atomically (tmp + replace)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            payload = {
                "open": self._open,
                "history": [c.__dict__ for c in self._history],
                "opened": self._opened,
            }
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._path)
        except OSError as exc:
            log.debug("Shadow book save failed: %s", exc)


__all__ = ["SHADOW_TTL", "ShadowBook", "ShadowClose"]
