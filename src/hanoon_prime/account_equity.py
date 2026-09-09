"""Account equity resolution for pre-market sizing.

IB's accountSummary reports no NetLiquidation before 09:30 ET. Pre-market
entries still need equity to size, so resolve in order: live NetLiq →
local cash + portfolio mark → last-good cache. Unknown stays unknown,
and the sizing veto owns that case (no fabricated equity).

Cache path is overridable via HANOON_EQUITY_CACHE (used by tests).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from ._ib_sync import read_portfolio

log = logging.getLogger(__name__)


def _cache_path() -> Path:
    env = os.getenv("HANOON_EQUITY_CACHE", "")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "runtime" / "equity_cache.json"


def _summary_tags(ib: Any, account: str | None) -> dict[str, float]:
    """Collect numeric accountSummary tags, tolerating missing entries."""
    out: dict[str, float] = {}
    try:
        items = list(ib.accountSummary(account))
    except Exception as exc:
        log.debug("accountSummary unreadable: %s", exc)
        items = []
    for item in items:
        try:
            out[str(item.tag)] = float(item.value)
        except (AttributeError, TypeError, ValueError) as exc:
            log.debug("skipping non-numeric account tag %r: %s", item.tag, exc)
    return out


def save_equity_cache(equity: float, path: Path | None = None) -> None:
    """Persist the last known equity (atomic tmp + replace, best-effort)."""
    p = path or _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"ts": time.time(), "equity": float(equity)}),
            encoding="utf-8",
        )
        tmp.replace(p)
    except Exception as exc:
        log.warning("equity cache write failed: %s", exc)


def load_equity_cache(path: Path | None = None) -> float | None:
    """Return the last cached equity, or None when missing/corrupt/<=0."""
    p = path or _cache_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        eq = float(data["equity"])
        return eq if math.isfinite(eq) and eq > 0 else None
    except Exception:
        return None


def _live_or_local(ib: Any, account: str | None) -> float | None:
    """Live NetLiq first; otherwise cash + portfolio mark (pre-market)."""
    tags = _summary_tags(ib, account)
    net = tags.get("NetLiquidation", 0.0)
    if math.isfinite(net) and net > 0:
        return net
    try:
        portfolio = read_portfolio(ib)
    except Exception:
        portfolio = {}
    cash = tags.get("CashBalance", 0.0)
    total = cash + sum(float(p.get("value", 0.0) or 0.0) for p in portfolio.values())
    return total if math.isfinite(total) and total > 0 else None


def resolve_account_equity(ib: Any, account: str | None) -> tuple[float | None, bool]:
    """(equity, synced). synced=True for any real value (live/local/cache)."""
    live = _live_or_local(ib, account)
    if live is not None:
        save_equity_cache(live)
        return live, True
    cached = load_equity_cache()
    if cached is not None:
        return cached, True
    return None, False
