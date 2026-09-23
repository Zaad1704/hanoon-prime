"""brain.strategy_registry — JULI's researched strategy library.

Halim (JULI's LM mouthpiece) researches strategies from the web; this
persistent registry stores them as bounded modifier bundles. Cortex is
the sole verdict source (R1).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from .learning_config import (
    STRATEGY_FILE,
    STRATEGY_MAX_POOL,
    STRATEGY_SCORE_MOD_BOUND,
    STRATEGY_SIZING_MAX,
    STRATEGY_SIZING_MIN,
)
from .strategy_priors import seeded_priors

log = logging.getLogger(__name__)
_VALID_REGIMES = frozenset({"trend_up", "trend_down", "range", "vol", "unknown"})


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return (slug or "strategy")[:48]


def _clean(c: dict[str, Any], key: str, n: int) -> str:
    return str(c.get(key, ""))[:n]


def _bundle(c: dict[str, Any]) -> tuple[float, float, float]:
    lo, hi, sb = STRATEGY_SIZING_MIN, STRATEGY_SIZING_MAX, STRATEGY_SCORE_MOD_BOUND
    get = c.get
    try:
        s = max(lo, min(hi, float(get("sizing", 1.0))))
        m = max(-sb, min(sb, float(get("score_mod", 0.0))))
        return s, m, max(0.0, min(1.0, float(get("confidence", 0.4))))
    except (TypeError, ValueError):
        return (1.0, 0.0, 0.4)


class StrategyRegistry:
    """Persistent, bounded library of researched strategies."""

    def __init__(self, path: Path | None = None) -> None:
        """Load (or seed) the strategy library."""
        self._path = path or STRATEGY_FILE
        self._lock = threading.RLock()
        self._strategies: dict[str, dict[str, Any]] = {}
        self._load_or_seed()

    def get(self, strategy_id: str) -> dict[str, Any] | None:
        """One strategy by id (copy), or None."""
        with self._lock:
            item = self._strategies.get(strategy_id)
            return dict(item) if item else None

    def ids_for(self, regime: str) -> list[str]:
        """Strategy ids applicable to a canonical regime (unknown = any)."""
        with self._lock:
            ok: tuple[str, ...] = ("unknown", "any")
            if regime in ok:
                return list(self._strategies)
            ok = (*ok, regime)
            return [s for s, i in self._strategies.items() if i.get("regime") in ok]

    def nudge_for(self, strategy_id: str) -> dict[str, float]:
        """Bounded nudge bundle: sizing scalar + score modifier."""
        with self._lock:
            item = self._strategies.get(strategy_id)
            size, mod, _ = _bundle(item if item is not None else {})
            return {"sizing": size, "score_mod": mod}

    def count(self) -> int:
        """Number of strategies in the library."""
        with self._lock:
            return len(self._strategies)

    def ingest(self, candidate: dict[str, Any]) -> str | None:
        """Ingest one researched strategy (bounded, deduped by name)."""
        name = _clean(candidate, "name", 80).strip()
        if not name or not candidate.get("entry") or not candidate.get("exit"):
            return None
        sid = _slug(name)
        regime = str(candidate.get("regime", "unknown"))
        if regime not in _VALID_REGIMES:
            regime = "unknown"
        size, mod, conf = _bundle(candidate)
        with self._lock:
            if self._strategies.get(sid):
                return sid
            if len(self._strategies) >= STRATEGY_MAX_POOL:
                return None
            self._strategies[sid] = {
                "id": sid,
                "name": name,
                "regime": regime,
                "thesis": _clean(candidate, "thesis", 300),
                "sizing": size,
                "score_mod": mod,
                "confidence": conf,
                "source": _clean(candidate, "source", 80) or "halim_research",
                "trials": 0,
                "wins": 0,
                "created_at": time.time(),
            }
            self._save()
            log.info("strategy ingested: %s (%s)", sid, regime)
            return sid

    def record(self, strategy_id: str, won: bool) -> None:
        """Record one realized outcome against a strategy (advisory)."""
        with self._lock:
            item = self._strategies.get(strategy_id)
            if not item:
                return
            item["trials"] = int(item.get("trials", 0)) + 1
            item["wins"] = int(item.get("wins", 0)) + (1 if won else 0)
            self._save()

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view of the whole library."""
        with self._lock:
            rows = []
            for i in self._strategies.values():
                t = int(i.get("trials", 0))
                row = dict(i)
                row["win_rate"] = round(i.get("wins", 0) / t, 3) if t else None
                rows.append(row)
            return {"count": len(rows), "strategies": rows}

    def reset(self) -> None:
        """Wipe the library back to seeded priors (poisoned-data path)."""
        with self._lock:
            self._strategies = {}
            self._seed()
            self._save()

    def _load_or_seed(self) -> None:
        """Load persisted library; missing/corrupt → seeded priors."""
        if not self._path.exists():
            self._strategies = {}
            self._seed()
            self._save()
            return
        try:
            pool = json.loads(self._path.read_text()).get("strategies")
            for sid, item in pool.items() if isinstance(pool, dict) else ():
                r = str(item.get("regime", "unknown"))
                item["regime"] = r if r in _VALID_REGIMES else "unknown"
                item["sizing"], item["score_mod"], _ = _bundle(item)
                self._strategies[str(sid)] = item
            self._seed()
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            self._strategies = {}
            self._seed()
            self._save()

    def _seed(self) -> None:
        """Ensure seeded priors are present (evict research if pool full)."""
        for prior in seeded_priors():
            if self.ingest(prior) is not None:
                continue
            sid = _slug(_clean(prior, "name", 80).strip())
            res = {
                k: v for k, v in self._strategies.items() if v.get("source") != "seeded"
            }
            if not sid or self._strategies.get(sid) or not res:
                continue
            victim = min(res, key=lambda k: self._rank(res[k]))
            del self._strategies[victim]
            self.ingest(prior)
            log.info("strategy seed evicted %s for %s", victim, sid)

    @staticmethod
    def _rank(v: dict[str, Any]) -> tuple[int, float]:
        return int(v.get("trials", 0)), float(v.get("created_at", 0.0))

    def _save(self) -> None:
        """Persist atomically (tmp + replace)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"strategies": self._strategies, "version": 1}))
            tmp.replace(self._path)
        except OSError as exc:
            log.debug("strategy library save failed: %s", exc)
