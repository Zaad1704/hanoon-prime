"""hanoon_prime.brain.strategy_research — HALIM research client + cadence.

JULI cannot generate text, so it delegates internet research to HALIM's
``/v1/research`` endpoint: HALIM fetches bounded web evidence, reasons the
strategy, and returns structured candidates. This module runs that on a
cadence (throttled, non-blocking) and ingests results into the strategy
registry the bandit consumes. Every ingest is bounded by the registry —
research can never hand the brain a gate.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.request
from typing import Any, Callable

from .learning_config import (
    RESEARCH_BASE_URL,
    RESEARCH_INTERVAL_SEC,
    RESEARCH_MAX_PER_CYCLE,
    RESEARCH_TIMEOUT,
    RESEARCH_TOPICS,
)
from .strategy_registry import StrategyRegistry

log = logging.getLogger(__name__)


def _research_query(base_url: str, query: str) -> dict[str, Any]:
    """POST one research request to HALIM's /v1/research (bounded)."""
    data = json.dumps({"query": query, "source": "juli_research"}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/research",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=RESEARCH_TIMEOUT) as resp:
        raw = json.loads(resp.read().decode())
    return raw if isinstance(raw, dict) else {}


def _candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract bounded strategy candidates from a research response."""
    out = result.get("strategies")
    if not isinstance(out, list):
        return []
    return [s for s in out if isinstance(s, dict)][:RESEARCH_MAX_PER_CYCLE]


class StrategyResearch:
    """Throttled research client feeding the strategy registry."""

    def __init__(
        self,
        registry: StrategyRegistry,
        base_url: str = RESEARCH_BASE_URL,
        interval_sec: float = RESEARCH_INTERVAL_SEC,
    ) -> None:
        """Bind a strategy registry and the research cadence."""
        self._registry = registry
        self._base_url = base_url
        self._interval_sec = interval_sec
        self._last_run: float = 0.0
        self._last_result: dict[str, Any] = {"ok": True, "count": 0}
        self._total_ingested: int = 0

    def maybe_run(
        self,
        query_fn: Callable[[str, str], dict[str, Any]] | None = None,
        enabled: bool = True,
    ) -> list[str]:
        """Throttled research pass: ingest new strategies (never blocks fast path)."""
        if not enabled:
            return []
        now = time.time()
        if now - self._last_run < self._interval_sec:
            return []
        self._last_run = now
        topic = random.choice(RESEARCH_TOPICS)
        try:
            result = (query_fn or _research_query)(self._base_url, topic)
        except Exception as exc:
            log.debug("strategy research query failed: %s", exc)
            return self._nil("lm_unavailable")
        if not isinstance(result, dict) or not result.get("ok"):
            return self._nil("lm_unavailable")
        inserted = []
        for cand in _candidates(result):
            sid = self._registry.ingest(cand)
            if sid:
                inserted.append(sid)
                self._total_ingested += 1
        self._last_result = {
            "ok": True,
            "count": len(inserted),
            "topic": topic,
            "regime_hint": str(result.get("regime_hint", "unknown")),
            "evidence": bool(result.get("evidence")),
        }
        if inserted:
            log.info("strategy research: %d new from '%s'", len(inserted), topic)
        return inserted

    def _nil(self, reason: str) -> list[str]:
        self._last_result = {"ok": False, "count": 0, "reason": reason}
        return []

    def snapshot(self) -> dict[str, Any]:
        """Telemetry view of the research loop."""
        return {
            "total_ingested": self._total_ingested,
            "last_result": dict(self._last_result),
            "interval_sec": self._interval_sec,
            "topics": list(RESEARCH_TOPICS),
        }

    def reset(self) -> None:
        """Reset the cadence clock and counters (state wiped upstairs)."""
        self._last_run = 0.0
        self._last_result = {"ok": True, "count": 0}
        self._total_ingested = 0


__all__ = [
    "RESEARCH_BASE_URL",
    "StrategyResearch",
    "_candidates",
    "_research_query",
]
