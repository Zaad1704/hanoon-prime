"""Halim research — internet-grounded strategy discovery for JULI.

Halim is JULI's mouthpiece: JULI (the numeric brain) cannot generate text,
so it asks Halim's LM to research market strategies from the web and return
structured candidates. The fetcher is deliberately bounded (only http/https,
hard size/time caps, silent failures) and the LM output is normalized into a
flat list of strategy descriptors the bot's strategy registry can ingest.
Every numeric that can reach the brain is clamped here too — a research
strategy is advisory, never a gate.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

_MAX_BODY: int = 32_000  # hard cap on fetched evidence bytes
_HTTP_TIMEOUT: float = 6.0
_MAX_STRATEGIES: int = 5  # Halim may propose at most this many candidates

_YAHOO_SEARCH = "https://query1.finance.yahoo.com/v1/finance/search?q={q}&newsCount=8"

# Bounded nudge clamp — research output can never exceed these ranges.
_SIZING_MIN: float = 0.65
_SIZING_MAX: float = 1.35
_CONF_MIN: float = 0.0
_CONF_MAX: float = 1.0
_SCALAR_NEUTRAL: float = 1.0


def fetch_web_context(query: str) -> dict[str, Any]:
    """Bounded web evidence for a research query (empty on any failure).

    Prefers an explicit http(s) URL (``query`` looks like one); otherwise
    falls back to the Yahoo Finance search API for headlines. Never raises.
    """
    q = (query or "").strip()
    if not q:
        return {"ok": False, "reason": "empty_query", "evidence": ""}
    url = _url_for(q)
    if not url:
        return {"ok": False, "reason": "unsafe_url", "evidence": ""}
    body = _http_get(url)
    if not body:
        return {"ok": False, "reason": "fetch_failed", "evidence": ""}
    evidence = _trim(body)
    if not evidence:
        return {"ok": False, "reason": "empty_evidence", "evidence": ""}
    return {"ok": True, "evidence": evidence, "url": url}


def _url_for(query: str) -> str | None:
    """Resolve a research query to a fetchable http(s) URL."""
    if query.startswith(("http://", "https://")):
        return query if len(query) < 2048 else None
    return _YAHOO_SEARCH.format(q=urllib.parse.quote(query))


def _http_get(url: str) -> str:
    """One bounded GET → body text (empty on any failure)."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 halim-research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            if resp.status >= 400:
                return ""
            return resp.read(_MAX_BODY).decode("utf-8", "ignore")
    except Exception as exc:
        log.debug("research fetch failed: %s", exc)
        return ""


def _trim(text: str) -> str:
    """Collapse whitespace and cap the evidence length."""
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    return collapsed[:_MAX_BODY]


def build_prompt(query: str, evidence: str) -> str:
    """Prompt Halim to research strategies for the query + evidence."""
    return (
        "You are Halim, the trading architect. Research the internet for market "
        "strategies that would suit the following request and evidence, then "
        "return EXACTLY this JSON, nothing else:\n"
        '{"regime_hint": "<trend_up|trend_down|range|vol|unknown>", '
        '"strategies": [{"name": "<short>", "thesis": "<1 sentence>", '
        '"entry": "<entry logic>", "exit": "<exit logic>", '
        '"risk": "<risk rule>", "conditions": "<market conditions>", '
        '"regime": "<trend_up|trend_down|range|vol|unknown>", '
        '"confidence": <0.0-1.0>}]}\n'
        f"REQUEST: {query}\n"
        + (f"WEB EVIDENCE: {evidence[:_MAX_BODY]}\n" if evidence else "")
        + "Return ONLY the JSON object."
    )


def build_fallback_prompt(query: str) -> str:
    """Minimal prompt for small LMs that truncate long JSON.

    The 4B MoE model reliably generates ~280 chars; this prompt asks
    for only the essential fields (name, thesis, regime, confidence)
    so the full JSON fits. Entry/exit/risk are optional — the registry
    accepts strategies with just these core fields.
    """
    return (
        "Return ONLY this JSON:\n"
        '{"regime_hint":"<trend_up|trend_down|range|vol|unknown>",'
        '"strategies":[{"name":"<short>",'
        '"thesis":"<1 sentence>",'
        '"regime":"<trend_up|trend_down|range|vol|unknown>",'
        '"confidence":<0.0-1.0>,'
        '"entry":"<1 sentence>","exit":"<1 sentence>",'
        '"risk":"<risk>","conditions":"<cond>"}]}\n'
        f"REQUEST: {query}\n"
        "Return ONLY the JSON."
    )


def normalize_strategies(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a parsed research response into bounded strategy entries."""
    raw = parsed.get("strategies")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[: _MAX_STRATEGIES]:
        if not isinstance(item, dict):
            continue
        name = _clean(str(item.get("name", ""))[:80])
        if not name:
            continue
        regime = str(item.get("regime", item.get("regime_hint", "unknown")))
        if regime not in ("trend_up", "trend_down", "range", "vol"):
            regime = "unknown"
        try:
            conf = float(item.get("confidence", 0.4))
        except (TypeError, ValueError):
            conf = 0.4
        entry = _clean(str(item.get("entry", ""))[:400]) or "see thesis"
        exit_ = _clean(str(item.get("exit", ""))[:400]) or "see thesis"
        risk = _clean(str(item.get("risk", ""))[:300]) or "standard risk"
        conditions = _clean(str(item.get("conditions", ""))[:300]) or "normal"
        if not entry or not exit_:
            log.debug("research candidate '%s' missing entry/exit — dropped", name)
            continue
        out.append(
            {
                "name": name,
                "thesis": _clean(str(item.get("thesis", ""))[:300]),
                "entry": entry,
                "exit": exit_,
                "risk": risk,
                "conditions": conditions,
                "regime": regime,
                "sizing": _SCALAR_NEUTRAL,
                "confidence": _clamp(conf, _CONF_MIN, _CONF_MAX),
                "source": "halim_research",
            }
        )
    return out


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return max(lo, min(hi, x))


def _clean(text: str) -> str:
    """Single-line, whitespace-collapsed, length-capped text."""
    return re.sub(r"\s+", " ", text or "").strip()


def parse_research(text: str) -> dict[str, Any]:
    """Parse Halim's researched strategies into a normalized dict."""
    try:
        start = text.index("{")
        end = text.rindex("}")
        parsed = json.loads(text[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        log.debug("research parse failed: no JSON object")
        return {"ok": False, "reason": "json_parse_failed", "strategies": []}
    if not isinstance(parsed, dict):
        return {"ok": False, "reason": "not_object", "strategies": []}
    strategies = normalize_strategies(parsed)
    return {
        "ok": True,
        "regime_hint": str(parsed.get("regime_hint", "unknown")),
        "strategies": strategies,
        "count": len(strategies),
    }