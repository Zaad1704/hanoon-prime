"""hanoon_prime.brain.halim_analysis — HALIM post-trade analysis utilities."""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

log = logging.getLogger(__name__)


def _postmortem_prompt(trade_data: dict[str, Any]) -> str:
    """Prompt HALIM for a compact post-trade self-critique (flat JSON)."""
    return (
        "You are the trading architect HALIM. Critique this just-closed trade "
        "and return EXACTLY this JSON, nothing else:\n"
        '{"insight": "<2-3 sentence critique: what worked, what failed, and the '
        'one thing to carry into the next trade>"}\n'
        f"trade={json.dumps(trade_data)}"
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of free-form model text."""
    try:
        start = text.index("{")
        end = text.rindex("}")
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, json.JSONDecodeError):
        return None


def analyze_trade(base_url: str, trade_data: dict[str, Any]) -> dict[str, Any]:
    """Post-trade analysis - returns insights for learning.

    Uses the live /v1/complete endpoint (the /analyze_trade route does not
    exist on the running HALIM server) and parses the flat insight JSON.
    """
    try:
        data = json.dumps(
            {
                "prompt": _postmortem_prompt(trade_data),
                "purpose": "analyze_trade",
                "priority": "high",
            }
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/complete",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw: dict[str, Any] = json.loads(resp.read().decode())
        parsed = _extract_json(raw.get("text", ""))
        if parsed and parsed.get("insight"):
            return {"insight": str(parsed["insight"])}
        return {}
    except Exception as e:
        log.debug("HALIM trade analysis failed: %s", e)
        return {}


def get_improvement_recommendations(base_url: str) -> list[dict[str, Any]]:
    """Get HALIM's tactical recommendations for improvement."""
    try:
        req = urllib.request.Request(
            f"{base_url}/recommendations",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw: dict[str, Any] = json.loads(resp.read().decode())
            recs: list[dict[str, Any]] = raw.get("recommendations", [])
            return recs
    except Exception as e:
        log.debug("HALIM recommendations failed: %s", e)
        return []


def get_health_advice(base_url: str) -> dict[str, Any]:
    """Get HALIM's health assessment and recommendations."""
    try:
        req = urllib.request.Request(
            f"{base_url}/health",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            health: dict[str, Any] = json.loads(resp.read().decode())
            return health
    except Exception as e:
        log.debug("HALIM health check failed: %s", e)
        return {}


__all__ = ["analyze_trade", "get_improvement_recommendations", "get_health_advice"]
