"""hanoon_prime.brain.halim_analysis — HALIM post-trade analysis utilities."""
from __future__ import annotations
import json
import logging
import urllib.request
from typing import Any

log = logging.getLogger(__name__)


def analyze_trade(base_url: str, trade_data: dict[str, Any]) -> dict[str, Any]:
    """Post-trade analysis - returns insights for learning."""
    try:
        data = json.dumps(trade_data).encode()
        req = urllib.request.Request(
            f"{base_url}/analyze_trade",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
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
            result = json.loads(resp.read().decode())
            return result.get("recommendations", [])
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
            return json.loads(resp.read().decode())
    except Exception as e:
        log.debug("HALIM health check failed: %s", e)
        return {}


__all__ = ["analyze_trade", "get_improvement_recommendations", "get_health_advice"]