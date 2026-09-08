"""hanoon_prime.brain.halim_recommendations — HALIM recommendation parser.

Converts HALIM's structured recommendations into Juli parameter adjustments.
HALIM returns JSON with actionable items; this module validates and applies them.

Format:
    {"recommendations": [
        {"action": "adjust_threshold", "param": "threshold", "value": 0.62, "reason": "..."},
        {"action": "adjust_weight", "param": "momentum", "value": 0.25, "reason": "..."},
        {"action": "adjust_risk", "param": "max_position_notional", "value": 20000, "reason": "..."},
    ]}
"""

from __future__ import annotations

import logging
import time
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

# Valid actions and their parameter bounds
_ACTIONS = {
    "adjust_threshold": {"min": 0.40, "max": 0.80, "target": "threshold"},
    "adjust_weight": {"min": -2.0, "max": 2.0, "target": "weights"},
    "adjust_risk_scalar": {"min": 0.1, "max": 2.0, "target": "risk_scalar"},
    "adjust_stop_mult": {"min": 1.0, "max": 5.0, "target": "stop_mult"},
    "adjust_target_mult": {"min": 2.0, "max": 10.0, "target": "target_mult"},
}

# Valid weight names (must match immune.py INDICATOR_WEIGHTS)
_VALID_WEIGHTS = frozenset({
    "vpin", "orderbook_imbalance", "institutional_flow", "momentum",
    "vwap_deviation", "rsi", "macd_hist", "bollinger_position", "adx",
    "stoch_k", "mfi", "ad_signal", "obv_divergence", "volume_profile_proximity",
    "spread_tightness", "trade_intensity", "hurst_exponent", "mean_reversion",
    "trend_strength", "sr_proximity", "elliott_wave", "institutional_wave",
    "keltner_position", "vw_macd_hist", "microstructure", "fib_proximity",
    "kelly_fraction",
})


def fetch_recommendations(base_url: str) -> list[dict[str, Any]]:
    """Fetch structured recommendations from HALIM."""
    try:
        req = urllib.request.Request(
            f"{base_url}/recommendations",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw: dict[str, Any] = resp.read().decode()
            import json
            data = json.loads(raw)
            return data.get("recommendations", [])
    except Exception as e:
        log.debug("HALIM recommendations fetch failed: %s", e)
        return []


def validate_recommendation(rec: dict[str, Any]) -> str | None:
    """Validate a single recommendation. Returns error string or None if valid."""
    action = rec.get("action", "")
    if action not in _ACTIONS:
        return f"unknown action: {action}"
    param = rec.get("param", "")
    value = rec.get("value")
    if value is None or not isinstance(value, (int, float)):
        return f"missing or non-numeric value"
    bounds = _ACTIONS[action]
    if not (bounds["min"] <= value <= bounds["max"]):
        return f"value {value} out of bounds [{bounds['min']}, {bounds['max']}]"
    if action == "adjust_weight" and param not in _VALID_WEIGHTS:
        return f"unknown weight: {param}"
    return None


def apply_recommendation(
    rec: dict[str, Any],
    dynamics: Any,
    memory: Any,
    immune_update: Any = None,
) -> bool:
    """Apply a validated recommendation to Juli's parameters. Returns True if applied."""
    action = rec["action"]
    param = rec["param"]
    value = float(rec["value"])
    reason = rec.get("reason", "")
    target = _ACTIONS[action]["target"]

    if target == "threshold":
        old = dynamics.threshold
        dynamics.threshold = value
        log.info("HALIM: threshold %.4f → %.4f (%s)", old, value, reason)
        return True

    if target == "weights":
        weights = memory.get_weights()
        old = weights.get(param, 0.0)
        weights[param] = value
        memory.set_weights(weights)
        log.info("HALIM: weight %s %.4f → %.4f (%s)", param, old, value, reason)
        return True

    if target == "risk_scalar" and immune_update:
        immune_update("risk_scalar", value, reason)
        return True

    if target in ("stop_mult", "target_mult") and immune_update:
        immune_update(target, value, reason)
        return True

    log.warning("HALIM: cannot apply %s (no applier)", action)
    return False


def process_recommendations(
    base_url: str,
    dynamics: Any,
    memory: Any,
    max_apply: int = 3,
    cooldown_secs: float = 300.0,
) -> list[dict[str, Any]]:
    """Fetch, validate, and apply HALIM recommendations.

    Returns list of applied recommendations. Limits to max_apply per call
    and respects cooldown to avoid thrashing.
    """
    now = time.time()
    if not hasattr(process_recommendations, "_last_apply"):
        process_recommendations._last_apply = 0.0  # type: ignore[attr-defined]
    if now - process_recommendations._last_apply < cooldown_secs:  # type: ignore[attr-defined]
        return []

    recs = fetch_recommendations(base_url)
    if not recs:
        return []

    applied: list[dict[str, Any]] = []
    for rec in recs[:max_apply]:
        error = validate_recommendation(rec)
        if error:
            log.warning("HALIM rec rejected: %s — %s", rec, error)
            continue
        if apply_recommendation(rec, dynamics, memory):
            applied.append(rec)

    if applied:
        process_recommendations._last_apply = now  # type: ignore[attr-defined]
        log.info("HALIM: applied %d/%d recommendations", len(applied), len(recs))

    return applied


__all__ = [
    "fetch_recommendations",
    "validate_recommendation",
    "apply_recommendation",
    "process_recommendations",
]
