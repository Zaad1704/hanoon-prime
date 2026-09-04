"""hanoon_prime.brain.halim_adapter — HALIM API client.

Communicates with the external HALIM AI advisor service.
Returns bounded modifier and rich regime classifications.
Non-blocking: starts debate async, reads cached result instantly.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any

from .config import HALIM_MOD_BOUND

log = logging.getLogger(__name__)
_VALID_REGIMES = frozenset(
    {
        "trending_bullish",
        "trending_bearish",
        "ranging",
        "volatile",
        "breakout",
        "consolidation",
        "mean_reverting",
        "normal",
    }
)
_REGIME_PROMPT_TPL = (
    "You are a market regime classifier for a trading bot.\n"
    "Analyze the following data and classify the current market regime.\n\n"
    "Recent prices: [{prices}]\nVolume: {volume:.0f}\nIndicators: {indicators}\n\n"
    "Respond with EXACTLY this JSON format:\n"
    '{"regime": "<one of: trending_bullish, trending_bearish, ranging, '
    'volatile, breakout, consolidation, mean_reverting>",'
    '"confidence": <0.0-1.0>, "multiplier": <0.5-1.5>,'
    '"key_drivers": ["<what drove this>"],'
    '"risk_adjustment": "<aggressive|normal|defensive>",'
    '"description": "<one sentence summary>"}\n'
    "Return ONLY the JSON object."
)
_FB = {
    "bull": ("trending_bullish", 1.2, "normal", ["ADX>25", "positive_momentum"]),
    "bear": ("trending_bearish", 0.8, "defensive", ["ADX>25", "negative_momentum"]),
    "range": ("ranging", 0.9, "defensive", ["ADX<20"]),
    "normal": ("normal", 1.0, "normal", ["fallback"]),
}


class HalimAdapter:
    """Async HALIM API client with caching."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765") -> None:
        self._base_url = base_url
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl: float = 60.0
        self._regime_cache: dict[str, Any] = {}
        self._regime_ts: float = 0.0
        self._regime_ttl: float = 30.0

    def get_modifier(
        self, ticker: str, alpha: dict[str, float], score: float, verdict: str
    ) -> float:
        """Get HALIM's advisory modifier. Non-blocking, cached."""
        cached = self._cache.get(ticker)
        if cached and (time.time() - float(cached.get("ts", 0))) < self._cache_ttl:
            return float(cached.get("modifier", 0.0))
        self._start_debate_async(ticker, alpha, score, verdict)
        return 0.0

    def get_regime(
        self, indicators: dict[str, float], prices: list[float], volume: float = 0.0
    ) -> dict[str, Any]:
        """Get rich regime classification from HALIM. Returns structured analysis."""
        now = time.time()
        if self._regime_cache and (now - self._regime_ts) < self._regime_ttl:
            return dict(self._regime_cache)
        prompt = self._build_regime_prompt(indicators, prices, volume)
        result = self._query_halim(prompt, purpose="regime", priority="high")
        if result.get("ok") and result.get("text"):
            parsed = self._parse_regime_response(result["text"])
            if parsed:
                self._regime_cache = parsed
                self._regime_ts = now
                log.info(
                    "HALIM regime: %s conf=%.2f mult=%.2f",
                    parsed.get("regime", "?"),
                    parsed.get("confidence", 0),
                    parsed.get("multiplier", 1.0),
                )
                return parsed
        return self._fallback_regime(indicators)

    def _build_regime_prompt(
        self, indicators: dict[str, float], prices: list[float], volume: float
    ) -> str:
        recent = prices[-20:] if len(prices) >= 20 else prices
        price_str = ", ".join(f"{p:.2f}" for p in recent[-10:])
        ind_str = ", ".join(f"{k}={v:.4f}" for k, v in list(indicators.items())[:12])
        return _REGIME_PROMPT_TPL.format(
            prices=price_str, volume=volume, indicators=ind_str
        )

    def _parse_regime_response(self, text: str) -> dict[str, Any] | None:
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            raw: dict[str, Any] = json.loads(text)
            if "regime" not in raw:
                return None
            if raw["regime"] not in _VALID_REGIMES:
                raw["regime"] = "normal"
            raw["confidence"] = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
            raw["multiplier"] = max(0.5, min(1.5, float(raw.get("multiplier", 1.0))))
            raw.setdefault("key_drivers", [])
            raw.setdefault("risk_adjustment", "normal")
            raw.setdefault("description", "")
            result: dict[str, Any] = raw
            return result
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def _fallback_regime(self, indicators: dict[str, float]) -> dict[str, Any]:
        adx = indicators.get("adx", 20.0)
        mom = indicators.get("momentum", 0.0)
        key = (
            "bull"
            if adx > 25 and mom > 0
            else (
                "bear" if adx > 25 and mom < 0 else ("range" if adx < 20 else "normal")
            )
        )
        r, m, risk, drivers = _FB[key]
        return {
            "regime": r,
            "confidence": 0.4,
            "multiplier": m,
            "key_drivers": drivers,
            "risk_adjustment": risk,
            "description": f"Mechanical fallback: {r}",
        }

    def _start_debate_async(
        self, ticker: str, alpha: dict[str, float], score: float, verdict: str
    ) -> None:
        try:
            data = json.dumps(
                {"ticker": ticker, "score": score, "verdict": verdict, "alpha": alpha}
            ).encode()
            req = urllib.request.Request(
                f"{self._base_url}/debate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            log.debug("HALIM debate failed: %s", e)

    def _query_halim(
        self, prompt: str, purpose: str = "reasoning", priority: str = "medium"
    ) -> dict[str, Any]:
        try:
            data = json.dumps(
                {"prompt": prompt, "purpose": purpose, "priority": priority}
            ).encode()
            req = urllib.request.Request(
                f"{self._base_url}/v1/complete",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result: dict[str, Any] = json.loads(resp.read().decode())
                return result
        except Exception as e:
            log.debug("HALIM query failed: %s", e)
            return {"ok": False, "reason": str(e)}

    def analyze_trade(self, trade_data: dict[str, Any]) -> dict[str, Any]:
        """Post-trade analysis - returns insights for learning."""
        from .halim_analysis import analyze_trade as _analyze
        return _analyze(self._base_url, trade_data)

    def get_improvement_recommendations(self) -> list[dict[str, Any]]:
        """Get HALIM's tactical recommendations for improvement."""
        from .halim_analysis import get_improvement_recommendations as _get_recs
        return _get_recs(self._base_url)

    def get_health_advice(self) -> dict[str, Any]:
        """Get HALIM's health assessment and recommendations."""
        from .halim_analysis import get_health_advice as _get_health
        return _get_health(self._base_url)
