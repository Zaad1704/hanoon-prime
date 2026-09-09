"""tests/test_halim_analysis.py — HALIM post-trade analysis via /v1/complete."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from hanoon_prime.brain.halim_analysis import analyze_trade


def _fake_response(text: str):
    resp = MagicMock()
    resp.__enter__.return_value.read.return_value = json.dumps({"text": text}).encode()
    return resp


class TestAnalyzeTrade:
    """analyze_trade must use the live /v1/complete endpoint, not the dead
    /analyze_trade route, and return the flat insight JSON."""

    def test_posts_to_v1_complete(self):
        with patch(
            "hanoon_prime.brain.halim_analysis.urllib.request.urlopen",
            return_value=_fake_response('{"insight": "cut losers faster"}'),
        ) as fn:
            result = analyze_trade(
                "http://127.0.0.1:8765",
                {"ticker": "TSLA", "won": False, "pnl_pct": -1.5},
            )
        url = fn.call_args[0][0].full_url
        assert url.startswith("http://127.0.0.1:8765/v1/complete")
        assert result == {"insight": "cut losers faster"}

    def test_extracts_insight_from_freeform_text(self):
        noisy = 'Sure! Here you go:\n{"insight": "scale into strength"}\n</end>'
        with patch(
            "hanoon_prime.brain.halim_analysis.urllib.request.urlopen",
            return_value=_fake_response(noisy),
        ):
            result = analyze_trade("http://x", {"ticker": "TSLA"})
        assert result == {"insight": "scale into strength"}

    def test_returns_empty_on_bad_response(self):
        with patch(
            "hanoon_prime.brain.halim_analysis.urllib.request.urlopen",
            return_value=_fake_response("no json here"),
        ):
            result = analyze_trade("http://x", {"ticker": "TSLA"})
        assert result == {}

    def test_returns_empty_on_network_error(self):
        with patch(
            "hanoon_prime.brain.halim_analysis.urllib.request.urlopen",
            side_effect=RuntimeError("connection refused"),
        ):
            result = analyze_trade("http://x", {"ticker": "TSLA"})
        assert result == {}
