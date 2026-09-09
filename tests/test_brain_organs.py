"""tests/test_brain_organs — every brain organ that produces a signal must
reach the decision path (produce-but-ignore is a wiring bug).

Covers:
- Slow-path Thinker output consumed in the fast path (modifier, confidence,
  risk scalar — all bounded).
- Episodic k-NN modifier queried live and mirrored into shared state.
- Dead wrapper organs removed from the orchestrator (deliberation module
  stays importable and functional as a library).
- News engine publishes every tracked ticker (0.0-fill ⇒ real polarity flows).
- Overnight monitor sends a browser User-Agent so Cloudflare stops 403-ing
  the tunnel health probe.
"""

import importlib.util
from pathlib import Path

import hanoon_prime.brain.news_sources as ns_mod
from hanoon_prime.brain.config import DEFAULT_WEIGHTS
from hanoon_prime.brain.orchestrator import NeuromorphicBrain
from hanoon_prime.brain.risk import SizingResult
from hanoon_prime.brain.shared_state import BrainState


def _alpha():
    return {k: 0.5 for k in DEFAULT_WEIGHTS}


def _brain():
    return NeuromorphicBrain(brain_state=BrainState(), enable_neuromorphic=False)


def _raw(b, thinker_mod=0.0):
    b.state.update(thinker_modifier=thinker_mod)
    return b._score_pipeline("TEST", _alpha(), 1.0, 0.0, 0.0)["raw_score"]


# ── Thinker (S2 deliberation) consumed by the fast path ─────────────


class TestThinkerConsumed:
    def test_modifier_shifts_raw_score_bounded(self):
        b = _brain()
        base = _raw(b, 0.0)
        pos = _raw(b, 0.06)
        neg = _raw(b, -0.06)
        assert base - 0.06 - 1e-9 <= neg < base < pos <= base + 0.06 + 1e-9

    def test_modifier_clamped_to_total_mod_bound(self):
        b = _brain()
        base = _raw(b, 0.0)
        wild = _raw(b, 10.0)
        assert wild <= base + 0.06 + 1e-9

    def test_confidence_nudge_bounded(self):
        b = _brain()
        b.state.update(thinker_confidence_mod=0.05)
        base = b._score_pipeline("TEST", _alpha(), 1.0, 0.0, 0.0)["confidence"]
        b.state.update(thinker_confidence_mod=-0.05)
        neg = b._score_pipeline("TEST", _alpha(), 1.0, 0.0, 0.0)["confidence"]
        assert 0.05 <= base <= 0.95 and 0.05 <= neg <= 0.95
        assert base - neg >= 0.09  # both nudges applied (bounded apart)

    def test_risk_scalar_clamped_into_size(self):
        b = _brain()
        sizing = SizingResult(shares=100, risk_pass=True)
        ctx = {"stabilized": 0.9, "confidence": 0.6}
        b.state.update(thinker_risk_scalar=0.85)
        b._scale_admitted_size(ctx, sizing, "unknown", "scalp", None)
        assert sizing.shares == 85
        sizing = SizingResult(shares=100, risk_pass=True)
        b.state.update(thinker_risk_scalar=99.0)
        b._scale_admitted_size(ctx, sizing, "unknown", "scalp", None)
        assert sizing.shares == 125  # clamped to RISK_CEIL 1.25
        # risk-pass=False stays untouched by the affective scalar
        sizing = SizingResult(shares=100, risk_pass=False)
        b._scale_admitted_size(ctx, sizing, "unknown", "scalp", None)
        assert sizing.shares == 100


# ── Episodic k-NN bias reaches decisions ────────────────────────────


class TestEpisodicBiasLive:
    def test_tick_queries_episodic_and_mirrors_state(self):
        alpha = _alpha()
        b0 = _brain()
        b0._check_eod_penalty = lambda: 1.0
        b0.tick(alpha, "TEST", 100.0, 2.0, 0)
        assert b0.state.get("episodic_bias") == 0.0  # empty memory

        b = _brain()
        b._check_eod_penalty = lambda: 1.0
        for _ in range(12):
            b.episodic.add(alpha, 0.02)  # confident positive neighbors
        b.tick(alpha, "TEST", 100.0, 2.0, 0)
        eb = b.state.get("episodic_bias")
        assert eb != 0.0  # tick mirrored the queried modifier into state
        raw_with = b._score_pipeline("TEST", alpha, 1.0, 0.0, eb)["raw_score"]
        raw_without = b._score_pipeline("TEST", alpha, 1.0, 0.0, 0.0)["raw_score"]
        assert eb > 0.0 and raw_with > raw_without


# ── Dead wrapper organs removed (deliberation stays a library) ──────


class TestNoDeadWrappers:
    def test_orchestrator_has_no_unused_deliberator_or_affective(self):
        b = _brain()
        assert not hasattr(b, "deliberator")
        assert not hasattr(b, "affective")

    def test_deliberation_module_still_works(self):
        from hanoon_prime.brain.deliberation import Deliberator, Modifiers

        result = Deliberator(threshold=0.58).deliberate(0.9, 0.6, Modifiers())
        assert result.verdict == "BUY"
        assert result.score > 0


# ── News engine publishes every tracked ticker ──────────────────────


class TestNewsPublishGate:
    def test_news_sentiment_zero_fills_all_tracked_tickers(self, monkeypatch):
        def fake_sentiment(t):
            return (0.25, 3) if t == "AAPL" else (0.0, 0)

        monkeypatch.setattr(ns_mod, "ticker_sentiment", fake_sentiment)
        state = BrainState()
        state.set_latest_alpha(
            {"AAPL": 0.6, "MSFT": 0.5, "NVDA": 0.4, "TSLA": 0.3, "SUNE": 0.2}
        )
        engine = ns_mod.NewsFeedEngine(state, interval=0.0)
        engine.maybe_refresh()
        sent = state.get("news_sentiment") or {}
        assert set(sent) == {"AAPL", "MSFT", "NVDA", "TSLA", "SUNE"}
        assert sent["AAPL"] == 0.25
        assert sent["MSFT"] == 0.0 and sent["NVDA"] == 0.0


# ── Overnight monitor tunnel probe UA ───────────────────────────────


class TestOvernightMonitorUA:
    def _load(self):
        path = Path(__file__).resolve().parent.parent / "scripts/overnight_monitor.py"
        spec = importlib.util.spec_from_file_location("overnight_monitor", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_tunnel_headers_send_browser_ua(self):
        mod = self._load()
        assert mod.TUNNEL_HEADERS["User-Agent"].startswith("Mozilla/5.0")

    def test_check_tunnel_sends_ua(self, monkeypatch):
        mod = self._load()
        captured = {}

        def fake_urlopen(req, timeout=0):
            captured["headers"] = req.headers
            return _FakeResponse()

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        assert mod.check_tunnel() == {}
        ua = next(
            (v for k, v in captured["headers"].items() if k.lower() == "user-agent"),
            "",
        )
        assert ua.startswith("Mozilla/5.0")


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b"{}"
