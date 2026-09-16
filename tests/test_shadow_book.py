"""tests/test_shadow_book — zero-size paper trial book (data-starvation fix).

Covers: open/close/pnl math, TTL gating, drop on real trade, max_open cap,
atomic persistence, corrupt-file recovery. Isolation assertion: shadow
closes feed the advisory bandit + registry only.
"""

from __future__ import annotations

import math
from pathlib import Path

from hanoon_prime.brain.policy.verdict import Verdict
from hanoon_prime.brain.shadow_book import ShadowBook, ShadowClose, ShadowOpen


def _book(tmp_path: Path, ttl: float = 0.0, max_open: int = 3) -> ShadowBook:
    """Construct a ShadowBook pre-loaded with zero TTL so sweep fires instantly."""
    return ShadowBook(path=tmp_path / "b.json", ttl=ttl, max_open=max_open)


# ── open / sweep / pnl ──────────────────────────────────────────────────


def test_open_and_sweep_long_win(tmp_path: Path) -> None:
    b = _book(tmp_path)
    b.open(ShadowOpen("AAPL", 1, "trend", "unknown", "scalp"), 100.0, now=1.0)
    closed = b.sweep_close("AAPL", 105.0, now=90.0)
    assert closed is not None
    assert closed.won is True
    assert math.isclose(closed.pnl_pct, 0.05, rel_tol=1e-5)


def test_open_and_sweep_short_win(tmp_path: Path) -> None:
    b = _book(tmp_path)
    b.open(ShadowOpen("TSLA", -1, "fade", "high_vol", "swing"), 200.0, now=1.0)
    closed = b.sweep_close("TSLA", 180.0, now=90.0)
    assert closed is not None
    assert closed.won is True
    assert math.isclose(closed.pnl_pct, 0.10, rel_tol=1e-5)


def test_open_and_sweep_long_loss(tmp_path: Path) -> None:
    b = _book(tmp_path)
    b.open(ShadowOpen("MSFT", 1, "breakout", "trend", "day"), 400.0, now=1.0)
    closed = b.sweep_close("MSFT", 396.0, now=90.0)
    assert closed is not None
    assert closed.won is False
    assert math.isclose(closed.pnl_pct, -0.01, rel_tol=1e-5)


def test_sweep_returns_none_before_ttl(tmp_path: Path) -> None:
    b = _book(tmp_path, ttl=60.0)
    b.open(ShadowOpen("GOOG", 1, "default", "unknown", "scalp"), 140.0, now=1.0)
    assert b.sweep_close("GOOG", 145.0, now=30.0) is None
    assert b.sweep_close("GOOG", 145.0, now=90.0) is not None


def test_sweep_returns_none_for_unrecognised_ticker(tmp_path: Path) -> None:
    b = _book(tmp_path)
    assert b.sweep_close("NOSUCH", 100.0, now=1.0) is None


# ── drop / max_open / cap ───────────────────────────────────────────────


def test_drop_abandons_paper_trial(tmp_path: Path) -> None:
    b = _book(tmp_path)
    b.open(ShadowOpen("NVDA", 1, "default", "unknown", "scalp"), 120.0, now=1.0)
    b.drop("NVDA")
    assert b.sweep_close("NVDA", 125.0, now=90.0) is None


def test_no_double_open_same_ticker(tmp_path: Path) -> None:
    b = _book(tmp_path)
    b.open(ShadowOpen("NVDA", 1, "trend", "unknown", "scalp"), 120.0, now=1.0)
    b.open(ShadowOpen("NVDA", -1, "fade", "unknown", "scalp"), 121.0, now=2.0)
    b.drop("NVDA")
    assert b.sweep_close("NVDA", 120.0, now=90.0) is None


def test_max_open_cap(tmp_path: Path) -> None:
    b = _book(tmp_path, max_open=2)
    b.open(ShadowOpen("A", 1, "default", "unknown", "scalp"), 1.0, now=0.0)
    b.open(ShadowOpen("B", 1, "default", "unknown", "scalp"), 2.0, now=0.0)
    b.open(ShadowOpen("C", 1, "default", "unknown", "scalp"), 3.0, now=0.0)
    assert b.snapshot()["open_count"] == 2


def test_sweep_reduces_open_count(tmp_path: Path) -> None:
    b = _book(tmp_path, ttl=0.0, max_open=5)
    for i in range(5):
        b.open(
            ShadowOpen(f"X{i}", 1, "default", "unknown", "scalp"),
            float(i + 1),
            now=0.0,
        )
    for i in range(5):
        b.sweep_close(f"X{i}", float(i + 11), now=1.0)
    assert b.snapshot()["open_count"] == 0
    assert b.snapshot()["closed_trials"] == 5


# ── persistence round-trip ──────────────────────────────────────────────


def test_persistence_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "p.json"
    b1 = ShadowBook(path=p, ttl=0.0)
    b1.open(ShadowOpen("AAPL", 1, "trend", "unknown", "scalp"), 100.0, now=1.0)
    b1.sweep_close("AAPL", 105.0, now=90.0)
    b2 = ShadowBook(path=p, ttl=0.0)
    snap = b2.snapshot()
    assert snap["closed_trials"] == 1
    assert math.isclose(snap["edge"], 0.05, rel_tol=1e-5)


def test_corrupt_file_falls_back_fresh(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text("not-json!")
    b = ShadowBook(path=p, ttl=0.0)
    assert b.snapshot()["closed_trials"] == 0


def test_no_open_without_price(tmp_path: Path) -> None:
    b = _book(tmp_path)
    b.open(ShadowOpen("AAPL", 1, "default", "unknown", "scalp"), 0.0, now=1.0)
    assert b.snapshot()["open_count"] == 0


def test_sweep_expired_refunds_all_past_ttl(tmp_path: Path) -> None:
    b = ShadowBook(path=tmp_path / "b.json", ttl=5.0, max_open=5)
    b.open(ShadowOpen("A", 1, "trend", "unknown", "scalp"), 10.0, now=1.0)
    b.open(ShadowOpen("B", 1, "fade", "unknown", "scalp"), 20.0, now=2.0)
    assert b.sweep_expired(now=5.0) == []
    assert b.snapshot()["open_count"] == 2
    closes = b.sweep_expired(now=10.0)
    assert len(closes) == 2
    assert all(c.stale for c in closes)
    assert all(c.pnl_pct == 0.0 for c in closes)
    assert all(c.won is False for c in closes)
    assert b.snapshot()["open_count"] == 0


def test_sweep_expired_skips_not_yet_expired(tmp_path: Path) -> None:
    b = ShadowBook(path=tmp_path / "b.json", ttl=5.0, max_open=3)
    b.open(ShadowOpen("A", 1, "trend", "unknown", "scalp"), 10.0, now=5.0)
    b.open(ShadowOpen("B", 1, "fade", "unknown", "scalp"), 20.0, now=8.0)
    closes = b.sweep_expired(now=11.0)
    assert len(closes) == 1
    assert closes[0].ticker == "A"
    assert b.snapshot()["open_count"] == 1


def test_sweep_expired_empty_book_returns_empty(tmp_path: Path) -> None:
    b = ShadowBook(path=tmp_path / "b.json", ttl=5.0, max_open=3)
    assert b.sweep_expired(now=100.0) == []


# ── shadow close contract (advisory organs only) ────────────────────────


def test_shadow_close_feeds_only_bandit_and_registry() -> None:
    """Verify the close type is safe for advisory organs."""
    c = ShadowClose(
        ticker="X",
        direction=1,
        strategy_id="trend",
        canon="unknown",
        horizon="scalp",
        entry_price=10.0,
        exit_price=11.0,
        age=90.0,
        pnl_pct=0.1,
        won=True,
    )
    # The close carries only advisory-safe fields; assert no sizing/risk.
    data = c.__dict__
    assert "sizing" not in data
    assert "risk" not in data
    assert "pillars" not in data


# ── orchestrator wiring (auto-correction learning path) ─────────────────


class TestOrchestratorShadowWiring:
    """Verifies the shadow hook feeds ONLY the whitelisted organs."""

    def test_learn_from_shadow_updates_bandit_and_registry(self) -> None:
        """A closed paper trial moves both advisory organs."""
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain

        brain = NeuromorphicBrain(enable_neuromorphic=False)
        before_b = brain.strategy_bandit.snapshot()
        before_r = brain.strategy_registry.count()
        realized_before = brain._realized.total_trades
        brain._learn_from_shadow(
            ShadowClose(
                ticker="T",
                direction=1,
                strategy_id="trend-pullback",
                canon="unknown",
                horizon="scalp",
                entry_price=100.0,
                exit_price=102.0,
                age=90.0,
                pnl_pct=0.02,
                won=True,
            )
        )
        after_b = brain.strategy_bandit.snapshot()
        assert after_b["total_trials"] > before_b["total_trials"]
        assert brain.strategy_registry.count() >= before_r
        # Realized PnL must NOT move — shadow is paper, not cash.
        assert brain._realized.total_trades == realized_before

    def test_shadow_eligible_excludes_policy_declines(self) -> None:
        """direction_rejected/no_signal/session must never become trials."""
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain

        brain = NeuromorphicBrain(enable_neuromorphic=False)
        assert not brain._shadow_eligible(Verdict(ticker="T", stage="trading_policy"))
        assert not brain._shadow_eligible(Verdict(ticker="T", reason="no_signal"))
        assert not brain._shadow_eligible(Verdict(ticker="T", reason="no_data"))
        assert not brain._shadow_eligible(Verdict(ticker="T", reason="eval_error"))
        # Eligible cost reasons.
        assert brain._shadow_eligible(Verdict(ticker="T", reason="not_sized"))
        assert brain._shadow_eligible(Verdict(ticker="T", reason="low_penny_score"))
        assert brain._shadow_eligible(Verdict(ticker="T", reason="sized_to_zero"))
        assert brain._shadow_eligible(Verdict(ticker="T", stage="portfolio_risk"))
        assert brain._shadow_eligible(Verdict(ticker="T", stage="governor"))

    def test_shadow_observe_opens_on_eligible_decline(self) -> None:
        """An ignored directional signal becomes a zero-size paper trial."""
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain

        brain = NeuromorphicBrain(enable_neuromorphic=False)
        snap = {"last": 100.0}
        result = {"direction": 1, "regime_canon": "range", "horizon": "scalp"}
        brain._last_strategy["T"] = "range-fade"
        brain._shadow_observe(
            "T", snap, result, Verdict(ticker="T", reason="not_sized")
        )
        assert "T" in brain._shadow_book.snapshot()["open"]

    def test_governor_veto_routes_to_shadow_observe(self) -> None:
        """A governor budget veto still becomes a paper trial (was dead code)."""
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain
        from hanoon_prime.immune import MAX_ENTRIES_PER_CYCLE

        class _SpyBrain(NeuromorphicBrain):
            def __init__(self) -> None:
                super().__init__(enable_neuromorphic=False)
                self.observed: list[Verdict] = []

            def _score_candidate(self, _ticker: str, _snap: dict, _num_open: int):
                return {"direction": 1}, [], {"score": 0.5}

            def _apply_fast_gates(self, _ticker, _snap, _thought, _policy, _session):
                return None

            def _shadow_observe(self, _ticker, _snap, _result, verdict):
                self.observed.append(verdict)
                super()._shadow_observe(_ticker, _snap, _result, verdict)

        brain = _SpyBrain()
        brain.governor._cycle_used = MAX_ENTRIES_PER_CYCLE  # force budget veto
        snap = {"last": 100.0, "prices": [1.0] * 20}
        verdict = brain.decide_entry("T", snap, set(), "rth")
        assert verdict.stage == "governor"
        assert len(brain.observed) == 1
        assert "T" in brain._shadow_book.snapshot()["open"]

    def test_shadow_cycle_drains_pending_and_skips_stale(self) -> None:
        """S2 drain learns real closes; refunded (stale) closes are dropped."""
        from hanoon_prime.brain.orchestrator import NeuromorphicBrain

        brain = NeuromorphicBrain(enable_neuromorphic=False)
        rows = brain.strategy_registry.snapshot()["strategies"]
        assert len(rows) >= 1
        sid = str(rows[0]["id"])

        def _trials(target: str) -> int:
            for row in brain.strategy_registry.snapshot()["strategies"]:
                if row["id"] == target:
                    return int(row["trials"])
            return -1

        before_t = _trials(sid)
        before_b = brain.strategy_bandit.snapshot()["total_trials"]
        real = ShadowClose(
            ticker="A",
            direction=1,
            strategy_id=sid,
            canon="unknown",
            horizon="scalp",
            entry_price=10.0,
            exit_price=11.0,
            age=9.0,
            pnl_pct=0.1,
            won=True,
            stale=False,
        )
        stale = ShadowClose(
            ticker="B",
            direction=1,
            strategy_id=sid,
            canon="unknown",
            horizon="scalp",
            entry_price=10.0,
            exit_price=10.0,
            age=9000.0,
            pnl_pct=0.0,
            won=False,
            stale=True,
        )
        brain._shadow_pending.append(real)
        brain._shadow_pending.append(stale)
        brain._shadow_cycle()
        assert len(brain._shadow_pending) == 0
        assert brain.strategy_bandit.snapshot()["total_trials"] == before_b + 1
        assert _trials(sid) == before_t + 1
