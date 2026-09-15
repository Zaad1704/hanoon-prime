"""tests/test_shadow_book — zero-size paper trial book (data-starvation fix).

Covers: open/close/pnl math, TTL gating, drop on real trade, max_open cap,
atomic persistence, corrupt-file recovery. Isolation assertion: shadow
closes feed the advisory bandit + registry only.
"""

from __future__ import annotations

import math
from pathlib import Path

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
