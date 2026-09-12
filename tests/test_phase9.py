"""tests/test_phase9.py — Phase 9: scheduled-earnings catalyst screen (Path A).

Verifies the Phase-9 sandbox funnel while keeping the lean stack untouched:
  * pre-market RVOL flags use ONLY strictly-prior sessions (no lookahead)
  * Before-Market-Open earnings map to the same day; After-Market-Close to
    the next trading session in the panel
  * the catalyst gate composes regime x pre-market-earnings sessions
  * ``run_walk_forward_catalyst`` is byte-identical to
    ``run_walk_forward_lean`` when every session is allowed, and empty folds
    otherwise (no trades = INSUFFICIENT, never fabricated).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hanoon_prime.eyes import load_ohlcv
from hanoon_prime.phase7 import LeanCfg, run_walk_forward_lean
from hanoon_prime.phase9 import (
    earnings_session_dates,
    load_earnings,
    make_catalyst_hooks,
    premkt_flagged_dates,
    run_walk_forward_catalyst,
)

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def _premkt_csv(path: Path, ticker: str, sessions: list[tuple[str, float]]) -> None:
    """Write an eyes-format pre-market CSV: one 08:00 bar per session."""
    lines = [
        "Price,Close,High,Low,Open,Volume",
        f"Ticker,{ticker},{ticker},{ticker},{ticker},{ticker}",
        "Datetime,,,,",
    ]
    for date_key, vol in sessions:
        ts = f"{date_key} 08:00:00-04:00"
        lines.append(f"{ts},100.0,100.5,99.5,99.5,{int(vol)}")
    path.write_text("\n".join(lines) + "\n")


def test_premkt_rvol_flags_use_prior_sessions_only(tmp_path: Path) -> None:
    """A 6x volume spike flags its session; quiet ones after it do not."""
    sessions = [
        ("2026-05-01", 100),
        ("2026-05-04", 100),
        ("2026-05-05", 100),
        ("2026-05-06", 100),
        ("2026-05-07", 100),
        ("2026-05-08", 600),
        ("2026-05-11", 100),
        ("2026-05-12", 100),
    ]
    csv = tmp_path / "TST_1min.csv"
    _premkt_csv(csv, "TST", sessions)
    flags = premkt_flagged_dates(csv, min_rvol=5.0, period=3)
    assert flags == {"2026-05-08"}
    assert "2026-05-07" not in flags  # warmup < period observations
    monkey = tmp_path / "HOT_1min.csv"
    _premkt_csv(
        monkey, "HOT", [("2026-05-01", 100), ("2026-05-04", 100), ("2026-05-05", 1000)]
    )
    assert premkt_flagged_dates(monkey, min_rvol=5.0, period=3) == set()


def test_earnings_session_mapping_bmo_amc_and_outside() -> None:
    """BMO -> same day; AMC at/after 16:00 -> next panel session; outside -> dropped."""
    rth = ["2026-05-01", "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07"]
    earnings = [
        "2026-05-05T06:00:00-04:00",  # BMO -> same day
        "2026-05-05T17:00:00-04:00",  # AMC -> next session
        "2026-05-04T16:00:00-04:00",  # AMC exactly -> next session
        "2026-04-30T17:00:00-04:00",  # AMC -> 2026-05-01 (panel start)
        "2026-05-09T17:00:00-04:00",  # Saturday; next session outside rth -> dropped
    ]
    assert earnings_session_dates(earnings, rth) == {
        "2026-05-05",
        "2026-05-06",
        "2026-05-01",
    }


def test_catalyst_gate_blocks_non_catalyst_sessions() -> None:
    """With regime gate off, the gate is exactly the catalyst mask."""
    times = ["2026-05-05 09:30:00-04:00", "2026-05-06 09:30:00-04:00"]
    mask = np.asarray([False, True], dtype=bool)
    bars = np.asarray([1.0, 2.0])
    # make_catalyst_hooks needs a BarSeries for RS/gate; use a tiny stub.
    hook = make_catalyst_hooks(times, use_gate=False, catalyst_ok=mask)
    assert not hook.gate(0, None, times[0], _StubBars(bars))
    assert hook.gate(1, None, times[1], _StubBars(bars))
    hook_all = make_catalyst_hooks(times, use_gate=False)
    assert hook_all.gate(0, None, times[0], _StubBars(bars))
    assert hook_all.gate(1, None, times[1], _StubBars(bars))


def test_catalyst_transparent_when_all_sessions_allowed() -> None:
    """Allow-every-session must reproduce the lean stack trade-for-trade."""
    from hanoon_prime.phase9 import _sorted_date_keys

    data = load_ohlcv(FIXTURES / "AAPL_1min.csv")
    cfg = LeanCfg(folds=4)
    allowed = set(_sorted_date_keys(list(data["datetime"])))
    base = run_walk_forward_lean("AAPL", data, cfg)
    gated = run_walk_forward_catalyst("AAPL", data, cfg, allowed)
    base_pnl = [p for f in base for p in f.pnl]
    gated_pnl = [p for f in gated for p in f.pnl]
    assert base_pnl == gated_pnl
    assert len(base_pnl) > 0  # sanity: portal fixtures actually trade


def test_catalyst_no_allowed_dates_makes_no_trades() -> None:
    """An empty catalyst universe yields empty folds (INSUFFICIENT), never fake trades."""
    data = load_ohlcv(FIXTURES / "AAPL_1min.csv")
    folds = run_walk_forward_catalyst("AAPL", data, LeanCfg(folds=4), set())
    assert all(f.trades == [] for f in folds)
    assert all(f.ev_per_trade == 0.0 for f in folds)


def test_load_earnings_missing_returns_empty(tmp_path: Path) -> None:
    """Missing earnings file degrades to an empty calendar, not an error."""
    assert load_earnings("NOPE", tmp_path) == []


class _StubBars:
    """Minimal BarSeries stand-in for the gate's RVOL math."""

    def __init__(self, close: np.ndarray) -> None:
        self.close = close
        self.volume = np.ones_like(close)


def test_sorted_date_keys_order() -> None:
    """Session dates come back in first-seen (chronological) order."""
    from hanoon_prime.phase9 import _sorted_date_keys

    times = [
        "2026-05-05 09:30:00-04:00",
        "2026-05-05 09:31:00-04:00",
        "2026-05-06 09:30:00-04:00",
        "2026-05-05 09:32:00-04:00",
    ]
    assert _sorted_date_keys(times) == ["2026-05-05", "2026-05-06"]
