"""tests/test_micro_live.py — Phase 5: guarded micro-live deployment.

Guards the GATE, not profitability:
   1. Phase-4 paper FAIL blocks every entry (5.3 completion gate).
   2. A PASS report still requires shadow agreement to authorize.
   3. Kill-switch latch and daily-loss halt mirror the live safety rails
      (latched = manual re-arm, matching SafetyProducer semantics).
   4. The baseline signal is the static-weights path (same snapshot), so a
      live BUY the baseline didn't signal is a recorded divergence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hanoon_prime.eyes import load_ohlcv
from hanoon_prime.immune import DAILY_LOSS_LIMIT, KILL_DAILY_LOSS_LIMIT
from hanoon_prime.micro_live import MicroLiveGuard, ShadowComparison, load_phase4_pass

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def _snapshot(ticker: str = "AAPL") -> dict:
    data = load_ohlcv(FIXTURES / f"{ticker}_1min.csv")
    return {
        "close": list(data["close"][-60:]),
        "close_arr": data["close"][-60:],
        "high": data["high"][-60:],
        "high_arr": data["high"][-60:],
        "low": data["low"][-60:],
        "low_arr": data["low"][-60:],
        "volume": data["volume"][-60:],
        "atr": float(np.std(data["close"][-60:])),
        "last": float(data["close"][-1]),
    }


def _phase4_report(verdict: str, tmp_path: Path) -> Path:
    p = tmp_path / "phase4_paper.json"
    p.write_text(json.dumps({"verdict": verdict}))
    return p


def test_phase4_fail_blocks_all_entries(tmp_path: Path) -> None:
    """5.3 gate: Phase-4 paper FAIL refuses entries even on strong signals."""
    g = MicroLiveGuard(_phase4_report("FAIL", tmp_path))
    assert g.phase4_pass is False
    st = g.guard("AAPL", _snapshot(), "BUY", 0.99)
    assert st.authorized is False
    assert "phase4_paper_not_pass" in st.reason
    assert st.eligible is False


def test_missing_report_is_treated_as_fail(tmp_path: Path) -> None:
    """No committed report is NOT an implicit pass."""
    g = MicroLiveGuard(tmp_path / "missing.json")
    assert g.phase4_pass is False


def test_pass_report_still_needs_shadow_agreement(tmp_path: Path) -> None:
    """PASS report + live signal differing from baseline → divergence block."""
    g = MicroLiveGuard(_phase4_report("PASS", tmp_path))
    assert g.phase4_pass is True
    # Baseline on this snapshot is whatever static weights say; force a
    # deliberate disagreement with a clearly-false loud BUY.
    st = g.guard("AAPL", _snapshot(), "BUY", 0.99)
    if not st.shadow or not st.shadow.agree:
        assert st.authorized is False
        assert "shadow_divergence" in st.reason
        assert len(g.divergences) >= 1
    else:
        # Sanity: if the baseline DID agree, auth requires no other blocker.
        assert st.authorized is True


def test_kill_switch_latches_and_requires_rearm(tmp_path: Path) -> None:
    """KILL_Daily_LOSS_LIMIT breach latches; only rearm() clears it."""
    g = MicroLiveGuard(_phase4_report("PASS", tmp_path))
    g.update_daily_pnl(-(KILL_DAILY_LOSS_LIMIT + 1.0), "AAPL")
    assert g._kill_latched is True  # private, but the mirror state drives auth
    st = g.guard("AAPL", _snapshot(), "BUY", 0.0)
    assert st.authorized is False
    assert "kill_switch_latched" in st.reason
    g.rearm()
    assert g._kill_latched is False
    st2 = g.guard("AAPL", _snapshot(), "BUY", 0.0)
    if st2.shadow and st2.shadow.agree:
        assert st2.authorized is True


def test_daily_loss_halt_blocks_without_latch(tmp_path: Path) -> None:
    """DAILY_LOSS_LIMIT hit blocks entries but does not latch."""
    g = MicroLiveGuard(_phase4_report("PASS", tmp_path))
    g.update_daily_pnl(-(DAILY_LOSS_LIMIT + 1.0), "AAPL")
    assert g._kill_latched is False  # below kill threshold → no latch
    st = g.guard("AAPL", _snapshot(), "BUY", 0.0)
    assert st.authorized is False
    assert st.daily_loss_hit is True
    g.rearm()
    assert g._daily_loss_hit is False


def test_insufficient_snapshot_is_never_authorized(tmp_path: Path) -> None:
    """A snapshot without 20 bars of close data cannot route an entry."""
    g = MicroLiveGuard(_phase4_report("PASS", tmp_path))
    st = g.guard("AAPL", {"last": 1.0}, "BUY", 0.5)
    assert st.authorized is False
    assert st.reason == "insufficient_snapshot"


def test_baseline_signal_is_static_and_deterministic(tmp_path: Path) -> None:
    """Two guards compute the same baseline verdict for the same snapshot."""
    g1 = MicroLiveGuard(_phase4_report("PASS", tmp_path))
    g2 = MicroLiveGuard(_phase4_report("PASS", tmp_path))
    s = _snapshot()
    b1 = g1.baseline_signal("AAPL", s)
    b2 = g2.baseline_signal("AAPL", s)
    assert b1.action == b2.action
    assert abs(b1.score - b2.score) < 1e-12


def test_shadow_comparison_semantics() -> None:
    """Only live directional signals that differ from baseline diverge."""
    y = ShadowComparison("X", "BUY", "BUY", 0.8, 0.8, True)
    assert y.is_divergence is False
    n = ShadowComparison("X", "BUY", "HOLD", 0.8, 0.2, False)
    assert n.is_divergence is True
    h = ShadowComparison("X", "HOLD", "BUY", 0.1, 0.8, False)
    assert h.is_divergence is False  # live HOLD never divergences


def test_load_phase4_pass_module_accessor(tmp_path: Path) -> None:
    assert load_phase4_pass(_phase4_report("PASS", tmp_path)) is True
    assert load_phase4_pass(_phase4_report("FAIL", tmp_path)) is False
