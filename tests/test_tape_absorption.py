"""tests/test_tape_absorption.py — Phase 1/2 tape buffer + detector.

Covers pure tape classification (CVD, side volumes, level holds), the
absorption detector's sign/score rules (live-only empty metrics → {}),
alpha injection from snap keys, absorption-break exit, and the risk
fixed-tick stop/target override.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from hanoon_prime.absorption import (  # noqa: E402
    ABSORPTION_KEY,
    detect_absorption,
    is_absorption_active,
)
from hanoon_prime.brain.exit_checks import check_absorption_break  # noqa: E402
from hanoon_prime.brain.learning_config import (  # noqa: E402
    ABSORPTION_BREAK_FACTOR,
    ABSORPTION_SCALP_MIN,
    ABSORPTION_SIGNAL_MIN,
    ABSORPTION_STOP_ATR,
    ABSORPTION_TARGET_TICKS,
    ABSORPTION_TICK_SIZE,
)
from hanoon_prime.brain.strategy_priors import seeded_priors  # noqa: E402
from hanoon_prime.juli_feed import compute_alpha_from_snap  # noqa: E402
from hanoon_prime.tape import TapeBook, TapeBuffer  # noqa: E402

# ── Phase 1: tape buffer ──────────────────────────────────────────────


def test_tape_classifies_prints_vs_l1():
    """Print at/above ask → +1; at/below bid → −1; mid → 0."""
    buf = TapeBuffer("T")
    buf.record_quote(1.0, 99.99, 100.01)
    buf.record_print(1.1, 100.01, 10.0, 99.99, 100.01)  # aggressive buy
    buf.record_print(1.2, 99.99, 5.0, 99.99, 100.01)  # aggressive sell
    buf.record_print(1.3, 100.00, 7.0, 99.99, 100.01)  # mid
    sides = [p.side for p in buf.prints]
    assert sides == [1, -1, 0]


def test_tape_cvd_and_side_volume():
    """CVD over a warm window reflects signed aggressor volume."""
    buf = TapeBuffer("T")
    base = 1000.0
    # Prints must be strictly increasing in ts (non-monotonic dropped).
    # At/above ask → buy; at/below bid → sell; mid → ignored by CVD.
    buf.record_print(base, 10.01, 30.0, 9.99, 10.01)  # buy 30
    buf.record_print(base + 1.0, 9.99, 10.0, 9.99, 10.01)  # sell 10
    buf.record_print(base + 2.0, 10.01, 5.0, 9.99, 10.01)  # buy 5
    now = base + 2.0
    # buy=35, sell=10 → cvd=(35-10)/45
    cvd = buf.cvd(30.0, now)
    assert abs(cvd - (35.0 - 10.0) / 45.0) < 1e-9
    assert buf.side_volume(1, 30.0, now) == 35.0
    assert buf.side_volume(-1, 30.0, now) == 10.0


def test_tape_metrics_empty_when_cold():
    """No prints (or prints older than the slow window) → empty metrics."""
    buf = TapeBuffer("T")
    assert buf.metrics(1000.0) == {}
    buf.record_print(1.0, 10.0, 10.0, 9.99, 10.01)
    assert buf.metrics(1000.0) == {}  # print is ancient relative to now


def test_tape_level_held_requires_enough_quotes():
    """Fewer hold-poll quotes than the configured length → False."""
    buf = TapeBuffer("T")
    buf.record_quote(1.0, 10.0, 10.02)
    assert buf.level_held(-1, now=1.0) is False


def test_tape_book_registry():
    """for_ticker is get-or-create; drop removes; metrics unknown → {}."""
    book = TapeBook()
    a = book.for_ticker("AAPL")
    assert book.for_ticker("AAPL") is a
    assert book.metrics("NOPE") == {}
    book.drop("AAPL")
    assert book.for_ticker("AAPL") is not a


# ── Phase 2: detector ─────────────────────────────────────────────────


def test_detect_empty_metrics_is_live_only():
    """Empty metrics → {} so the key is absent from bar-only alpha."""
    assert detect_absorption({}) == {}


def test_detect_sell_side_absorption_positive():
    """Heavy sells + bid held + volume dominance → positive absorption."""
    out = detect_absorption(
        {
            "cvd_fast": -0.7,
            "vol_buy": 10.0,
            "vol_sell": 100.0,
            "bid_held": 1.0,
            "ask_held": 0.0,
        }
    )
    assert out[ABSORPTION_KEY] > 0.0
    assert out[ABSORPTION_KEY] <= 1.0


def test_detect_buy_side_absorption_negative():
    """Heavy buys + ask held → negative (MM selling the ask)."""
    out = detect_absorption(
        {
            "cvd_fast": 0.7,
            "vol_buy": 100.0,
            "vol_sell": 10.0,
            "bid_held": 0.0,
            "ask_held": 1.0,
        }
    )
    assert out[ABSORPTION_KEY] < 0.0


def test_detect_requires_level_hold_and_volume_ratio():
    """Level not held, or volume not dominant → zero signal key."""
    no_hold = detect_absorption(
        {
            "cvd_fast": -0.7,
            "vol_buy": 10.0,
            "vol_sell": 100.0,
            "bid_held": 0.0,
            "ask_held": 0.0,
        }
    )
    assert no_hold[ABSORPTION_KEY] == 0.0
    flat_vol = detect_absorption(
        {
            "cvd_fast": -0.7,
            "vol_buy": 50.0,
            "vol_sell": 50.0,
            "bid_held": 1.0,
            "ask_held": 0.0,
        }
    )
    assert flat_vol[ABSORPTION_KEY] == 0.0


def test_is_absorption_active_floor():
    """Active only at/above the provided floor."""
    between = 0.20  # SIGNAL_MIN(0.15) < 0.20 < SCALP_MIN(0.25)
    alpha = {ABSORPTION_KEY: between}
    assert is_absorption_active(alpha, ABSORPTION_SIGNAL_MIN) is True
    assert is_absorption_active(alpha, ABSORPTION_SCALP_MIN) is False
    assert is_absorption_active({}, ABSORPTION_SIGNAL_MIN) is False
    strong = {ABSORPTION_KEY: 0.8}
    assert is_absorption_active(strong, ABSORPTION_SCALP_MIN) is True


def test_alpha_injection_from_snap_and_absent_when_bar_only():
    """Snap with tape keys injects absorption; bar-only snap leaves it out."""
    bars = list(range(1, 30))
    live = {
        "close_arr": bars,
        "high_arr": bars,
        "low_arr": bars,
        "volume_arr": [100.0] * 29,
        "buy_volume_arr": [50.0] * 29,
        "bid_sizes_arr": [10.0] * 29,
        "ask_sizes_arr": [10.0] * 29,
        "cvd_fast": -0.7,
        "vol_buy": 10.0,
        "vol_sell": 100.0,
        "bid_held": 1.0,
        "ask_held": 0.0,
    }
    alpha = compute_alpha_from_snap(live, ticker="T")
    assert ABSORPTION_KEY in alpha
    assert alpha[ABSORPTION_KEY] > 0.0
    tape_keys = ("cvd_fast", "vol_buy", "vol_sell", "bid_held", "ask_held")
    bar_only = {k: v for k, v in live.items() if k not in tape_keys}
    alpha_bar = compute_alpha_from_snap(bar_only, ticker="T")
    assert ABSORPTION_KEY not in alpha_bar


# ── Phase 4: break exit + risk override + prior ───────────────────────


def test_absorption_break_sign_flip_and_decay():
    """Sign flip or collapse below break-factor×entry exits; floor gates."""
    assert check_absorption_break(0.0, 0.0).should_exit is False
    strong = detect_absorption(
        {
            "cvd_fast": -0.9,
            "vol_buy": 5.0,
            "vol_sell": 200.0,
            "bid_held": 1.0,
            "ask_held": 0.0,
        }
    )[ABSORPTION_KEY]
    assert strong >= ABSORPTION_SIGNAL_MIN
    assert check_absorption_break(strong, -strong).should_exit is True
    assert check_absorption_break(strong, 0.0).should_exit is True
    decayed = ABSORPTION_BREAK_FACTOR * strong * 0.5
    assert check_absorption_break(strong, decayed).should_exit is True
    held = strong * 0.9
    assert check_absorption_break(strong, held).should_exit is False


def test_risk_absorption_fixed_tick_override():
    """Strong absorption overrides stop/target with fixed-tick scalp profile."""
    from hanoon_prime.brain.risk import RiskEngine  # noqa: PLC0415

    mgr = RiskEngine()
    score = 0.8
    entry = 100.0
    atr = 2.0
    plain = mgr.evaluate(score, 0.7, entry, atr, 0, alpha=None, ticker="T")
    strong_alpha = {ABSORPTION_KEY: 0.8}
    ab = mgr.evaluate(score, 0.7, entry, atr, 0, alpha=strong_alpha, ticker="T")
    expected_target = round(
        entry + 1 * ABSORPTION_TARGET_TICKS * ABSORPTION_TICK_SIZE, 2
    )
    expected_stop = round(entry - 1 * ABSORPTION_STOP_ATR * atr, 2)
    assert ab.target_price == expected_target
    assert ab.stop_price == expected_stop
    assert plain.risk_pass is True and ab.risk_pass is True
    # Without absorption the target is ATR-based (not the fixed 8-tick), so
    # the override must have actually fired for the absorption case.
    assert plain.target_price != ab.target_price or plain.stop_price != ab.stop_price


def test_seeded_priors_include_mm_absorption():
    """The 4th seeded prior is present with scalp/absorption semantics."""
    names = [p["name"] for p in seeded_priors()]
    assert "mm-absorption" in names
    prior = next(p for p in seeded_priors() if p["name"] == "mm-absorption")
    assert prior["source"] == "seeded"
    assert prior["score_mod"] > 0.0
    from hanoon_prime.brain.learning_config import (  # noqa: PLC0415
        STRATEGY_SCORE_MOD_BOUND,
    )

    assert prior["score_mod"] <= STRATEGY_SCORE_MOD_BOUND


def test_absorption_weight_is_tied_top_and_sum_in_bounds():
    """Absorption sits at the top tier; DEFAULT_WEIGHTS stays repair-safe."""
    from hanoon_prime.brain.config import DEFAULT_WEIGHTS  # noqa: PLC0415

    w = DEFAULT_WEIGHTS["absorption"]
    assert w >= 0.10
    assert w == max(DEFAULT_WEIGHTS.values())
    total = sum(DEFAULT_WEIGHTS.values())
    assert 0.90 <= total <= 1.10


def test_absorption_score_mod_bounded_signed():
    """Score boost only at the scalp floor, only when flow-aligned."""
    from hanoon_prime.brain.learning_config import (  # noqa: PLC0415
        ABSORPTION_SCALP_MIN,
        ABSORPTION_SCORE_MOD,
    )
    from hanoon_prime.brain.orchestrator import NeuromorphicBrain  # noqa: PLC0415

    mod = NeuromorphicBrain._absorption_score_mod
    assert mod({}, 0.1, score=0.5) == 0.1
    assert mod({ABSORPTION_KEY: 0.10}, 0.1, score=0.5) == 0.1  # below scalp floor
    # Flow-aligned: +abs with positive score, −abs with negative score.
    assert mod({ABSORPTION_KEY: 1.0}, 0.0, score=0.5) == ABSORPTION_SCORE_MOD
    assert mod({ABSORPTION_KEY: -1.0}, 0.0, score=-0.5) == -ABSORPTION_SCORE_MOD
    # Counter-flow / zero score: no boost (Design A).
    assert mod({ABSORPTION_KEY: 1.0}, 0.0, score=-0.5) == 0.0
    assert mod({ABSORPTION_KEY: -1.0}, 0.0, score=0.5) == 0.0
    assert mod({ABSORPTION_KEY: 1.0}, 0.0, score=0.0) == 0.0
    at_floor = ABSORPTION_SCALP_MIN
    boosted = mod({ABSORPTION_KEY: at_floor}, 0.0, score=0.4)
    assert abs(boosted) <= ABSORPTION_SCORE_MOD + 1e-12
    assert abs(boosted) > 0.0


# ── Design A: absorption flow gate ────────────────────────────────────


def test_absorption_flow_direction_sign_and_floor():
    """Flow side is sign(abs) at/above floor; 0 when inactive/missing."""
    from hanoon_prime.absorption import absorption_flow_direction  # noqa: PLC0415

    assert absorption_flow_direction({}, ABSORPTION_SIGNAL_MIN) == 0
    assert (
        absorption_flow_direction({ABSORPTION_KEY: 0.20}, ABSORPTION_SCALP_MIN) == 0
    )  # SIGNAL_MIN < 0.20 < SCALP_MIN
    assert absorption_flow_direction({ABSORPTION_KEY: 0.20}, ABSORPTION_SIGNAL_MIN) == 1
    assert (
        absorption_flow_direction({ABSORPTION_KEY: -0.8}, ABSORPTION_SIGNAL_MIN) == -1
    )
    assert absorption_flow_direction({ABSORPTION_KEY: 1.0}, ABSORPTION_SIGNAL_MIN) == 1


def test_absorption_flow_gate_blocks_counter_cortex():
    """Active absorption + opposite cortex direction → block; else pass."""
    from hanoon_prime.absorption import absorption_flow_blocked  # noqa: PLC0415

    floor = ABSORPTION_SIGNAL_MIN
    # +abs = MM buying = long flow; short cortex opposes → block.
    assert absorption_flow_blocked({ABSORPTION_KEY: 0.8}, -1, floor) is True
    assert absorption_flow_blocked({ABSORPTION_KEY: 0.8}, 1, floor) is False
    # −abs = MM selling = short flow; long cortex opposes → block.
    assert absorption_flow_blocked({ABSORPTION_KEY: -0.8}, 1, floor) is True
    assert absorption_flow_blocked({ABSORPTION_KEY: -0.8}, -1, floor) is False
    # Inactive / missing / zero direction → never blocks.
    assert absorption_flow_blocked({}, -1, floor) is False
    assert absorption_flow_blocked({ABSORPTION_KEY: 0.10}, -1, floor) is False
    assert absorption_flow_blocked({ABSORPTION_KEY: 0.8}, 0, floor) is False


def test_decide_entry_flow_rejects_counter_absorption():
    """decide_entry vetoes cortex that fights active absorption (Design A)."""
    import time
    from dataclasses import replace
    from types import SimpleNamespace

    from hanoon_prime.absorption import ABSORPTION_KEY
    from hanoon_prime.brain.orchestrator import NeuromorphicBrain
    from hanoon_prime.brain.policy.verdict import VETOED
    from hanoon_prime.brain.shared_state import DEFAULT_POLICY_STATE

    b = NeuromorphicBrain(enable_neuromorphic=False)
    # Sell-side absorption (+) = bullish flow; cortex says SELL → block.
    b.tick = lambda alpha, ticker, **kw: {
        "price": 100.0,
        "verdict": "SELL",
        "signals": {"entry": 1.0},
        "thought": SimpleNamespace(score=-0.9, direction=-1, confidence=0.5),
    }
    b.state.update(
        policy_state={
            **DEFAULT_POLICY_STATE,
            "equity_synced": True,
            "equity": 100_000.0,
        },
        account_feed={"equity": 100_000.0, "daily_pnl": 0.0, "positions": {}},
    )
    # direction_mode both so only the flow gate can veto this short.
    b.trading_policy = replace(b.trading_policy, direction_mode="both")
    b._last_alpha["NVD"] = {ABSORPTION_KEY: 0.8}
    snap = {
        "prices": [1.0] * 40,
        "bid": 100.0,
        "ask": 100.1,
        "mid": 100.05,
        "last": 100.0,
        "ts": time.time(),
        "volume": 1000,
    }
    v = b.decide_entry("NVD", snap, {}, "rth")
    assert v.action == VETOED
    assert v.reason == "flow_rejected"
    assert v.score == -0.9  # cortex conviction survives onto the veto


def test_decide_entry_flow_allows_aligned_absorption():
    """Aligned cortex + active absorption is not flow-vetoed (Design A)."""
    import time
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import patch

    from hanoon_prime.absorption import ABSORPTION_KEY
    from hanoon_prime.brain.orchestrator import NeuromorphicBrain
    from hanoon_prime.brain.policy.verdict import ENTER
    from hanoon_prime.brain.shared_state import DEFAULT_POLICY_STATE

    b = NeuromorphicBrain(enable_neuromorphic=False)
    b.tick = lambda alpha, ticker, **kw: {
        "price": 100.0,
        "verdict": "BUY",
        "signals": {"entry": 1.0},
        "thought": SimpleNamespace(score=0.9, direction=1, confidence=0.5),
    }
    b.state.update(
        policy_state={
            **DEFAULT_POLICY_STATE,
            "equity_synced": True,
            "equity": 100_000.0,
        },
        account_feed={"equity": 100_000.0, "daily_pnl": 0.0, "positions": {}},
    )
    b.trading_policy = replace(b.trading_policy, direction_mode="both")
    b._last_alpha["NVD"] = {ABSORPTION_KEY: 0.8}
    snap = {
        "prices": [1.0] * 40,
        "bid": 100.0,
        "ask": 100.1,
        "mid": 100.05,
        "last": 100.0,
        "ts": time.time(),
        "volume": 1000,
    }
    with patch("hanoon_prime.brain.orchestrator.META_DNN_ENABLED", False):
        v = b.decide_entry("NVD", snap, {}, "rth")
    assert v.action == ENTER
    assert v.reason != "flow_rejected"
