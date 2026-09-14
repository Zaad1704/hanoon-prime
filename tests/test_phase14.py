"""tests/test_phase14.py — PDH/PDL sweep-and-reclaim engine (sandbox).

Verifies the pre-registered protocol (protocols/sweep_reclaim_protocol.md):
reclaim-triggered entries, mirror symmetry, stale-sweep expiry, R:R floor,
the one-trade-per-session cap, the SPY veto hook, and session-space folds.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from hanoon_prime.phase14 import (
    ENTRY_FROM,
    ENTRY_TO,
    FOLDS,
    MIN_RR,
    SESSION_CLOSE,
    STOP_CUSHION_MULT,
    SWEEP_MAX_BARS,
    _session_stats,
    run_sweep,
    sweep_trades,
)

N_QUIET = 14  # quiet sessions building the ATR lookback
LEVEL_DAY = "2026-07-15"  # session whose high/low becomes next session's PDH/PDL
TRADE_DAY = "2026-07-16"


def _minutes(date: str, n: int, start_min: int = 570) -> list[str]:
    """n timestamps from ``start_min`` minutes past midnight (09:30 default)."""
    base = dt.datetime(int(date[:4]), int(date[5:7]), int(date[8:10]))
    base += dt.timedelta(minutes=start_min)
    return [
        (base + dt.timedelta(minutes=k)).strftime("%Y-%m-%d %H:%M:%S") + "-04:00"
        for k in range(n)
    ]


def _quiet_ts(date: str) -> list[str]:
    """130 bars near 100 with TR=1.0 (high 100.5 / low 99.5)."""
    return _minutes(date, 130)


def _data(
    ts: list[str],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None = None,
) -> dict[str, object]:
    v = volumes or [1000.0] * len(ts)
    return {
        "datetime": ts,
        "open": np.asarray(opens),
        "high": np.asarray(highs),
        "low": np.asarray(lows),
        "close": np.asarray(closes),
        "volume": np.asarray(v),
    }


def _chain(
    parts: list[tuple[list[str], list[float], list[float], list[float], list[float]]],
) -> dict[str, object]:
    ts: list[str] = []
    o: list[float] = []
    h: list[float] = []
    l: list[float] = []
    c: list[float] = []
    for p in parts:
        ts += p[0]
        o += p[1]
        h += p[2]
        l += p[3]
        c += p[4]
    return _data(ts, o, h, l, c)


def _quiet(
    date: str,
) -> tuple[list[str], list[float], list[float], list[float], list[float]]:
    ts = _quiet_ts(date)
    return ts, [100.0] * 130, [100.5] * 130, [99.5] * 130, [100.0] * 130


def _levels(
    high: float, low: float
) -> tuple[list[str], list[float], list[float], list[float], list[float]]:
    """Session whose high/low becomes the NEXT session's PDH/PDL."""
    ts = _quiet_ts(LEVEL_DAY)
    return ts, [100.0] * 130, [high] * 130, [low] * 130, [100.0] * 130


def _history(
    n: int = 14,
) -> list[tuple[list[str], list[float], list[float], list[float], list[float]]]:
    return [_quiet(f"2026-07-{day + 1:02d}") for day in range(n)]


def _flat(
    n: int,
    o: list[float],
    h: list[float],
    l: list[float],
    c: list[float],
) -> tuple[list[str], list[float], list[float], list[float], list[float]]:
    return _minutes(TRADE_DAY, n), o, h, l, c


def test_no_trade_without_prior_levels() -> None:
    """A lone session has no prior high/low and must emit nothing."""
    d = _chain([_quiet("2026-08-27")])
    assert sweep_trades(d) == []


def _atr_of(data: dict[str, object], sess: int) -> float:
    """The engine's own ATR_prev value at session ``sess``."""
    _o, _h, _l, _c, _starts, atr = _session_stats(data, list(data["datetime"]))
    return float(atr[sess])


def test_long_sweep_reclaim_fills() -> None:
    """PDL=95: wick below, reclaim over → long at reclaim close, stop under
    the sweep extreme, target = session mid."""
    ts, o, h, l, c = _flat(
        40,
        [100.0] * 40,
        [100.5] * 40,
        [99.5] * 40,
        [100.0] * 40,
    )
    i0 = 30  # 10:00 ET
    l[i0] = 94.5
    c[i0] = 94.8  # sweep bar still below PDL
    c[i0 + 1] = 96.0  # reclaim bar closes above PDL → entry
    d = _chain(_history() + [_levels(105.0, 95.0), (ts, o, h, l, c)])

    trades = sweep_trades(d)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == 1
    assert t.entry == 96.0
    assert t.target == 100.0  # mid = (105 + 95)/2
    assert t.stop == 94.5 - STOP_CUSHION_MULT * _atr_of(d, t.session)
    assert t.exit == 100.0  # ran to mid
    assert t.r == (100.0 - 96.0) / (96.0 - t.stop)


def test_short_sweep_reclaim_mirror() -> None:
    """PDH=110: wick above, reclaim below → short at reclaim close."""
    ts, o, h, l, c = _flat(
        40,
        [100.0] * 40,
        [100.5] * 40,
        [99.5] * 40,
        [100.0] * 40,
    )
    i0 = 30
    h[i0] = 111.0  # wick above PDH
    c[i0] = 110.5
    c[i0 + 1] = 108.0  # reclaim below PDH → entry (high enough to clear MIN_RR)
    d = _chain(_history() + [_levels(110.0, 95.0), (ts, o, h, l, c)])

    trades = sweep_trades(d)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == -1
    assert t.entry == 108.0
    assert t.target == 102.5  # mid = (110 + 95)/2
    assert t.stop == 111.0 + STOP_CUSHION_MULT * _atr_of(d, t.session)


def test_target_hit_r_math() -> None:
    """When price runs to mid, r = (mid-entry)/(entry-stop) in R units."""
    n = 120
    ts, o, h, l, c = _flat(n, [100.0] * n, [100.5] * n, [99.5] * n, [100.0] * n)
    i0 = 30  # 10:00 ET
    l[i0] = 94.5
    c[i0] = 94.8
    c[i0 + 1] = 96.0  # reclaim → entry here
    for k in range(2, 26):  # ramp 96 → mid over 24 bars
        c[i0 + k] = 96.0 + k * 4.0 / 24.0
    d = _chain(_history() + [_levels(105.0, 95.0), (ts, o, h, l, c)])

    trades = sweep_trades(d)
    assert len(trades) == 1
    t = trades[0]
    assert abs(t.entry - 96.0) < 1e-9
    assert t.target == 100.0
    assert t.r > 0
    assert t.exit == 100.0


def test_stale_sweep_expires() -> None:
    """No reclaim within SWEEP_MAX_BARS → sweep dropped, no trade."""
    n = 120
    ts, o, h, l, c = _flat(n, [100.0] * n, [100.5] * n, [99.5] * n, [100.0] * n)
    i0 = 30  # 10:00
    l[i0] = 94.5
    c[i0] = 94.8
    for k in range(1, SWEEP_MAX_BARS + 2):  # stay below PDL past expiry
        c[i0 + k] = 94.5
    c[i0 + SWEEP_MAX_BARS + 3] = 97.0  # reclaim too late
    d = _chain(_history() + [_levels(105.0, 95.0), (ts, o, h, l, c)])
    assert sweep_trades(d) == []


def test_min_rr_floor_rejects_deep_sweep() -> None:
    """Deep wick makes (mid-entry)/(entry-stop) < MIN_RR → no trade."""
    n = 60
    ts, o, h, l, c = _flat(n, [100.0] * n, [100.5] * n, [99.5] * n, [100.0] * n)
    i0 = 30
    l[i0] = 91.0  # very deep sweep
    c[i0] = 94.8
    c[i0 + 1] = 96.0  # reclaim — but stop too far for MIN_RR
    d = _chain(_history() + [_levels(105.0, 95.0), (ts, o, h, l, c)])
    assert sweep_trades(d) == []


def test_one_trade_per_session() -> None:
    """Second setup inside the same session must not fill a second trade."""
    n = 120
    ts, o, h, l, c = _flat(n, [100.0] * n, [100.5] * n, [99.5] * n, [100.0] * n)
    i0 = 30
    l[i0] = 94.5
    c[i0] = 94.8
    c[i0 + 1] = 96.0  # first reclaim → long entry
    c[i0 + 2] = 60.0  # stop hit → trade closed
    j = i0 + 20  # second sweep still inside the entry window
    l[j] = 94.0
    c[j] = 94.5
    c[j + 1] = 97.0
    d = _chain(_history() + [_levels(105.0, 95.0), (ts, o, h, l, c)])
    assert len(sweep_trades(d)) == 1


def test_spy_veto_hook() -> None:
    """spy_ok False blocks the entry; True allows it (same bars)."""
    ts, o, h, l, c = _flat(
        40,
        [100.0] * 40,
        [100.5] * 40,
        [99.5] * 40,
        [100.0] * 40,
    )
    i0 = 30
    l[i0] = 94.5
    c[i0] = 94.8
    c[i0 + 1] = 96.0
    d = _chain(_history() + [_levels(105.0, 95.0), (ts, o, h, l, c)])
    assert sweep_trades(d, spy_ok=lambda _ts: False) == []
    assert len(sweep_trades(d, spy_ok=lambda _ts: True)) == 1


def test_folds_score_trade_in_level_session_fold() -> None:
    """Folds are session-indexed; a trade lands once, in the owning fold."""
    # Needs > EDGE_LOOKBACK sessions (>50) for fold_windows's warmup.
    parts = [
        _quiet(f"2026-{5 + day // 28:02d}-{day % 28 + 1:02d}") for day in range(40)
    ]
    parts += _history() + [_levels(105.0, 95.0)]
    ts, o, h, l, c = _flat(
        40,
        [100.0] * 40,
        [100.5] * 40,
        [99.5] * 40,
        [100.0] * 40,
    )
    i0 = 30
    l[i0] = 94.5
    c[i0] = 94.8
    c[i0 + 1] = 96.0
    parts.append((ts, o, h, l, c))
    d = _chain(parts)

    folds = run_sweep(d, folds=FOLDS)
    assert len(folds) == FOLDS
    direct = sweep_trades(d)
    total = sum(len(f.pnl) for f in folds)
    assert total == len(direct)  # every completed trade is scored exactly once
    assert total >= 1


def test_parameters_are_frozen() -> None:
    """Lock the pre-registered parameter block (sweep_reclaim_protocol.md §6)."""
    assert ENTRY_FROM == "10:00"
    assert ENTRY_TO == "11:30"
    assert SESSION_CLOSE == "15:50"
    assert SWEEP_MAX_BARS == 10
    assert STOP_CUSHION_MULT == 0.25
    assert MIN_RR == 1.0
    assert FOLDS == 6
