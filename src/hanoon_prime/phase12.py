"""hanoon_prime.phase12 — daily cross-sectional momentum (sandbox).

Pre-registered successor to phase7/9: instead of scoring the absolute LEVEL
of a composite (falsified in phase 11), it RANKS a cross-sectional spread —
long top-K / short bottom-K (dollar-neutral), held session O→C.

Frozen per task_plan.md §12.3 (K=7, lookback=5); emits wfa.FoldResult so the
existing verdict/deflation/PBO surface applies. No engine mutations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .eyes import load_ohlcv
from .types import F64Array
from .wfa import FoldResult, fold_windows

LOOKBACK: int = 5  # trailing sessions used for the rank signal
K: int = 7  # legs: top-K long / bottom-K short (dollar-neutral)
FOLDS: int = 6


def _session_open_close(
    ts: list[str], open_: F64Array, close: F64Array
) -> tuple[F64Array, F64Array]:
    """First-open/last-close per calendar date from 1-min bars."""
    o_s, c_s, last = [], [], ""
    for t, o, c in zip(ts, open_, close):
        d = t[:10]
        if d != last:
            o_s.append(float(o))
            c_s.append(float(c))
            last = d
        else:
            c_s[-1] = float(c)  # last close of the session wins
    return np.asarray(o_s), np.asarray(c_s)


def load_daily(path: str) -> dict[str, Any]:
    """Reduce one 1-min RTH CSV to per-session OHLC (phase12 feed)."""
    data = load_ohlcv(path)
    o, c = _session_open_close(data["datetime"], data["open"], data["close"])
    return {"open": o, "close": c}


def _signal(soc: F64Array) -> F64Array:
    """Trailing LOOKBACK-session log return per session (feeds NEXT session)."""
    out = np.full(len(soc), np.nan)
    for i in range(LOOKBACK, len(soc)):
        prev = soc[i - LOOKBACK]
        if prev > 0:
            out[i] = float(np.log(soc[i] / prev))
    return out


def _book(scores: F64Array, n: int) -> tuple[F64Array, F64Array] | None:
    """(long_mask, short_mask) for one session from its cross-sectional scores."""
    valid = ~np.isnan(scores)
    if int(valid.sum()) < 2 * K:
        return None
    idx = np.argsort(scores, kind="mergesort")  # ascending (stable/deterministic)
    valid_idx = [i for i in idx if valid[i]]
    long_ = np.zeros(n, dtype=bool)
    short_ = np.zeros(n, dtype=bool)
    for take, mask in ((valid_idx[-K:], long_), (valid_idx[:K], short_)):
        for i in take:
            mask[i] = True
    return long_, short_


def _fold_pnl(
    ticker: str,
    tickers: list[str],
    daily: dict[str, dict[str, Any]],
    per_sess: list[tuple[F64Array, F64Array] | None],
    start: int,
    end: int,
) -> list[float]:
    """O→C pnl for one ticker across one session-fold (book legs only)."""
    close = daily[ticker]["close"]
    open_ = daily[ticker]["open"]
    pnl: list[float] = []
    i = tickers.index(ticker)
    for s in range(max(1, start), min(end, len(close))):
        book = per_sess[s]
        if book is None:
            continue
        sign = 1.0 if book[0][i] else -1.0 if book[1][i] else 0.0
        if sign:
            pnl.append(sign * float(close[s] / open_[s] - 1.0))
    return pnl


def _session_books(
    tickers: list[str], sig_arr: dict[str, F64Array], n_sess: int
) -> list[tuple[F64Array, F64Array] | None]:
    """Book per session (signal from s-1 close; first session never trades)."""
    per_sess: list[tuple[F64Array, F64Array] | None] = [None]
    for s in range(1, n_sess):
        prev = np.asarray([sig_arr[t][s - 1] for t in tickers], dtype=float)
        per_sess.append(_book(prev, len(tickers)))
    return per_sess


def run_spread(
    tickers: list[str], data_dir: str, folds: int = FOLDS
) -> dict[str, list[FoldResult]]:
    """Cross-sectional spread WFA ({ticker: [fold...]}, SESSION-indexed folds).

    Book at session s uses signal from close of s-1 (no lookahead).
    """
    daily: dict[str, dict[str, Any]] = {}
    for t in tickers:
        try:
            daily[t] = load_daily(f"{data_dir}/{t}_1min.csv")
        except (FileNotFoundError, ValueError):
            continue
    tickers = [t for t in tickers if t in daily]
    n_sess = min(len(daily[t]["close"]) for t in tickers)
    if n_sess <= LOOKBACK + folds * 5:
        return {}

    sig_arr = {t: _signal(daily[t]["close"]) for t in tickers}
    per_sess = _session_books(tickers, sig_arr, n_sess)

    out: dict[str, list[FoldResult]] = {}
    for ticker in tickers:
        folds_out = [
            FoldResult(
                fold=k,
                start=start,
                end=end,
                trades=(pnl := _fold_pnl(ticker, tickers, daily, per_sess, start, end)),
                ev_per_trade=float(np.mean(pnl)) if pnl else 0.0,
                sharpe=0.0,
                pnl=pnl,
            )
            for k, (start, end) in enumerate(fold_windows(n_sess, folds))
        ]
        if any(f.trades for f in folds_out):
            out[ticker] = folds_out
    return out


def _decile_cells(
    sig: dict[str, F64Array],
    oc: dict[str, F64Array],
    s: int,
    deciles: int,
) -> list[tuple[float, int]]:
    """(mean O→C, decile) per valid ticker for session s, ranked by signal."""
    vals: list[tuple[float, float]] = []
    for t in sig:
        if s - 1 < len(sig[t]) and not np.isnan(sig[t][s - 1]):
            vals.append((sig[t][s - 1], oc[t][s]))
    if len(vals) < 2 * K or not vals:
        return []
    vals.sort(key=lambda p: p[0])
    n = len(vals)
    out: list[tuple[float, int]] = []
    for d in range(deciles):
        lo, hi = n * d // deciles, n * (d + 1) // deciles
        if hi > lo:
            out.append((sum(v for _, v in vals[lo:hi]) / (hi - lo), d))
    return out


def monotonicity_profile(
    tickers: list[str], data_dir: str, deciles: int = 10
) -> dict[str, float]:
    """Decile O→C v lag-1 signal (d1=weakest); non-monotone → NO-GO."""
    sig, oc, n_ticks = {}, {}, {}
    for t in tickers:
        try:
            daily = load_daily(f"{data_dir}/{t}_1min.csv")
        except (FileNotFoundError, ValueError):
            continue
        sig[t] = _signal(daily["close"])
        oc[t] = daily["close"] / daily["open"] - 1.0
        n_ticks[t] = len(daily["close"])
    if not sig or min(n_ticks.values(), default=0) <= LOOKBACK:
        return {}

    n_sess = min(n_ticks.values())
    cells, denom = [0.0] * deciles, [0] * deciles
    for s in range(1, n_sess):
        for mean_oc, d in _decile_cells(sig, oc, s, deciles):
            cells[d] += mean_oc
            denom[d] += 1
    return {
        f"d{d + 1}": float(cells[d] / denom[d]) if denom[d] else 0.0
        for d in range(deciles)
    }


__all__ = ["LOOKBACK", "K", "FOLDS", "load_daily", "run_spread", "monotonicity_profile"]
