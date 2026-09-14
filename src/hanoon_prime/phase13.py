"""Pre-registered control who checks whether §12's flat decile profile was BETA-MASKED rather than absent: trailing 5-session cross-sectional momentum is rescued by stripping each name's rolling SPY beta out of its rank signal (Option A of task_plan."""

from __future__ import annotations

from typing import Any

import numpy as np

from .phase12 import LOOKBACK, K, load_daily
from .wfa import FoldResult, fold_windows

BETA_LOOKBACK: int = 63  # trailing sessions used for the rolling OLS beta
FOLDS: int = 6


def _session_beta(close: np.ndarray, spy_close: np.ndarray) -> np.ndarray:
    out = np.full(len(close), np.nan)
    name5 = _trailing_return(close)
    spy5 = _trailing_return(spy_close)
    for i in range(BETA_LOOKBACK + 1, len(close)):
        x = spy5[i - BETA_LOOKBACK - 1 : i - 1]  # window ending at i−2
        y = name5[i - BETA_LOOKBACK - 1 : i - 1]
        valid = ~(np.isnan(x) | np.isnan(y))
        n = int(valid.sum())
        if n < 20:
            continue
        xv, yv = x[valid], y[valid]
        sxx = float(np.sum(xv * xv))
        if sxx <= 0:
            continue
        sxy = float(np.sum(xv * yv))
        out[i] = sxy / sxx  # beta = slope with intercept forced through origin
    return out


def _trailing_return(close: np.ndarray) -> np.ndarray:
    out = np.full(len(close), np.nan)
    for i in range(LOOKBACK, len(close)):
        prev = close[i - LOOKBACK]
        if prev > 0:
            out[i] = float(close[i] / prev - 1.0)
    return out


def _residual_signal(
    tickers: list[str], daily: dict[str, dict[str, Any]], n_sess: int
) -> dict[str, np.ndarray]:
    spy_close = daily["SPY"]["close"]
    out: dict[str, np.ndarray] = {}
    for t in tickers:
        if t == "SPY":
            continue
        close = daily[t]["close"]
        beta = _session_beta(close, spy_close)
        tr = _trailing_return(close)
        spy5 = _trailing_return(spy_close)
        out[t] = np.full(n_sess, np.nan)
        for s in range(1, n_sess):
            b = beta[s - 1]
            if not (np.isnan(b) or np.isnan(tr[s - 1]) or np.isnan(spy5[s - 1])):
                out[t][s] = tr[s - 1] - b * spy5[s - 1]  # residual at s−1 close
    return out


def _book(scores: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    valid = ~np.isnan(scores)
    if int(valid.sum()) < 2 * K:
        return None
    idx = np.argsort(scores, kind="mergesort")
    valid_idx = [i for i in idx if valid[i]]
    long_ = np.zeros(n, dtype=bool)
    short_ = np.zeros(n, dtype=bool)
    for take, mask in ((valid_idx[-K:], long_), (valid_idx[:K], short_)):
        mask[take] = True
    return long_, short_


def _fold_pnl(
    ticker: str,
    tickers: list[str],
    daily: dict[str, dict[str, Any]],
    per_sess: list[tuple[np.ndarray, np.ndarray] | None],
    start: int,
    end: int,
) -> list[float]:
    close = daily[ticker]["close"]
    open_ = daily[ticker]["open"]
    pnl: list[float] = []
    i = tickers.index(ticker)
    for s in range(max(1, start), min(end, len(close))):
        book = per_sess[s]
        if book is None or not (book[0][i] or book[1][i]):
            continue
        ret = float(close[s] / open_[s] - 1.0)
        if book[0][i]:
            pnl.append(ret)
        else:
            pnl.append(-ret)
    return pnl


def _books(
    tickers: list[str], sig: dict[str, np.ndarray], n_sess: int
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    per_sess: list[tuple[np.ndarray, np.ndarray] | None] = [None]
    for s in range(1, n_sess):
        prev = np.asarray([sig[t][s] for t in tickers], dtype=float)
        per_sess.append(_book(prev, len(tickers)))
    return per_sess


def run_spread(
    tickers: list[str], data_dir: str, folds: int = FOLDS
) -> dict[str, list[FoldResult]]:
    """Book at session s uses the residual signal at close of s−1 (§13."""
    daily: dict[str, dict[str, Any]] = {}
    for t in tickers:
        try:
            daily[t] = load_daily(f"{data_dir}/{t}_1min.csv")
        except (FileNotFoundError, ValueError):
            continue
    tickers = [t for t in tickers if t in daily]
    n_sess = min(len(daily[t]["close"]) for t in tickers)
    if n_sess <= BETA_LOOKBACK + folds * 5:
        return {}

    sig = _residual_signal(tickers, daily, n_sess)
    per_sess = _books(tickers, sig, n_sess)

    out: dict[str, list[FoldResult]] = {}
    for ticker in tickers:
        folds_out = [
            FoldResult(
                fold=k,
                start=start,
                end=end,
                trades=list(
                    pnl := _fold_pnl(ticker, tickers, daily, per_sess, start, end)
                ),
                ev_per_trade=float(np.mean(pnl)) if pnl else 0.0,
                sharpe=0.0,
                pnl=pnl,
            )
            for k, (start, end) in enumerate(fold_windows(n_sess, folds))
        ]
        if any(f.trades for f in folds_out):
            out[ticker] = folds_out
    return out


def monotonicity_profile(
    tickers: list[str], data_dir: str, deciles: int = 10
) -> dict[str, float]:
    """Non-monotone profile → beta-residual momentum is NOT a live candidate; the §13 verdict rig reports NO-GO (same fail-fast as phase12)."""
    sig, oc, n_ticks = {}, {}, {}
    for t in tickers:
        try:
            daily = load_daily(f"{data_dir}/{t}_1min.csv")
        except (FileNotFoundError, ValueError):
            continue
        sig[t] = _trailing_return(daily["close"])
        oc[t] = daily["close"] / daily["open"] - 1.0
        n_ticks[t] = len(daily["close"])
    if not sig or min(n_ticks.values(), default=0) <= LOOKBACK:
        return {}

    n_sess = min(n_ticks.values())
    cells = [0.0] * deciles
    denom = [0] * deciles
    for s in range(1, n_sess):
        vals: list[tuple[float, float]] = []
        for t in sig:
            if s - 1 < len(sig[t]) and not np.isnan(sig[t][s - 1]):
                vals.append((sig[t][s - 1], oc[t][s]))
        if len(vals) < 2 * K or not vals:
            continue
        vals.sort(key=lambda pv: pv[0])
        n = len(vals)
        for d in range(deciles):
            lo, hi = n * d // deciles, n * (d + 1) // deciles
            if hi > lo:
                seg = vals[lo:hi]
                cells[d] += sum(v for _, v in seg) / (hi - lo)
                denom[d] += 1
    return {
        f"d{d + 1}": float(cells[d] / denom[d]) if denom[d] else 0.0
        for d in range(deciles)
    }


__all__ = [
    "BETA_LOOKBACK",
    "FOLDS",
    "LOOKBACK",
    "K",
    "load_daily",
    "run_spread",
    "monotonicity_profile",
]
