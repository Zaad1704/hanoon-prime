"""hanoon_prime.wfa — walk-forward validation + anti-overfit statistics.

The Phase-1 backtest proved honest but too thin (5 days) and it never
**fits anything** — ``simulate_ticker`` runs static INDICATOR_WEIGHTS
(live-side learning is a separate loop). That means no train/test leakage
inside one backtest run, but it does NOT mean the weights were not selected
elsewhere (e.g. eyeballed on other data). Phase 2 answers the selection-
bias question with:

  * contiguous walk-forward folds over each ticker's bars
    (trades are counted only inside a fold — no lookback across folds)
  * a Monte-Carlo deflated Sharpe that penalises the number of trials
    (tickers × folds) tried before reporting an edge
  * CSCV probability-of-backtest-overfitting (Bailey/López de Prado)

Rules of the gate:
  * a ticker needs ``MIN_TRADES`` completed OOS trades before its EV is
    admissible, else verdict = INSUFFICIENT (which FAILS, never passes)
  * the universe passes only if deflated OOS Sharpe is significantly
    positive across all admissible tickers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from .eyes import load_ohlcv
from .hands import simulate_ticker
from .immune import EDGE_LOOKBACK
from .types import BarSeries

# ── Tunables (locked once the Phase-4 protocol is registered) ───────────
MIN_TRADES: int = 30  # OOS trades required before EV is admissible
DEFAULT_FOLDS: int = 6  # contiguous OOS folds per ticker
BOOTSTRAP_ITERS: int = 2000
RNG = np.random.default_rng(0)  # deterministic deflation/re-sampling


@dataclass
class FoldResult:
    """One contiguous OOS fold for one ticker."""

    fold: int
    start: int
    end: int
    trades: list[Any] = field(default_factory=list)
    ev_per_trade: float = 0.0
    sharpe: float = 0.0
    pnl: list[float] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        """Count of completed trades inside this fold."""
        return len(self.trades)

    @property
    def is_empty(self) -> bool:
        """True when no trade completed inside the fold."""
        return self.total_trades == 0


def _bars_for_fold(data: dict[str, Any], start: int, end: int) -> BarSeries:
    """Slice bars for one fold, keeping EDGE_LOOKBACK warmup bars."""
    warm = max(0, start - EDGE_LOOKBACK)
    s = slice(warm, end)
    return BarSeries(
        np.asarray(data["close"])[s],
        np.asarray(data["high"])[s],
        np.asarray(data["low"])[s],
        np.asarray(data["volume"])[s],
    )


def fold_windows(total_bars: int, folds: int = DEFAULT_FOLDS) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping OOS windows covering the series tail.

    Returns [(start, end), ...]. The evaluation region is everything after
    the warmup; each fold holds floor(total/folds) bars of test data.
    """
    usable = total_bars - EDGE_LOOKBACK
    if usable <= 0:
        return []
    per_fold = max(1, int(usable // folds))
    windows: list[tuple[int, int]] = []
    for k in range(folds):
        start = EDGE_LOOKBACK + k * per_fold
        end = start + per_fold
        if start >= total_bars:
            break
        end = min(end, total_bars)
        windows.append((start, end))
    return windows


def _fold_sharpe(pnl: list[float]) -> float:
    """Per-trade return Sharpe inside one fold (0 when < 2 trades)."""
    if len(pnl) < 2:
        return 0.0
    r = np.asarray(pnl, dtype=float)
    sd = float(np.std(r))
    return float(np.mean(r) / (sd + 1e-12)) if sd > 0 else 0.0


def run_walk_forward(
    ticker: str,
    data: dict[str, Any],
    folds: int = DEFAULT_FOLDS,
) -> list[FoldResult]:
    """Evaluate ``ticker`` OOS across contiguous folds with static weights.

    Static weights (production backtest path — learning is off). Trades are
    counted only when they complete inside the fold; a trade may open at the
    fold boundary but must CLOSE before ``end`` to be scored.
    """
    close = np.asarray(data["close"], dtype=float)
    total = len(close)
    out: list[FoldResult] = []
    for k, (start, end) in enumerate(fold_windows(total, folds)):
        fold_bars = _bars_for_fold(data, start, end)
        trades, equity = simulate_ticker(ticker, fold_bars, EDGE_LOOKBACK)
        # Only score trades whose exit bar is strictly inside [start, end).
        # simulate_ticker's idx is relative to the folded slice; offset by warmup:
        # warm = start - EDGE_LOOKBACK  → absolute_exit = relative + warm.
        warm = start - EDGE_LOOKBACK
        scored = [
            t for t in trades if warm + t.exit_idx < end and warm + t.entry_idx >= start
        ]
        pnl = [t.pnl_pct for t in scored]
        out.append(
            FoldResult(
                fold=k,
                start=start,
                end=end,
                trades=scored,
                ev_per_trade=float(np.mean(pnl)) if pnl else 0.0,
                sharpe=_fold_sharpe(pnl),
                pnl=pnl,
            )
        )
    return out


def total_oos_trades(windows: list[FoldResult]) -> int:
    """Total completed OOS trades across all folds."""
    return sum(w.total_trades for w in windows)


def trial_pnl_by_ticker(
    results: dict[str, list[FoldResult]],
) -> dict[str, list[float]]:
    """Flatten each ticker's OOS trade pnl into one return series."""
    return {t: [p for w in folds for p in w.pnl] for t, folds in results.items()}


def _pooled_returns(results: dict[str, list[FoldResult]]) -> np.ndarray:
    """Pool every admissible ticker's OOS trade returns into one array."""
    vals: list[float] = []
    for ticker, folds in results.items():
        n = sum(w.total_trades for w in folds)
        if n >= MIN_TRADES:
            vals.extend(p for w in folds for p in w.pnl)
    return np.asarray(vals, dtype=float)


def pooled_sharpe(results: dict[str, list[FoldResult]]) -> float:
    """Sharpe of pooled admissible OOS trade returns (per-trade basis)."""
    r = _pooled_returns(results)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r))
    return float(np.mean(r) / (sd + 1e-12)) if sd > 0 else 0.0


def deflated_sharpe(results: dict[str, list[FoldResult]]) -> float:
    """Monte-Carlo deflated Sharpe.

    Builds the null distribution of the BEST Sharpe obtainable by chance
    over ``n_trials`` independent trials (tickers × folds with trades),
    then returns the percentile rank of the observed pooled Sharpe against
    that null. A rank below 0.05 = no evidence of edge after deflation.
    """
    r = _pooled_returns(results)
    if r.size < 2:
        return 0.0
    r = r - float(np.mean(r))  # demean → null has zero mean
    n_trials = _count_trials(results)
    if n_trials <= 0:
        return 0.0
    observed = pooled_sharpe(results)
    null_best_vals: list[float] = []
    n_obs = r.size
    for _ in range(BOOTSTRAP_ITERS):
        tries = []
        for _ in range(n_trials):
            sample = RNG.choice(r, size=n_obs, replace=True)
            sd = float(np.std(sample))
            tries.append(float(np.mean(sample) / (sd + 1e-12)) if sd > 0 else 0.0)
        null_best_vals.append(max(tries))
    null_best = np.asarray(null_best_vals, dtype=float)
    # Deflated edge = observed pooled Sharpe minus the 95th percentile of the
    # best-Sharpe-by-chance null over ``n_trials`` trials. <= 0 → no edge.
    return observed - float(np.quantile(null_best, 0.95))


def _count_trials(results: dict[str, list[FoldResult]]) -> int:
    """Number of independent trials (ticker×fold with ≥2 trades)."""
    n = 0
    for folds in results.values():
        n += sum(1 for w in folds if w.total_trades >= 2)
    return n


# ── CSCV: probability of backtest overfitting ───────────────────────────


def pbo(results: dict[str, list[FoldResult]]) -> float:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CVC.

    Builds an observations × trials matrix whose columns are each ticker's
    OOS per-trade returns (padded to the common length), then, for every
    train/test split of trials, counts when a strategy that looks BEST
    in-sample is BELOW median out-of-sample. PBO = that fraction.
    """
    series = {t: [p for w in folds for p in w.pnl] for t, folds in results.items()}
    cols = [np.asarray(v, dtype=float) for v in series.values() if len(v) >= 2]
    if len(cols) < 4:
        return 0.5  # too few tickers for a meaningful CSCV → unknown
    min_len = min(len(c) for c in cols)
    m = np.vstack([c[:min_len] for c in cols])  # (tickers × obs)
    trials_count = m.shape[0]
    # Every binary vector selects the training half of the trials.
    selections = _half_subsets(trials_count, limit=64)
    if not selections:
        return 0.5
    overfit = 0
    total = 0
    for idx in selections:
        mask = np.zeros(trials_count, dtype=bool)
        mask[list(idx)] = True
        is_perf = _rank_performance(m[mask])
        oos_perf = _rank_performance(m[~mask])
        # Which trial was best in-sample? Was it below median OOS?
        best_is = int(np.argmax(is_perf))
        if oos_perf[best_is] < np.median(oos_perf):
            overfit += 1
        total += 1
    return float(overfit / max(total, 1))


def _rank_performance(mat: np.ndarray) -> np.ndarray:
    """Rank each trial's mean OOS return within its split (ties→average)."""
    means = np.mean(mat, axis=1) if mat.ndim == 2 else np.asarray([[0.0]])
    order = np.asarray(stats.rankdata(-means))  # higher return → smaller rank
    return order


def _half_subsets(n: int, limit: int) -> list[tuple[int, ...]]:
    """All C(n, n//2) index subsets, capped at ``limit`` (deterministic)."""
    from itertools import combinations

    k = n // 2
    if k <= 0 or k >= n:
        return []
    out: list[tuple[int, ...]] = list(combinations(range(n), k))
    step = max(1, len(out) // limit)
    return out[::step][:limit]


# ── Verdicts ────────────────────────────────────────────────────────────


@dataclass
class TickerVerdict:
    """Per-ticker OOS statistical verdict."""

    ticker: str
    total_oos_trades: int
    ev_per_trade: float
    sharpe: float
    admissible: bool
    verdict: str  # PASS | FAIL | INSUFFICIENT


@dataclass
class UniverseVerdict:
    """Aggregate WFA verdict for the universe."""

    admissible_tickers: list[TickerVerdict]
    pooled_sharpe: float
    deflated_edge: float
    pbo: float
    verdict: str
    detail: str


def verdicts(results: dict[str, list[FoldResult]]) -> UniverseVerdict:
    """Build PASS/FAIL/INSUFFICIENT verdicts per ticker + universe."""
    per_ticker: list[TickerVerdict] = []
    for ticker, folds in results.items():
        n = total_oos_trades(folds)
        pnl = [p for w in folds for p in w.pnl]
        ev = float(np.mean(pnl)) if pnl else 0.0
        sharpe = _fold_sharpe(pnl)
        admissible = n >= MIN_TRADES
        if not admissible:
            verdict = "INSUFFICIENT"
        elif ev > 0 and sharpe > 0:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        per_ticker.append(TickerVerdict(ticker, n, ev, sharpe, admissible, verdict))

    adm = [v for v in per_ticker if v.admissible]
    pooled_sr = pooled_sharpe(results)
    defl = deflated_sharpe(results)
    overfit = pbo(results)

    if not adm:
        verdict = "INSUFFICIENT"
        detail = "No ticker reached the OOS min-trade floor — needs more data."
    elif defl <= 0:
        verdict = "FAIL"
        detail = f"Deflated OOS Sharpe {defl:.3f} <= 0 after {_count_trials(results)} trials."
    elif pooled_sr <= 0:
        verdict = "FAIL"
        detail = f"Pooled OOS Sharpe {pooled_sr:.3f} <= 0."
    else:
        verdict = "PASS"
        detail = (
            f"Deflated edge {defl:.3f} > 0 across {_count_trials(results)} trials; "
            f"PBO {overfit:.2f}."
        )
    return UniverseVerdict(adm, pooled_sr, defl, overfit, verdict, detail)


def run_wfa_universe(
    tickers: list[str],
    data_dir: str | Any,
    folds: int = DEFAULT_FOLDS,
) -> dict[str, list[FoldResult]]:
    """Run walk-forward across all tickers from a data directory."""
    from pathlib import Path

    data_dir = Path(data_dir)
    results: dict[str, list[FoldResult]] = {}
    for ticker in tickers:
        path = data_dir / f"{ticker}_1min.csv"
        if not path.exists():
            continue
        try:
            data = load_ohlcv(path)
        except ValueError:
            continue
        if len(data["close"]) < EDGE_LOOKBACK + MIN_TRADES:
            continue
        results[ticker] = run_walk_forward(ticker, data, folds)
    return results


def serialize(verdict: UniverseVerdict) -> dict[str, Any]:
    """Plain-JSON view of a UniverseVerdict."""
    return {
        "admissible_tickers": [
            {
                "ticker": v.ticker,
                "oos_trades": v.total_oos_trades,
                "ev_per_trade": round(v.ev_per_trade, 4),
                "sharpe": round(v.sharpe, 4),
                "verdict": v.verdict,
            }
            for v in verdict.admissible_tickers
        ],
        "pooled_sharpe": round(verdict.pooled_sharpe, 4),
        "deflated_edge": round(verdict.deflated_edge, 4),
        "pbo": round(verdict.pbo, 4),
        "verdict": verdict.verdict,
        "detail": verdict.detail,
    }


__all__ = [
    "FoldResult",
    "TickerVerdict",
    "UniverseVerdict",
    "fold_windows",
    "run_walk_forward",
    "run_wfa_universe",
    "verdicts",
    "deflated_sharpe",
    "pbo",
    "MIN_TRADES",
    "DEFAULT_FOLDS",
]
