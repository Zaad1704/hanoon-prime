"""hanoon_prime.validator — self-evaluating indicator edge assessment.

Uses permutation tests to determine whether each indicator's
correlation with next-bar returns is significant. Flow:
pool → permute (p<0.05) → weight by |corr|.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .cerebellum import INDICATOR_NAMES, compute_alpha
from .eyes import compute_buy_volume, estimate_bid_ask, load_ohlcv
from .immune import EDGE_LOOKBACK, EDGE_MIN_SAMPLES, EDGE_P_VALUE

__all__ = [
    "evaluate_indicator_pooled",
    "evaluate_indicator_edge",
    "pooled_signals",
    "calibrate_weights",
]


def _perm_pvalue(signals: np.ndarray, returns: np.ndarray, n_perm: int) -> float:
    """Permutation test: p-value for corr(signals, returns) != 0."""
    observed = abs(np.corrcoef(signals, returns)[0, 1])
    if np.isnan(observed):
        return 1.0
    rng = np.random.default_rng(42)
    count = sum(
        1 for _ in range(n_perm)
        if (lambda p: not np.isnan(p) and p >= observed)(
            abs(np.corrcoef(rng.permutation(signals), returns)[0, 1])
        )
    )
    return count / n_perm


def _compute_ticker_signals(
    data: dict[str, Any], window: int = EDGE_LOOKBACK
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute all 5 indicator signals + next-bar returns for one ticker."""
    close, volume = data["close"], data["volume"]
    high, low = data["high"], data["low"]
    bv = compute_buy_volume(close, high, low, volume)
    signals: dict[str, list[float]] = {name: [] for name in INDICATOR_NAMES}
    returns: list[float] = []
    for i in range(window, len(close) - 1):
        c_w, v_w = close[i - window : i + 1], volume[i - window : i + 1]
        bv_w = bv[i - window : i + 1]
        alpha = compute_alpha(c_w, v_w, bv_w, *estimate_bid_ask(v_w, bv_w))
        for name in INDICATOR_NAMES:
            signals[name].append(alpha[name])
        returns.append(float((close[i + 1] - close[i]) / close[i]))
    return {k: np.array(v) for k, v in signals.items()}, np.array(returns)


def pooled_signals(
    tickers: list[str], data_dir: Path, window: int = EDGE_LOOKBACK
) -> tuple[dict[str, list[float]], np.ndarray]:
    """Pool raw indicator signals + returns across tickers."""
    pooled: dict[str, list[float]] = {n: [] for n in INDICATOR_NAMES}
    pooled_returns: list[float] = []
    for ticker in tickers:
        path = data_dir / f"{ticker}_1min.csv"
        if not path.exists():
            continue
        try:
            data = load_ohlcv(path)
        except ValueError:
            continue
        if len(data["close"]) < window + 50:
            continue
        sigs, rets = _compute_ticker_signals(data, window)
        for name in INDICATOR_NAMES:
            pooled[name].extend(sigs[name].tolist())
        pooled_returns.extend(rets.tolist())
    return pooled, np.array(pooled_returns)


def _test_pooled_indicator(
    name: str, sig_array: np.ndarray, ret_array: np.ndarray, n_perm: int
) -> dict[str, Any]:
    """Run permutation test for one indicator on pooled data."""
    if len(sig_array) < EDGE_MIN_SAMPLES:
        return {"pvalue": 1.0, "corr": 0.0, "n_samples": len(sig_array), "significant": False}
    p = _perm_pvalue(sig_array, ret_array, n_perm)
    c = np.corrcoef(sig_array, ret_array)[0, 1]
    return {"pvalue": p, "corr": float(c) if not np.isnan(c) else 0.0,
            "n_samples": len(sig_array), "significant": p < EDGE_P_VALUE}


def evaluate_indicator_pooled(
    tickers: list[str], data_dir: Path, n_perm: int = 500
) -> dict[str, dict[str, Any]]:
    """Pool signals+returns across tickers, run permutation test per indicator."""
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    pooled, ret_array = pooled_signals(tickers, data_dir)
    return {
        name: _test_pooled_indicator(name, np.array(pooled[name]), ret_array, n_perm)
        for name in INDICATOR_NAMES
    }


def evaluate_indicator_edge(
    tickers: list[str], data_dir: Path, n_perm: int = 200
) -> dict[str, dict[str, Any]]:
    """Per-ticker permutation test, aggregated across tickers."""
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    pvals: dict[str, list[float]] = {n: [] for n in INDICATOR_NAMES}
    corrs: dict[str, list[float]] = {n: [] for n in INDICATOR_NAMES}
    sig_counts: dict[str, int] = {n: 0 for n in INDICATOR_NAMES}
    totals: dict[str, int] = {n: 0 for n in INDICATOR_NAMES}
    for ticker in tickers:
        _eval_ticker(ticker, data_dir, n_perm, pvals, corrs, sig_counts, totals)
    return _aggregate_edges(pvals, corrs, sig_counts, totals)


def _eval_ticker(
    ticker: str, data_dir: Path, n_perm: int,
    pvals: dict[str, list[float]], corrs: dict[str, list[float]],
    sig_counts: dict[str, int], total_counts: dict[str, int],
) -> None:
    """Evaluate all indicators for one ticker."""
    path = data_dir / f"{ticker}_1min.csv"
    if not path.exists():
        return
    try:
        data = load_ohlcv(path)
    except ValueError:
        return
    if len(data["close"]) < EDGE_LOOKBACK + 50:
        return
    sigs, rets = _compute_ticker_signals(data)
    for name in INDICATOR_NAMES:
        sig = sigs[name]
        if len(sig) < 50:
            continue
        p = _perm_pvalue(sig, rets, n_perm)
        c = np.corrcoef(sig, rets)[0, 1]
        pvals[name].append(p)
        corrs[name].append(abs(c) if not np.isnan(c) else 0.0)
        total_counts[name] += 1
        if p < EDGE_P_VALUE:
            sig_counts[name] += 1


def _aggregate_edges(
    pvals: dict[str, list[float]], corrs: dict[str, list[float]],
    sig_counts: dict[str, int], total_counts: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Build result dict from per-ticker accumulators."""
    results: dict[str, dict[str, Any]] = {}
    for name in INDICATOR_NAMES:
        avg_p = float(np.mean(pvals[name])) if pvals[name] else 1.0
        avg_c = float(np.mean(corrs[name])) if corrs[name] else 0.0
        results[name] = {
            "pvalue": avg_p, "corr": avg_c,
            "tickers_significant": sig_counts[name],
            "tickers_total": total_counts[name],
        }
    return results


def calibrate_weights(pooled_edge: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Weight indicators by relative |correlation| strength."""
    floor = 0.01
    raw = {
        n: info["corr"] if info["significant"] and abs(info["corr"]) > 0 else floor
        for n, info in pooled_edge.items()
    }
    total = sum(abs(v) for v in raw.values())
    if total <= 0:
        return {n: 0.20 for n in INDICATOR_NAMES}
    weights = {k: v / total for k, v in raw.items()}
    wsum = sum(abs(v) for v in weights.values())
    return {k: v / wsum for k, v in weights.items()} if wsum > 0 else weights
