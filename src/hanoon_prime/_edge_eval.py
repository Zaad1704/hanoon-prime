"""hanoon_prime._edge_eval — self-evaluating indicator edge assessment.

Uses permutation tests to determine whether each indicator's correlation
with next-bar returns is statistically significant — no hardcoded thresholds.

The system pools signals across all tickers, then runs a permutation test
per indicator on the pooled data. This gives maximum statistical power.
"""
from __future__ import annotations

import numpy as np

from .data import load_ohlcv, compute_buy_volume
from .alpha import (
    compute_vpin, compute_orderbook_imbalance, compute_institutional_flow,
    compute_momentum, compute_vwap_deviation,
)
from .constants import EDGE_LOOKBACK, EDGE_P_VALUE
from .alpha import INDICATOR_NAMES

__all__ = ["evaluate_indicator_edge", "evaluate_indicator_pooled"]


def _perm_pvalue(signals: np.ndarray, returns: np.ndarray,
                 n_perm: int) -> float:
    """Permutation test: p-value for corr(signals, returns) != 0."""
    observed = abs(np.corrcoef(signals, returns)[0, 1])
    if np.isnan(observed):
        return 1.0
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(signals)
        perm_corr = abs(np.corrcoef(shuffled, returns)[0, 1])
        if not np.isnan(perm_corr) and perm_corr >= observed:
            count += 1
    return count / n_perm


def _compute_ticker_signals(data: dict, window: int = EDGE_LOOKBACK):
    """Compute all 5 indicator signals + next-bar returns for one ticker."""
    close = data["close"]
    volume = data["volume"]
    high = data["high"]
    low = data["low"]
    bv = compute_buy_volume(close, high, low, volume)

    signals: dict[str, list] = {name: [] for name in INDICATOR_NAMES}
    returns: list = []

    for i in range(window, len(close) - 1):
        c_w = close[i - window : i + 1]
        v_w = volume[i - window : i + 1]
        bv_w = bv[i - window : i + 1]
        sv_w = np.maximum(v_w - bv_w, 0)

        signals["vpin"].append(compute_vpin(v_w, bv_w))
        signals["momentum"].append(compute_momentum(c_w))
        signals["vwap_deviation"].append(compute_vwap_deviation(c_w, v_w))

        tb, ts = float(np.sum(bv_w)), float(np.sum(sv_w))
        denom = tb + ts
        signals["orderbook_imbalance"].append(
            (tb - ts) / denom if denom > 1e-12 else 0.0
        )
        mean_v = float(np.mean(v_w))
        signals["institutional_flow"].append(
            compute_institutional_flow(v_w, mean_v if mean_v > 0 else 1.0)
        )
        returns.append(float((close[i + 1] - close[i]) / close[i]))

    return {k: np.array(v) for k, v in signals.items()}, np.array(returns)


def _pool_all_tickers(tickers, data_dir):
    """Load and pool all ticker signals and returns."""
    pooled = {name: [] for name in INDICATOR_NAMES}
    pooled_returns = []
    for ticker in tickers:
        path = data_dir / f"{ticker}_1min.csv"
        if not path.exists():
            continue
        data = load_ohlcv(path)
        if len(data["close"]) < EDGE_LOOKBACK + 50:
            continue
        sigs, rets = _compute_ticker_signals(data)
        for name in INDICATOR_NAMES:
            pooled[name].extend(sigs[name].tolist())
        pooled_returns.extend(rets.tolist())
    return pooled, np.array(pooled_returns)


def _test_pooled_indicator(name, sig_array, ret_array, n_perm):
    """Run permutation test for one indicator on pooled data."""
    if len(sig_array) < 200:
        return {"pvalue": 1.0, "corr": 0.0, "n_samples": len(sig_array),
                "significant": False}
    p = _perm_pvalue(sig_array, ret_array, n_perm)
    c = np.corrcoef(sig_array, ret_array)[0, 1]
    return {"pvalue": p,
            "corr": float(c) if not np.isnan(c) else 0.0,
            "n_samples": len(sig_array),
            "significant": p < EDGE_P_VALUE}


def evaluate_indicator_pooled(tickers, data_dir, n_perm=500):
    """Pool signals+returns across tickers, run one permutation test per indicator.

    This is the primary edge test. Pooling gives the statistical power
    that individual-ticker tests lack on short 5-min series.
    """
    from pathlib import Path
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    pooled, ret_array = _pool_all_tickers(tickers, data_dir)
    results = {}
    for name in INDICATOR_NAMES:
        sig_array = np.array(pooled[name])
        results[name] = _test_pooled_indicator(name, sig_array, ret_array, n_perm)
    return results


def _eval_ticker(ticker, data_dir, n_perm, pvals, corrs, sig_counts, total_counts):
    """Evaluate all indicators for one ticker. Mutates accumulators."""
    path = data_dir / f"{ticker}_1min.csv"
    if not path.exists():
        return
    data = load_ohlcv(path)
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


def evaluate_indicator_edge(tickers, data_dir, n_perm=200):
    """Auto-evaluate each indicator per-ticker via permutation test.

    Returns per-indicator aggregates (avg p-value, significant ticker count).
    """
    from pathlib import Path
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    pvals = {n: [] for n in INDICATOR_NAMES}
    corrs = {n: [] for n in INDICATOR_NAMES}
    sig_counts = {n: 0 for n in INDICATOR_NAMES}
    total_counts = {n: 0 for n in INDICATOR_NAMES}

    for ticker in tickers:
        _eval_ticker(ticker, data_dir, n_perm, pvals, corrs,
                     sig_counts, total_counts)

    return _aggregate_edge_results(pvals, corrs, sig_counts, total_counts)


def _aggregate_edge_results(pvals, corrs, sig_counts, total_counts):
    """Build result dict from per-ticker accumulators."""
    results = {}
    for name in INDICATOR_NAMES:
        avg_p = float(np.mean(pvals[name])) if pvals[name] else 1.0
        avg_c = float(np.mean(corrs[name])) if corrs[name] else 0.0
        results[name] = {"pvalue": avg_p, "corr": avg_c,
                         "tickers_significant": sig_counts[name],
                         "tickers_total": total_counts[name]}
    return results
