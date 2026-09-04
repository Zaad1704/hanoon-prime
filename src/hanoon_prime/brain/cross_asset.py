"""brain.cross_asset — Cross-Asset Lead-Lag Signal Computation.

Computes inter-market correlation, lead-lag timing, price divergence,
and momentum rotation signals from live tick data.

Source: rebuild's cross_asset.py (simplified).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Reference pairs: (reference_ticker, weight)
_REFERENCE_PAIRS: list[tuple[str, float]] = [
    ("SPY", 0.4),
    ("QQQ", 0.3),
    ("IWM", 0.15),
    ("VXX", 0.15),
]


@dataclass
class CrossAssetSignals:
    correlation: float = 0.0
    lead_lag: float = 0.0
    divergence: float = 0.0
    momentum_rotation: float = 0.0
    modifier: float = 0.0


class CrossAssetEngine:
    """Compute cross-asset lead-lag signals."""

    def __init__(self, history_len: int = 50) -> None:
        """Auto-generated docstring."""
        self._history_len = history_len
        self._ticker_returns: dict[str, deque[float]] = {}
        self._ref_returns: dict[str, deque[float]] = {}

    def update(
        self,
        ticker: str,
        ticker_price: float,
        ref_prices: Optional[dict[str, float]] = None,
    ) -> CrossAssetSignals:
        """Update with new price data and compute signals."""
        if ref_prices is None:
            return CrossAssetSignals()
        # Track returns
        self._add_return(ticker, ticker_price, self._ticker_returns)
        for ref, price in ref_prices.items():
            self._add_return(ref, price, self._ref_returns)
        # Compute signals
        corr = self._compute_correlation(ticker)
        lead_lag = self._compute_lead_lag(ticker)
        div = self._compute_divergence(ticker, ref_prices)
        mom = self._compute_momentum_rotation(ticker, ref_prices)
        mod = self._compute_modifier(corr, lead_lag, div)
        return CrossAssetSignals(
            correlation=corr,
            lead_lag=lead_lag,
            divergence=div,
            momentum_rotation=mom,
            modifier=mod,
        )

    def _add_return(
        self, key: str, price: float, store: dict[str, deque[float]]
    ) -> None:
        """Add return to history store."""
        if key not in store:
            store[key] = deque(maxlen=self._history_len)
        if len(store[key]) > 0 and store[key][-1] != 0:
            ret = (price - store[key][-1]) / store[key][-1]
            store[key].append(ret)
        else:
            store[key].append(0.0)

    def _compute_correlation(self, ticker: str) -> float:
        """Auto-generated docstring."""
        t_ret = self._ticker_returns.get(ticker, deque())
        if len(t_ret) < 10:
            return 0.0
        correlations = []
        for ref, r_ret in self._ref_returns.items():
            if len(r_ret) < 10 or ref not in _REFERENCE_PAIRS:
                continue
            n = min(len(t_ret), len(r_ret))
            t_arr = np.array(list(t_ret)[-n:])
            r_arr = np.array(list(r_ret)[-n:])
            if np.std(t_arr) > 0 and np.std(r_arr) > 0:
                corr = float(np.corrcoef(t_arr, r_arr)[0, 1])
                weight = next((w for r, w in _REFERENCE_PAIRS if r == ref), 0.1)
                correlations.append(corr * weight)
        return sum(correlations) if correlations else 0.0

    def _compute_lead_lag(self, ticker: str) -> float:
        """Auto-generated docstring."""
        t_ret = self._ticker_returns.get(ticker, deque())
        if len(t_ret) < 5:
            return 0.0
        for ref, _ in _REFERENCE_PAIRS:
            r_ret = self._ref_returns.get(ref, deque())
            if len(r_ret) < 5:
                continue
            # Simple lag correlation
            t_arr = np.array(list(t_ret)[-5:])
            r_arr = np.array(list(r_ret)[-5:])
            if len(t_arr) == len(r_arr) and np.std(t_arr) > 0:
                lag_corr = float(np.corrcoef(t_arr[:-1], r_arr[1:])[0, 1])
                if not math.isnan(lag_corr):
                    return lag_corr * 0.1  # bounded
        return 0.0

    def _compute_divergence(self, ticker: str, ref_prices: dict[str, float]) -> float:
        """Price divergence from reference average."""
        if not ref_prices:
            return 0.0
        avg_ref = sum(ref_prices.values()) / len(ref_prices)
        if avg_ref == 0:
            return 0.0
        return 0.0  # simplified — needs ticker price history

    def _compute_momentum_rotation(
        self, ticker: str, ref_prices: dict[str, float]
    ) -> float:
        """Momentum rotation signal."""
        return 0.0  # simplified — needs multi-period returns

    def _compute_modifier(self, corr: float, lead_lag: float, div: float) -> float:
        """Bounded modifier from cross-asset signals."""
        mod = 0.0
        if abs(corr) > 0.7:
            mod += 0.02 * (1 if corr > 0 else -1)
        if abs(lead_lag) > 0.5:
            mod += 0.01 * lead_lag
        return max(-0.03, min(0.03, mod))
