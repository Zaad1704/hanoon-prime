"""hanoon_prime.fracdiff — fractional differentiation for stationarity.

López de Prado (Advances in Financial ML, ch. 5): integer differencing (I(1))
turns a random walk stationary but destroys every bit of memory. Fractionally
differencing by the SMALLEST ``d`` whose result passes an Augmented
Dickey-Fuller test keeps the series stationary while preserving the most
long-memory inertia — exactly the trade the 5-indicator cortex wants.

Weights are the binomial expansion of ``(1 - B) ** d``::

    w_0 = 1,  w_k = -w_{k-1} * (d - k + 1) / k

The window is FIXED (truncated once ``|w_k| < tau``), never expanding, so the
transform is causal and carries no future information. This module is a pure
feature transform: it never produces a verdict (R1) and the read site stays
OFF via ``FRACDIFF_ENABLED`` until a money gate clears it.
"""

from __future__ import annotations

import numpy as np

from .immune import FRACDIFF_ADF_PVALUE, FRACDIFF_D_STEP, FRACDIFF_MAX_LEN, FRACDIFF_TAU

# MacKinnon (1994) asymptotic critical values for the ADF t-statistic
# (constant, no trend). Finite-sample response-surface corrections are NOT
# applied: only the 5% decision boundary is consumed, and it is exact at the
# asymptotic limit. Upper-tail p-values are a documented approximation.
_ADF_P: tuple[float, ...] = (0.001, 0.005, 0.010, 0.025, 0.050, 0.100)
_ADF_TAU: tuple[float, ...] = (-4.30, -3.75, -3.43035, -3.11791, -2.86154, -2.56677)
_ADF_TAU_MEDIAN: float = -1.57513  # tau at p = 0.50
_ADF_P_MAX: float = 0.90  # clamp for the extrapolated upper tail


def fracdiff_weights(
    d: float, tau: float = FRACDIFF_TAU, max_len: int = FRACDIFF_MAX_LEN
) -> np.ndarray:
    """Fixed-window FracDiff weights for order ``d``.

    Expands ``(1 - B) ** d`` and truncates at the first ``|w_k| < tau`` or
    ``max_len`` terms, whichever comes first. ``d = 0`` returns ``[1.0]``
    (identity); ``d = 1`` returns ``[1.0, -1.0]`` (first difference).
    """
    weights = [1.0]
    for k in range(1, max_len + 1):
        wk = -weights[-1] * (d - k + 1) / k
        if abs(wk) < tau:
            break
        weights.append(wk)
    return np.asarray(weights, dtype=float)


def fracdiff(
    series: np.ndarray,
    d: float,
    tau: float = FRACDIFF_TAU,
    max_len: int = FRACDIFF_MAX_LEN,
) -> np.ndarray:
    """Causally fractionally differentiate *series* by order ``d``.

    Returns the transformed series aligned to ``series[len(w) - 1:]``: the
    warm-up prefix, which sees only partial history, is dropped. An input
    shorter than the weight window returns an empty array.
    """
    x = np.asarray(series, dtype=float).ravel()
    weights = fracdiff_weights(d, tau=tau, max_len=max_len)
    if x.size < weights.size:
        return np.empty(0, dtype=float)
    return np.asarray(np.convolve(x, weights, mode="valid"), dtype=float)


def _adf_max_lag(n: int) -> int:
    """Schwert (1989) rule for the maximum ADF lag, floored at 1."""
    return max(1, int(np.floor(12.0 * (n / 100.0) ** 0.25)))


def _adf_tstat(y: np.ndarray, p: int) -> float:
    """OLS t-statistic on the unit-root coefficient of an ADF regression."""
    dy = np.diff(y)
    n = dy.size - p
    if n < 5:
        return 0.0
    cols = [np.ones(n), y[p:-1]]
    for i in range(1, p + 1):
        cols.append(dy[p - i : dy.size - i])
    design = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(design, dy[p:], rcond=None)
    resid = dy[p:] - design @ beta
    dof = n - design.shape[1]
    if dof <= 0:
        return 0.0
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = float(np.sqrt(max(sigma2 * xtx_inv[1, 1], 1e-18)))
    return float(beta[1]) / se


def adf_pvalue(tstat: float) -> float:
    """Approximate ADF p-value from the MacKinnon asymptotic surface.

    Interpolates the lower tail through the 0.1%..10% critical values and
    extrapolates above the 10% point via the median. Only the 5% decision
    boundary (``tstat < -2.86154``) is treated as exact.
    """
    if tstat <= _ADF_TAU[0]:
        return _ADF_P[0]
    if tstat < _ADF_TAU[-1]:
        return float(np.interp(tstat, _ADF_TAU, _ADF_P))
    slope = (_ADF_P_MAX - _ADF_P[-1]) / (_ADF_TAU_MEDIAN - _ADF_TAU[-1])
    return float(min(_ADF_P_MAX, _ADF_P[-1] + (tstat - _ADF_TAU[-1]) * slope))


def is_stationary(
    series: np.ndarray,
    max_lag: int | None = None,
    alpha: float = FRACDIFF_ADF_PVALUE,
) -> bool:
    """True when the ADF test rejects the unit root at level ``alpha``."""
    y = np.asarray(series, dtype=float).ravel()
    if y.size < 10:
        return False
    lag = _adf_max_lag(y.size) if max_lag is None else max_lag
    return adf_pvalue(_adf_tstat(y, lag)) < alpha


def memory_retention(original: np.ndarray, transformed: np.ndarray) -> float:
    """Correlation of levels vs FracDiff output (higher = more memory kept)."""
    a = np.asarray(original, dtype=float).ravel()
    b = np.asarray(transformed, dtype=float).ravel()
    if b.size == 0 or a.size < b.size:
        return 0.0
    tail = a[a.size - b.size :]
    denom = float(np.std(tail) * np.std(b))
    if denom <= 1e-12:
        return 0.0
    cov = float(np.mean((tail - tail.mean()) * (b - b.mean())))
    return cov / denom


def find_min_d(
    series: np.ndarray,
    step: float = FRACDIFF_D_STEP,
    tau: float = FRACDIFF_TAU,
    alpha: float = FRACDIFF_ADF_PVALUE,
) -> tuple[float, np.ndarray]:
    """Smallest ``d`` in (0, 1] whose FracDiff passes ADF at ``alpha``.

    Scans ``d`` ascending and returns ``(d, transformed)`` at the first
    stationary order — the minimum order preserves the most memory. If no
    grid point is stationary, returns ``(1.0, first difference)``.
    """
    x = np.asarray(series, dtype=float).ravel()
    for d in np.arange(step, 1.0 + 1e-9, step):
        transformed = fracdiff(x, float(d), tau=tau)
        if is_stationary(transformed, alpha=alpha):
            return float(d), transformed
    return 1.0, fracdiff(x, 1.0, tau=tau)
