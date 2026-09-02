"""hanoon_prime._grid_search — find optimal stop/target/threshold.

Runs a fast bar-by-bar simulation with different parameter configurations
and picks the one with best aggregate EV. The system auto-tunes itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    EDGE_LOOKBACK, MAX_POSITION_NOTIONAL, MAX_LOSS_PER_TRADE,
)
from .alpha import compute_alpha
from .scoring import compute_score
from .edge import score_to_win_prob, compute_ev, kelly_fraction
from .thinker import deliberate
from ._sim import SimPosition, _check_exit, _close_position
from .data import load_ohlcv, compute_buy_volume, estimate_bid_ask

__all__ = ["grid_search", "sim_with_params", "SimConfig", "GRID_CONFIGS"]


@dataclass
class SimConfig:
    stop_pct: float
    target_pct: float
    threshold: float
    direction_mode: str = "mean_rev"  # "trend" or "mean_rev"


GRID_CONFIGS = [
    SimConfig(0.03, 0.12, 0.60, "mean_rev"),
    SimConfig(0.03, 0.12, 0.58, "mean_rev"),
    SimConfig(0.02, 0.08, 0.60, "mean_rev"),
    SimConfig(0.03, 0.10, 0.62, "mean_rev"),
    SimConfig(0.03, 0.12, 0.60, "trend"),
    SimConfig(0.03, 0.12, 0.58, "trend"),
    SimConfig(0.04, 0.16, 0.60, "trend"),
]


def _build_position(idx, price, direction, stop_pct, target_pct,
                    score, shares):
    """Create a SimPosition with proper stop/target."""
    if direction > 0:
        sl, tp = price * (1 - stop_pct), price * (1 + target_pct)
    else:
        sl, tp = price * (1 + stop_pct), price * (1 - target_pct)
    return SimPosition(
        ticker="T", entry_idx=idx, entry_price=price, shares=shares,
        direction=direction, stop_price=sl, target_price=tp,
        peak_price=price, score=score,
    )


def _compute_direction(alpha, mode="mean_rev"):
    """Compute trade direction based on auto-calibrated edge.

    mean_rev: invert momentum to capture mean-reverting behavior.
    trend: raw positive correlation.
    """
    inst = alpha.get("institutional_flow", 1.0) - 1.0
    mom = alpha.get("momentum", 0.0)
    vpin_u = alpha.get("vpin_magnitude", 0.5)  # unsigned [0,1]
    vpin_c = vpin_u - 0.5  # centered at 0
    vwap = alpha.get("vwap_deviation", 0.0)

    if mode == "trend":
        net = inst * 0.4 + mom * 0.3 + vpin_c * 0.2 + vwap * 0.1
    else:
        # mean_rev: invert momentum (captures reversion on noisy data)
        net = inst * 0.3 + (-mom) * 0.3 + vpin_c * 0.2 + vwap * 0.2
    if net > 0.03:
        return 1
    if net < -0.03:
        return -1
    return 0


def _compute_signal_at(close, high, low, volume, i, weights, stop_pct, threshold, mode):
    """Compute alpha + score at bar i, return (thought, score, shares) or None."""
    start = max(0, i - EDGE_LOOKBACK)
    c = close[start : i + 1].tolist()
    v = volume[start : i + 1].tolist()
    bv = compute_buy_volume(close[start : i + 1], high[start : i + 1],
                            low[start : i + 1], volume[start : i + 1])

    bv_w = compute_buy_volume(c, high[start : i + 1], low[start : i + 1], v)
    bv = np.asarray(bv_w)
    bid_arr, ask_arr = estimate_bid_ask(v, bv)
    alpha = compute_alpha(c, v, bv.tolist(), bid_arr, ask_arr)
    score = compute_score(alpha, weights)
    wp = score_to_win_prob(score)
    ev_info = compute_ev(wp)
    kc = kelly_fraction(wp)
    direction = _compute_direction(alpha, mode)

    if direction == 0 or score < threshold or ev_info["gross_ev"] <= 0:
        return None

    risk = float(close[i + 1]) * stop_pct
    if risk > 0:
        shares = min(MAX_POSITION_NOTIONAL / close[i + 1], MAX_LOSS_PER_TRADE / risk)
    else:
        shares = 0.01
    if shares < 0.01:
        return None
    return direction, score, shares


def sim_with_params(close, high, low, volume, stop_pct, target_pct,
                    threshold, weights, window=EDGE_LOOKBACK, mode="mean_rev"):
    """Fast bar-by-block sim with custom params. Returns realized trades."""
    trades = []
    equity = [0.0]
    pos = None

    for i in range(window, len(close) - 1):
        if i + 1 >= len(close):
            break

        if pos is None:
            result = _compute_signal_at(
                close, high, low, volume, i, weights, stop_pct, threshold, mode)
            if result:
                direction, score, shares = result
                pos = _build_position(i + 1, float(close[i + 1]),
                                      direction, stop_pct, target_pct, score, shares)

        if pos is not None:
            exit_r = _check_exit(pos, float(low[i + 1]), float(high[i + 1]), i + 1)
            if exit_r:
                trades.append(_close_position(
                    pos, exit_r[0], i + 1, exit_r[1], None, equity))
                pos = None

    return trades


def grid_search(tickers, data_dir, weights, n_tickers=50):
    """Find best SimConfig by aggregate EV across tickers."""
    from ._metrics import calculate_metrics

    best_ev = -999.0
    best_idx = 0
    for ci, cfg in enumerate(GRID_CONFIGS):
        total_ev, n = 0.0, 0
        for ticker in tickers[:n_tickers]:
            path = data_dir / f"{ticker}_1min.csv"
            if not path.exists():
                continue
            data = load_ohlcv(path)
            c, h, l, v = data["close"], data["high"], data["low"], data["volume"]
            if len(c) < EDGE_LOOKBACK + 50:
                continue
            trades = sim_with_params(c, h, l, v, cfg.stop_pct,
                                     cfg.target_pct, cfg.threshold, weights,
                                     mode=cfg.direction_mode)
            if trades:
                m = calculate_metrics(ticker, trades, [0.0])
                total_ev += m["ev_per_trade"]
                n += 1
        avg_ev = total_ev / n if n > 0 else -999.0
        print(f"    Config {ci}: stop={cfg.stop_pct:.0%} "
              f"target={cfg.target_pct:.0%} thr={cfg.threshold:.2f} "
              f"mode={cfg.direction_mode} → avg_EV={avg_ev:.3f}R ({n} tickers)")
        if avg_ev > best_ev:
            best_ev = avg_ev
            best_idx = ci

    return GRID_CONFIGS[best_idx], {"avg_ev": best_ev}
