"""hanoon_prime.metrics — performance calculation from trades + equity curve.

Computes win rate, R:R, expectancy, drawdown, and Sharpe from the
list of completed Trade objects produced by hands.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _equity_stats(
    equity_curve: list[float],
) -> tuple[float, float]:
    """Return (max_drawdown, sharpe_ratio) from equity curve."""
    ec = np.array(equity_curve, dtype=float)
    if len(ec) <= 1:
        return 0.0, 0.0
    running_max = np.maximum.accumulate(ec)
    dd = (ec - running_max) / np.maximum(running_max, 1e-12)  # type: ignore[operator]
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
    returns = np.diff(ec)
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-12))
    return max_dd, sharpe


def calculate_metrics(
    ticker: str,
    trades: list[Any],
    equity_curve: list[float],
) -> dict[str, Any]:
    """Compute performance metrics from trade list and equity curve."""
    total_trades = len(trades)
    if total_trades == 0:
        return _empty_metrics(ticker)

    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    wr = len(wins) / total_trades
    aw = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    al = abs(float(np.mean([t.pnl_pct for t in losses]))) if losses else 0.0
    realized_rr = aw / al if al > 0 else 0.0
    ev = wr * realized_rr - (1 - wr) if al > 0 else wr * 1.0
    total_return = sum(t.pnl_pct for t in trades)
    max_dd, sr = _equity_stats(equity_curve)
    return _build_metrics(
        ticker,
        total_trades,
        len(wins),
        len(losses),
        wr,
        ev,
        total_return,
        aw,
        al,
        realized_rr,
        max_dd,
        sr,
    )


def _build_metrics(
    ticker: str,
    n: int,
    wins: int,
    losses: int,
    wr: float,
    ev: float,
    ret: float,
    aw: float,
    al: float,
    rr: float,
    dd: float,
    sr: float,
) -> dict[str, Any]:
    """Assemble the metrics dict (helper to stay under 40 lines)."""
    return {
        "ticker": ticker,
        "total_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 4),
        "expectancy": round(ev, 4),
        "ev_per_trade": round(ev, 4),
        "total_return_pct": round(ret * 100, 2),
        "avg_win_pct": round(aw * 100, 4),
        "avg_loss_pct": round(al * 100, 4),
        "realized_rr": round(rr, 4),
        "max_drawdown": round(dd, 4),
        "sharpe_ratio": round(sr, 4),
        "status": "PROFITABLE" if ev > 0 else "UNPROFITABLE",
    }


def _empty_metrics(ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "total_trades": 0,
        "win_rate": 0.0,
        "expectancy": 0.0,
        "ev_per_trade": 0.0,
        "total_return_pct": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "realized_rr": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "status": "INSUFFICIENT_DATA",
    }
