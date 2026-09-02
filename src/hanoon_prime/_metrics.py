"""hanoon_prime._metrics — performance metrics from trades + equity curve."""
from __future__ import annotations

import numpy as np
from dataclasses import is_dataclass

# SimTrade is a dataclass; import for type checking only
from ._sim import SimTrade


def _equity_stats(equity_curve: list[float]) -> tuple[float, float]:
    """Return (max_drawdown, sharpe_ratio) from equity curve."""
    ec = np.array(equity_curve)
    if len(ec) <= 1:
        return 0.0, 0.0
    running_max = np.maximum.accumulate(ec)
    dd = (ec - running_max) / np.maximum(running_max, 1e-12)
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
    returns = np.diff(ec)
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-12))
    return max_dd, sharpe


def calculate_metrics(ticker: str, trades: list, equity_curve: list[float]) -> dict:
    """Compute performance metrics from trade list and equity curve."""
    total_trades = len(trades)
    if total_trades == 0:
        return _empty_metrics(ticker)

    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    win_rate = len(wins) / total_trades
    avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    avg_loss = abs(float(np.mean([t.pnl_pct for t in losses]))) if losses else 0.0

    # EV per trade in R-units (R = 0.03 = risk percent = 3% stop)
    avg_loss_for_ev = avg_loss if avg_loss > 0 else 0.03
    ev_per_trade = win_rate * (avg_win / 0.03) - (1 - win_rate) * (avg_loss_for_ev / 0.03)
    total_return = sum(t.pnl_pct for t in trades)
    realized_rr = avg_win / avg_loss if avg_loss > 0 else 0.0

    max_dd, sharpe = _equity_stats(equity_curve)

    return _build_metrics_dict(
        ticker, total_trades, len(wins), len(losses), win_rate,
        ev_per_trade, total_return, avg_win, avg_loss, realized_rr, max_dd, sharpe,
    )


def _build_metrics_dict(ticker, n, wins, losses, wr, ev, ret, aw, al, rr, dd, sr):
    """Assemble the metrics dict (helper to keep calculate_metrics under 40 lines)."""
    return {
        "ticker": ticker, "total_trades": n, "wins": wins, "losses": losses,
        "win_rate": round(wr, 4), "expectancy": round(ev, 4),
        "ev_per_trade": round(ev, 4), "total_return_pct": round(ret * 100, 2),
        "avg_win_pct": round(aw * 100, 4), "avg_loss_pct": round(al * 100, 4),
        "realized_rr": round(rr, 4), "max_drawdown": round(dd, 4),
        "sharpe_ratio": round(sr, 4),
        "status": "PROFITABLE" if ev > 0 else "UNPROFITABLE",
    }


def _empty_metrics(ticker: str) -> dict:
    return {
        "ticker": ticker, "total_trades": 0, "win_rate": 0.0,
        "expectancy": 0.0, "ev_per_trade": 0.0, "total_return_pct": 0.0,
        "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "realized_rr": 0.0,
        "max_drawdown": 0.0, "sharpe_ratio": 0.0, "status": "INSUFFICIENT_DATA",
    }
