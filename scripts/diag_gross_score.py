#!/usr/bin/env python3
"""Diagnostic: gross-vs-net EV decomposition + score-band monotonicity on LEAN.

Runs the SAME lean stack as phase7_bench (3-factor + gate) on the 180d Alpaca
panel, but instead of pooling pnl it keeps per-trade pnl, entry score, and
reconstructed GROSS pnl (net + fee drag) so we can answer:

  Q1: is the EV loss driven by costs (gross>0, net<0) or the signal (gross<0)?
  Q2: does EV / win-rate rise monotonically with entry score — i.e. is there a
      higher threshold that fixes entries?
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from hanoon_prime.backtest import _discover_tickers
from hanoon_prime.eyes import load_ohlcv
from hanoon_prime.hands import FEE_RATE, FIXED_FEE
from hanoon_prime.phase7 import LeanCfg, load_spy_closes, run_walk_forward_lean

DATA = Path("data/research/alpaca_180d")


def _gross_pct(trade) -> float:
    """Reverse hands._compute_pnl(frac=1.0): add back the fee drag."""
    notional = trade.entry_price * trade.shares
    if notional <= 0:
        return trade.pnl_pct
    drag = 2.0 * (FIXED_FEE + FEE_RATE * notional) / notional
    return trade.pnl_pct + drag


def pool(pnls) -> dict:
    r = np.asarray(pnls, dtype=float)
    if r.size == 0:
        return {"ev": 0.0, "wr": 0.0, "trades": 0, "rr": 0.0}
    wins = r[r > 0]
    losses = r[r <= 0]
    aw = float(np.mean(wins)) if wins.size else 0.0
    al = abs(float(np.mean(losses))) if losses.size else 0.0
    rr = aw / al if al > 0 else 0.0
    wr = float(np.mean(r > 0))
    return {"ev": wr * rr - (1 - wr), "wr": wr, "trades": int(r.size), "rr": rr}


def main() -> None:
    tickers = [t for t in _discover_tickers(DATA) if t != "SPY"]
    spy = load_spy_closes(str(DATA / "SPY_1min.csv"))

    all_trades: list = []
    for ticker in tickers:
        path = DATA / f"{ticker}_1min.csv"
        if not path.exists():
            continue
        try:
            data = load_ohlcv(path)
        except ValueError:
            continue
        folds = run_walk_forward_lean(ticker, data, LeanCfg(spy_closes=spy))
        for f in folds:
            all_trades.extend(f.trades)

    print(f"tickers={len(tickers)} total_trades={len(all_trades)}")

    scores = np.array([t.score for t in all_trades])
    net = np.array([t.pnl_pct for t in all_trades])
    gross = np.array([_gross_pct(t) for t in all_trades])

    # ── Q1: gross vs net ──────────────────────────────────────────────────
    print("\n== Q1: gross vs net (all entries, pooled) ==")
    print(f"  NET  : {pool(net)}")
    print(f"  GROSS: {pool(gross)}")
    drag = float(np.mean(net - gross))
    print(f"  mean fee drag/trade: {drag:+.6f} R-percent")

    # ── Q2: score-band monotonicity ───────────────────────────────────────
    print("\n== Q2: score-band monotonicity (net + gross, OOS trades) ==")
    bands = [
        (-np.inf, 0.0),
        (0.0, 0.5),
        (0.5, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 900.0),
    ]
    print(f"  {'band':>14s} {'trades':>6s} {'EV_net':>7s} {'EV_gr':>7s} {'WR':>6s}")
    for lo, hi in bands:
        m = (scores >= lo) & (scores < hi)
        if not m.any():
            print(f"  [{lo:>5.1f},{hi:>5.1f}) {'0':>6s}  --  --  --")
            continue
        pn = pool(net[m])
        pg = pool(gross[m])
        print(
            f"  [{lo:>5.1f},{hi:>5.1f}) {pn['trades']:6d} "
            f"{pn['ev']:+.3f}R {pg['ev']:+.3f}R {pn['wr']:.1%}"
        )

    print(
        "\n  score percentiles:", np.percentile(scores, [0, 25, 50, 75, 100]).round(3)
    )


if __name__ == "__main__":
    main()
