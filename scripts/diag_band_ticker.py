#!/usr/bin/env python3
"""Ticker-level decomposition of the >=0.90 conviction band (net EV).

Answers: is the +0.004R pocket structurally real (spread across tickers,
>-, per-ticker admissible) or a mega-cap concentration artifact?
Also reports what the universe looks like when you TRADE ONLY the >=0.90
slice (per-ticker pooled EV) — and what verdict WFA would give it.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from hanoon_prime.backtest import _discover_tickers
from hanoon_prime.eyes import load_ohlcv
from hanoon_prime.phase7 import LeanCfg, load_spy_closes, run_walk_forward_lean
from hanoon_prime.wfa import MIN_TRADES

DATA = Path("data/research/alpaca_180d")
BAND_LO = 0.90


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

    band: dict[str, list[float]] = {}
    for ticker in tickers:
        path = DATA / f"{ticker}_1min.csv"
        if not path.exists():
            continue
        try:
            data = load_ohlcv(path)
        except ValueError:
            continue
        for f in run_walk_forward_lean(ticker, data, LeanCfg(spy_closes=spy)):
            for t in f.trades:
                if t.score >= BAND_LO:
                    band.setdefault(ticker, []).append(t.pnl_pct)

    # per-ticker table
    print(f"\n== >=0.90 band by ticker (MIN_TRADES={MIN_TRADES}) ==")
    print(f"{'ticker':>6s} {'trades':>6s} {'EV':>7s} {'WR':>6s}   admissible")
    rows = []
    for tk, pnls in sorted(
        band.items(), key=lambda kv: pool(kv[1])["ev"], reverse=True
    ):
        p = pool(pnls)
        rows.append((tk, p))
        print(
            f"{tk:>6s} {p['trades']:6d} {p['ev']:+.3f}R {p['wr']:.1%}   "
            f"{'YES' if p['trades'] >= MIN_TRADES else 'no'}"
        )

    # concentration: share of band trades + band EV by top vs rest
    total_trades = sum(p["trades"] for _, p in rows)
    pos_ev = [r for r in rows if r[1]["ev"] > 0]
    neg_ev = [r for r in rows if r[1]["ev"] <= 0]
    pos_trades = sum(p["trades"] for _, p in pos_ev)
    neg_trades = sum(p["trades"] for _, p in neg_ev)
    adm_tickers = [r for r in rows if r[1]["trades"] >= MIN_TRADES and r[1]["ev"] > 0]
    print(f"\n  band total trades={total_trades}")
    print(
        f"  tickers EV>0: {len(pos_ev)} ({pos_trades}t, "
        f"{100*pos_trades/max(total_trades,1):.0f}% of trades) | "
        f"EV<=0: {len(neg_ev)} ({neg_trades}t)"
    )
    print(
        f"  positive AND >=MIN_TRADES tickers: {len(adm_tickers)} "
        f"-> {[tk for tk, _ in adm_tickers]}"
    )

    # pooled EV across admissible tickers ONLY (mirror _pooled_returns)
    all_r = np.concatenate([pnls for pnls in band.values()]) if band else np.array([])
    adq = {tk: pnls for tk, pnls in band.items() if len(pnls) >= MIN_TRADES}
    adq_r = np.concatenate([pnls for pnls in adq.values()]) if adq else np.array([])
    print(f"\n  pooled band EV all tickers   = {all_r.mean():+.3f}R (n={all_r.size})")
    print(f"  pooled band EV admissible-only= {adq_r.mean():+.3f}R (n={adq_r.size})")
    wins = adq_r[adq_r > 0]
    losses = adq_r[adq_r <= 0]
    if losses.size:
        aw = float(np.mean(wins)) if wins.size else 0.0
        al = abs(float(np.mean(losses)))
        rr = aw / al if al > 0 else 0.0
        wr = float(np.mean(adq_r > 0))
        print(
            f"  admissible band: WR={wr:.1%} rr={rr:.2f} -> "
            f"EV={wr*rr-(1-wr):+.3f}R, breakeven WR={1/(1+rr):.1%}"
        )

    # top-3 concentration test
    top3 = heapq.nlargest(3, band.items(), key=lambda kv: pool(kv[1])["ev"])
    top3_trades = sum(len(pnls) for _, pnls in top3)
    print(
        f"\n  top-3 EV tickers = {[(tk, pool(p)['ev']) for tk, p in top3]} "
        f"= {top3_trades}t ({100*top3_trades/max(total_trades,1):.0f}% of band)"
    )


if __name__ == "__main__":
    import heapq

    main()
