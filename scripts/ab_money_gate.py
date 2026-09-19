"""scripts/ab_money_gate.py — cohort-level money gate over a trade trace.

The deep funnel reports only aggregate stats; this recomputes the Bailey &
López de Prado deflated-SR money gate for ARBITRARY entry cohorts selected
from a ``--trace FILE`` (dumped by ab_brain_replay --trace), augmented with
per-trade market context computed offline from the same data deck.

Cohort cuts (chainable):
  --tickers        comma list (edge-name subset / IS-selected universe)
  --min-score      |entry signal| must clear this (conviction lift)
  --min-atr-pct    ATR/last at entry must clear this (cost-feasibility floor)
  --regime         restrict to ranged/normal/trending_bullish/trending_bearish
                   /volatile/unknown (entry-bar local regime, same detector
                   as production)
  --fraction       first F by elapsed bars (in-sample)
  --oos-last       last F by elapsed bars (out-of-sample)

Parameters that must stay FROZEN between the screening run and the confirm
run are reported; the gate verdict printed at the end mirrors the funnel's
``deflated_sr > 0`` with the given ``--trials`` multiplicity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats

from hanoon_prime.brain.regime import RegimeDetector
from hanoon_prime.eyes import _read_csv

GAMMA = 0.5772156649
_VOL_WIN = 20
_TREND_WIN = 20
_HIST = 200


def _deflated(r: list[float], trials: int) -> dict[str, float]:
    """Exact copy of the funnel's deflation so veredicts are comparable."""
    if len(r) < 4:
        return {"n": len(r), "sr": 0.0, "sr_max": 0.0, "deflated": 0.0}
    a = np.asarray(r, dtype=float)
    mu = float(np.mean(a))
    sd = float(np.std(a))
    if sd <= 1e-12:
        return {"n": len(r), "sr": 0.0, "sr_max": 0.0, "deflated": 0.0}
    sr = mu / sd
    n = float(len(a))
    skew = float(stats.skew(a))
    kurt = float(stats.kurtosis(a, fisher=True))
    v_est = (1.0 + 0.5 * sr**2 - skew * sr + 0.25 * kurt * sr**2) / n
    v_est = max(v_est, 1e-12)
    trials_f = max(2, int(trials))
    z_max = (1.0 - GAMMA) * stats.norm.ppf(
        1.0 - 1.0 / trials_f
    ) + GAMMA * stats.norm.ppf(1.0 - 1.0 / (trials_f * np.e))
    sr_max = float(np.sqrt(v_est)) * z_max
    denom = float(np.sqrt(max(1.0 - skew * sr + 0.25 * kurt * sr**2, 1e-12)))
    deflated = (sr - sr_max) / denom if denom > 0 else 0.0
    return {"n": len(r), "sr": sr, "sr_max": sr_max, "deflated": deflated}


def _ticker_regime_points(c: np.ndarray) -> dict[int, str]:
    """Regime labels per bar for one close array (vectorized semantics)."""
    ret = np.diff(c) / np.maximum(c[:-1], 1e-12)
    n = int(len(ret))
    vol = np.zeros(n)
    for j in range(_VOL_WIN - 1, n):
        vol[j] = float(np.std(ret[j - _VOL_WIN + 1 : j + 1]))
    x = np.arange(_TREND_WIN, dtype=float) - float(_TREND_WIN - 1) / 2.0
    sxx = float(np.sum(x * x))
    trend = np.zeros(n)
    for j in range(_TREND_WIN - 1, n):
        w = c[j - _TREND_WIN + 1 : j + 1]
        std = float(np.std(w))
        if std > 0:
            slope = float(np.dot(x, w - np.mean(w))) / sxx
            trend[j] = float(np.clip(slope / (std + 1e-12) * 5.0, -1.0, 1.0))
    labels: dict[int, str] = {}
    for i in range(200, len(c) - 1):
        vi = vol[i - 1]
        hist = vol[max(198, i - 1 - _HIST) : i]
        pct = float(np.mean(hist <= vi)) if len(hist[hist > 0]) >= 5 else 0.5
        at = abs(trend[i])
        if pct > 0.80:
            labels[i] = "volatile"
        elif at > 0.3 and pct < 0.60:
            labels[i] = f"trending_{'bullish' if trend[i] > 0 else 'bearish'}"
        elif at < 0.1 and pct < 0.40:
            labels[i] = "ranging"
        else:
            labels[i] = "normal"
    return labels


def load_regimes(data_dir: Path, tickers: set[str]) -> dict[str, dict[int, str]]:
    """Per-ticker entry-bar regime labels, vectorized clone of production.

    Matches RegimeDetector semantics (volatility percentile over the rolling
    history + normalized trend + ``_classify`` thresholds). Inference only —
    the labels feed a cohort analysis, not the trading loop.
    """
    out: dict[str, dict[int, str]] = {}
    for path in data_dir.glob("*_1min.csv"):
        name = path.stem.replace("_1min", "")
        if name not in tickers:
            continue
        c = np.asarray(_read_csv(path)["close"], dtype=float)
        if len(c) < 200:
            continue
        out[name] = _ticker_regime_points(c)
    return out


def _cohort_matches(
    trade: dict[str, Any],
    regimes: dict[str, dict[int, str]],
    lengths: dict[str, int],
    args: argparse.Namespace,
) -> bool:
    """Apply the frozen cohort cuts to one traced trade."""
    if args.tickers and trade["ticker"] not in args.tickers.split(","):
        return False
    if abs(float(trade["score"])) < args.min_score:
        return False
    if float(trade["atr_pct"]) < args.min_atr_pct:
        return False
    if args.regime:
        label = regimes.get(trade["ticker"], {}).get(int(trade["entry_bar"]), "unknown")
        if label != args.regime:
            return False
    ntot = lengths.get(trade["ticker"], 0)
    denom = max(ntot - 200, 1)
    frac = (int(trade["entry_bar"]) - 200) / denom if ntot else 1.0
    if args.fraction < 1.0 and frac >= args.fraction:
        return False
    if args.oos_last > 0.0 and frac < 1.0 - args.oos_last:
        return False
    slice_end = (
        args.fraction
        if args.fraction < 1.0
        else (1.0 - args.oos_last)
        if args.oos_last > 0.0
        else 1.0
    )
    exit_frac = (int(trade["exit_bar"]) - 200) / denom if ntot else 1.0
    if exit_frac >= slice_end:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--data-dir", default="data/research/ab_deep_deck")
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--min-atr-pct", type=float, default=0.0)
    ap.add_argument("--regime", default="")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--oos-last", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=13)
    args = ap.parse_args()

    trace = json.loads(Path(args.trace).read_text())
    if not trace:
        print("empty trace")
        return 1
    tickers = {t["ticker"] for t in trace if t["ticker"] != ""}
    regimes = load_regimes(Path(args.data_dir), tickers)
    lengths: dict[str, int] = {
        path.stem.replace("_1min", ""): len(_read_csv(path)["close"])
        for path in Path(args.data_dir).glob("*_1min.csv")
    }

    rows = [t for t in trace if _cohort_matches(t, regimes, lengths, args)]

    r = [float(x["r"]) for x in rows]
    d = _deflated(r, args.trials)
    print(
        f"cohort: n={d['n']:4d} ev={float(np.mean(r)) if r else 0.0:+.3f} "
        f"SR={d['sr']:+.3f} sr_max(@trials={args.trials})={d['sr_max']:+.3f} "
        f"deflated={d['deflated']:+.3f}  {'PASS' if d['deflated'] > 0 else 'FAIL'}"
    )
    if args.fraction < 1.0 or args.oos_last > 0.0:
        print(
            "frozen params: trials=%d min_score=%.2f min_atr_pct=%.2e "
            "tickers=%s" % (args.trials, args.min_score, args.min_atr_pct, args.tickers)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
