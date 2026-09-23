"""brain.strategy_priors — seeded 'educated' strategies for JULI.

Hand-curated priors give JULI a trained (not blank) start: classic, bounded
approaches the registry, bandit, and research client can build on.
"""

from __future__ import annotations

from typing import Any

# name, thesis, entry, exit, risk, conditions, regime, sizing, score_mod, conf
_PRIORS: list[tuple[str, str, str, str, str, str, str, float, float, float]] = [
    (
        "trend-pullback",
        "Buy shallow pullbacks inside an up-trend; exit on fade.",
        "Pullback to rising 20-bar EMA, trend_up, ADX>25.",
        "Trail under last swing low; MACD hist cross down.",
        "Hard stop 2xATR; never add past one pullback.",
        "trend_up only, liquid ticker, no gap-catalyst.",
        "trend_up",
        1.05,
        0.0,
        0.55,
    ),
    (
        "range-fade",
        "Fade extremes in a well-defined range; mean reversion.",
        "Buy near lower band in a range regime (RSI<35, support).",
        "Mid-range or upper band; scale out 50% at the mean.",
        "Invalidation stop past range extreme; no entry on breakout.",
        "range regime only, ADX<20 confirmed.",
        "range",
        0.95,
        0.0,
        0.5,
    ),
    (
        "vol-scalp",
        "Size down in volatile tape; take quicker profits.",
        "Confirmed breakout in a vol regime, tight 1.5xATR stop.",
        "Quick 1.5R target; giveback lock at 0.4R.",
        "Sized below neutral; requires spread_tightness positive.",
        "vol regime only, width constrained.",
        "vol",
        0.8,
        0.0,
        0.5,
    ),
    (
        "mm-absorption",
        "Trade with market-makers when one-sided flow fails to move the level.",
        "Active absorption signal (|absorption| >= floor), scalp only.",
        "Exit when absorption level breaks (signal collapses/flips) or 5-10 tick target.",
        "Tighter 1.5xATR stop; fixed-tick target override; forced scalp horizon.",
        "Liquid ticker, live tape only — no OHLCV backtest equivalent.",
        "unknown",
        0.9,
        0.02,
        0.55,
    ),
]


def seeded_priors() -> list[dict[str, Any]]:
    """Hand-curated 'educated' priors so JULI starts trained, not blank."""
    return [
        {
            "name": n,
            "thesis": th,
            "entry": en,
            "exit": ex,
            "risk": rk,
            "conditions": co,
            "regime": rg,
            "sizing": sz,
            "score_mod": sm,
            "confidence": cf,
            "source": "seeded",
        }
        for n, th, en, ex, rk, co, rg, sz, sm, cf in _PRIORS
    ]


__all__ = ["seeded_priors"]
