"""hanoon_prime.calibrate — auto-calibration CLI.

Run:  python -m hanoon_prime.calibrate --data-dir /path/to/csvs

The system evaluates its own indicators, tunes weights by edge
strength, and verifies profitability on historical data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backtest import backtest_ticker
from .eyes import load_ohlcv
from .immune import EDGE_LOOKBACK, INDICATOR_NAMES
from .validator import calibrate_weights, evaluate_indicator_pooled

log = logging.getLogger(__name__)


@dataclass
class Calibration:
    """Result of auto-calibration: tuned weights + edge stats."""

    weights: dict[str, float] = field(default_factory=dict)
    indicator_corrs: dict[str, float] = field(default_factory=dict)
    indicator_pvalues: dict[str, float] = field(default_factory=dict)
    indicator_significant: dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.5
    n_indicators_significant: int = 0
    n_indicators_total: int = len(INDICATOR_NAMES)

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibration result to a plain dict for JSON export."""
        return {
            "weights": self.weights,
            "confidence": self.confidence,
            "n_indicators_significant": self.n_indicators_significant,
            "indicator_corrs": self.indicator_corrs,
            "indicator_pvalues": self.indicator_pvalues,
        }


def _all_tickers(data_dir: Path) -> list[str]:
    """Auto-discover all valid CSV tickers."""
    result: list[str] = []
    for f in sorted(data_dir.glob("*_1min.csv")):
        try:
            load_ohlcv(f)
            result.append(f.stem.replace("_1min", ""))
        except ValueError:
            continue
    return result


def calibrate(tickers: list[str], data_dir: Path) -> Calibration:
    """Full auto-calibration: evaluate edge, weight indicators, verify."""
    if not tickers:
        tickers = _all_tickers(data_dir)
    cal = Calibration()
    log.info("Step 1: Evaluating indicator edge (pooled permutation test)...")
    pooled = evaluate_indicator_pooled(tickers, data_dir, n_perm=500)
    for name, info in pooled.items():
        sig = "✅" if info["significant"] else "❌"
        log.info(
            "  %-25s: corr=%+.4f  p=%.4f  n=%s  %s",
            name,
            info["corr"],
            info["pvalue"],
            info["n_samples"],
            sig,
        )
    cal.indicator_corrs = {k: v["corr"] for k, v in pooled.items()}
    cal.indicator_pvalues = {k: v["pvalue"] for k, v in pooled.items()}
    cal.indicator_significant = {k: v["significant"] for k, v in pooled.items()}
    cal.n_indicators_significant = sum(1 for v in pooled.values() if v["significant"])

    log.info("\nStep 2: Auto-calibrating weights by edge strength...")
    cal.weights = calibrate_weights(pooled)
    for name, w in sorted(cal.weights.items(), key=lambda x: -abs(x[1])):
        c = cal.indicator_corrs.get(name, 0.0)
        log.info("  %-25s: %+.3f  (corr=%+.4f)", name, w, c)

    log.info("\nStep 3: Verifying profitability on full universe...")
    profitable, total = _check_profitability(tickers, data_dir)
    cal.confidence = 0.4 * cal.n_indicators_significant / len(
        INDICATOR_NAMES
    ) + 0.6 * profitable / max(total, 1)
    log.info(
        "\n  Edge score: %s/%s", cal.n_indicators_significant, len(INDICATOR_NAMES)
    )
    log.info("  Profitable: %s/%s tickers", profitable, total)
    log.info("  Self-confidence: %.0f%%", cal.confidence)
    return cal


def _check_profitability(tickers: list[str], data_dir: Path) -> tuple[int, int]:
    """Run full backtest; count profitable tickers."""
    profitable, total = 0, 0
    for ticker in tickers:
        path = data_dir / f"{ticker}_1min.csv"
        if not path.exists():
            continue
        try:
            data = load_ohlcv(path)
        except ValueError:
            continue
        if len(data["close"]) < EDGE_LOOKBACK + 50:
            continue
        m = backtest_ticker(
            ticker,
            data["close"],
            data["high"],
            data["low"],
            data["volume"],
        )
        total += 1
        is_prof = m["ev_per_trade"] > 0
        if is_prof:
            profitable += 1
        status = "✅" if is_prof else "❌"
        log.info(
            "  %-8s %s  EV=%+.3fR  %s trades",
            ticker,
            status,
            m["ev_per_trade"],
            m["total_trades"],
        )
    return profitable, total


def main() -> int:
    """CLI entry point for auto-calibration."""
    parser = argparse.ArgumentParser(description="HANOON PRIME auto-calibration")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tickers", default="ALL")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.tickers.upper() == "ALL":
        tickers = _all_tickers(data_dir)
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]

    log.info("HANOON PRIME Auto-Calibration")
    log.info("  Tickers: %s", len(tickers))
    cal = calibrate(tickers, data_dir)

    if args.output:
        result = cal.to_dict()
        result["safe_to_activate"] = cal.confidence > 0.5
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        log.info("\n  Written to: %s", args.output)

    return 0 if cal.confidence > 0.5 else 1


def _configure_logging() -> None:
    """Configure logging for CLI output to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


if __name__ == "__main__":
    _configure_logging()
    sys.exit(main())
