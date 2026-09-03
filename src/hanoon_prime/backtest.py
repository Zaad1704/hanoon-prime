"""hanoon_prime.backtest — runs the JULI pipeline on historical data.

Top-level orchestrator. Delegates bar-by-bar simulation to hands.py
and performance calculation to metrics.py.

CI uses this to enforce R2: no merges if expectancy < 0 on any ticker.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from .eyes import load_ohlcv
from .hands import simulate_ticker
from .immune import ATR_PERIOD, ATR_STOP_MULT, ATR_TARGET_MULT, EDGE_LOOKBACK
from .metrics import calculate_metrics

log = logging.getLogger(__name__)


def _discover_tickers(data_dir: Path) -> list[str]:
    """Auto-discover all valid CSV tickers in a directory."""
    return sorted(f.stem.replace("_1min", "") for f in data_dir.glob("*_1min.csv"))


def backtest_ticker(
    ticker: str,
    close: Any,
    high: Any,
    low: Any,
    volume: Any,
    window: int = EDGE_LOOKBACK,
    brain: Optional[Any] = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Backtest the full JULI pipeline on one ticker's 1-min bars.

    Returns metrics dict (also writes JSON if output_dir given).
    """
    trades, equity_curve = simulate_ticker(
        ticker, close, high, low, volume, window, brain
    )
    metrics = calculate_metrics(ticker, trades, equity_curve)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{ticker}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def run_backtest(
    tickers: list[str],
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    window: int = EDGE_LOOKBACK,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Run backtest on all tickers. Returns (results, errors)."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else None
    results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for ticker in tickers:
        result = _try_backtest_ticker(ticker, data_dir, window, output_dir, errors)
        if result is not None:
            results[ticker] = result
            status = "✅" if result["ev_per_trade"] > 0 else "❌"
            tr = f"{result['total_trades']} trades"
            wr = (
                f"WR={result.get('win_rate', 0):.1%}"
                if result["total_trades"]
                else "WR=NA"
            )
            ev = (
                f"EV={result['ev_per_trade']:.3f}R"
                if result["total_trades"]
                else "EV=0"
            )
            log.info(
                "  %-8s %s  %s  %s  %s  R:R=%s",
                ticker,
                status,
                tr,
                wr,
                ev,
                f"{result.get('realized_rr', 0):.2f}",
            )

    return results, errors


def _try_backtest_ticker(
    ticker: str,
    data_dir: Path,
    window: int,
    output_dir: Path | None,
    errors: list[str],
) -> dict[str, Any] | None:
    """Load one ticker, backtest it, append errors on failure."""
    csv_path = data_dir / f"{ticker}_1min.csv"
    if not csv_path.exists():
        log.info("SKIP: %s not found", csv_path)
        errors.append(f"{ticker}: no data file")
        return None
    try:
        data = load_ohlcv(csv_path)
    except ValueError:
        log.info("SKIP: %s — corrupt/empty data file", ticker)
        errors.append(f"{ticker}: corrupt data file")
        return None
    close = data["close"]
    if len(close) < window + 10:
        log.info("SKIP: %s — only %s bars", ticker, len(close))
        errors.append(f"{ticker}: insufficient bars ({len(close)})")
        return None
    return backtest_ticker(
        ticker,
        close,
        data["high"],
        data["low"],
        data["volume"],
        window=window,
        output_dir=output_dir,
    )


def _print_results(results: dict[str, dict[str, Any]]) -> int:
    """Print backtest summary and return exit code (0 if all profitable)."""
    profitable = sum(1 for m in results.values() if m["ev_per_trade"] > 0)
    total_trades = sum(m["total_trades"] for m in results.values())
    log.info("\n%s", "=" * 60)
    log.info(
        "RESULTS: %s/%s tickers profitable | %s trades",
        profitable,
        len(results),
        total_trades,
    )
    log.info("%s", "=" * 60)
    unprofitable = _find_unprofitable(results)
    if unprofitable:
        log.info("\n💀 PROFITABILITY GATE FAILED: %s", unprofitable)
        return 1
    log.info("\n✅ All tickers pass profitability gate.")
    return 0


def _find_unprofitable(results: dict[str, dict[str, Any]]) -> list[str]:
    """Return list of tickers with negative EV."""
    return [
        t
        for t, m in results.items()
        if m["ev_per_trade"] <= 0 and m["total_trades"] > 0
    ]


def main() -> int:
    """CLI entry point for backtest."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="HANOON PRIME backtest")
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--window", type=int, default=EDGE_LOOKBACK)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tickers = (
        _discover_tickers(data_dir)
        if args.tickers.upper() == "ALL"
        else [t.strip() for t in args.tickers.split(",")]
    )
    log.info("Running backtest on %s tickers from %s", len(tickers), data_dir)
    log.info(
        "  Window: %s bars  (ATR=%s×%s/%s)",
        args.window,
        ATR_PERIOD,
        ATR_STOP_MULT,
        ATR_TARGET_MULT,
    )
    results, _ = run_backtest(tickers, data_dir, args.output, args.window)
    return _print_results(results)


if __name__ == "__main__":
    sys.exit(main())
