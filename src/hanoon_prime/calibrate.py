"""hanoon_prime.calibrate — auto-calibration CLI.

Run:  python -m hanoon_prime.calibrate --data-dir /path/to/csvs

The system evaluates its own indicators, tunes weights/thresholds/R:R,
and writes calibration.json. If profitability is proven, it emits a
"safe_to_activate" flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._calibrate import calibrate
from .data import load_ohlcv


def _all_tickers(data_dir: Path) -> list[str]:
    return sorted(
        f.stem.replace("_1min", "") for f in data_dir.glob("*_1min.csv")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="HANOON PRIME auto-calibration")
    parser.add_argument("--data-dir", required=True, help="Directory with *_1min.csv files")
    parser.add_argument("--tickers", default="ALL", help="Comma-separated tickers (or 'ALL')")
    parser.add_argument("--output", default=None, help="Output path for calibration.json")
    parser.add_argument("--n-perm", type=int, default=500, help="Permutation test iterations")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.tickers.upper() == "ALL":
        tickers = _all_tickers(data_dir)
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]

    print(f"HANOON PRIME Auto-Calibration")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Permutations: {args.n_perm}")
    print()

    print("Step 1: Evaluating indicator edge (permutation test)...")
    cal = calibrate(tickers[:50], data_dir)

    _print_calibration(cal)
    _write_output(cal, args.output)

    return 0 if cal.profitable else 1


def _print_calibration(cal):
    """Print calibrated parameters."""
    print()
    print("Weights (auto-calibrated by edge strength):")
    for name, w in sorted(cal.weights.items(), key=lambda x: -x[1]):
        p = cal.indicator_pvalues.get(name, 1.0)
        c = cal.indicator_corrs.get(name, 0.0)
        print(f"    {name:25s}: {w:.3f}  |corr|={c:.4f}  p={p:.4f}")
    print()
    print(f"  Threshold: {cal.threshold}")
    print(f"  Stop/Target: {cal.stop_pct:.1%} / {cal.target_pct:.1%}")
    print(f"  Max position: ${cal.max_position_notional}")
    print(f"  Self-confidence: {cal.confidence:.0%}")
    print(f"  Profitable on: {cal.n_profitable}/{cal.n_total} tickers")
    print(f"  Safe to activate: {'YES' if cal.profitable else 'NO'}")


def _write_output(cal, output_path):
    if output_path:
        result = cal.to_dict()
        result["safe_to_activate"] = cal.profitable
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Written to: {output_path}")

    return 0 if cal.profitable else 1


if __name__ == "__main__":
    sys.exit(main())
