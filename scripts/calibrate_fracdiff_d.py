#!/usr/bin/env python3
"""Offline calibration: find optimal per-ticker d* for fractional differentiation.

Reads 1-minute bar CSVs from data/market_data/, runs ADF grid search via
find_min_d(), and writes results to runtime/fracdiff_d_config.json.

Usage:
    python scripts/calibrate_fracdiff_d.py [--data-dir data/market_data] [--out runtime/fracdiff_d_config.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hanoon_prime.fracdiff import find_min_d
from hanoon_prime.immune import FRACDIFF_D


def load_close(csv_path: Path) -> np.ndarray | None:
    """Load Close column from a CSV, skip header, return as float array."""
    closes: list[float] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                closes.append(float(row["Close"]))
            except (ValueError, KeyError):
                continue
    if len(closes) < 200:
        return None
    return np.array(closes, dtype=float)


def calibrate(data_dir: Path, p_threshold: float = 0.05) -> dict[str, float]:
    """Run find_min_d on every ticker CSV, return {ticker: d*}."""
    csvs = sorted(data_dir.glob("*_1min.csv"))
    if not csvs:
        print(f"No *_1min.csv found in {data_dir}")
        return {}

    config: dict[str, float] = {"DEFAULT": FRACDIFF_D}
    total = len(csvs)
    found = 0
    skipped = 0
    t0 = time.time()

    for i, csv_path in enumerate(csvs):
        ticker = csv_path.stem.replace("_1min", "")
        series = load_close(csv_path)
        if series is None:
            skipped += 1
            continue
        try:
            d_star, _ = find_min_d(series)
            config[ticker] = round(d_star, 2)
            found += 1
        except Exception as exc:
            print(f"  {ticker}: FAILED ({exc})")
            skipped += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i + 1}/{total}] {found} calibrated, {skipped} skipped, {elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    print(f"Done: {found} tickers calibrated, {skipped} skipped in {elapsed:.1f}s")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate per-ticker fracdiff d*")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "market_data")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runtime" / "fracdiff_d_config.json"
    )
    parser.add_argument("--p-threshold", type=float, default=0.05)
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    config = calibrate(args.data_dir, args.p_threshold)
    if not config:
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(config)} entries to {args.out}")

    d_vals = [v for k, v in config.items() if k != "DEFAULT"]
    if d_vals:
        print(
            f"d* range: {min(d_vals):.2f} — {max(d_vals):.2f}, median: {np.median(d_vals):.2f}"
        )


if __name__ == "__main__":
    main()
