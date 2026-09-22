#!/usr/bin/env python3
"""Offline DNN meta-label trainer: fixture data → Purged CV → MetaDNN weights.

Pipeline:
  1. Replay OFF variant to harvest entry signals (features + outcomes).
  2. Train MetaDNN with purged walk-forward CV on the harvested data.
  3. Persist the trained weights to META_DNN_FILE for live inference.

Usage:
  python scripts/train_meta_dnn.py --data-dir data/fixtures --tickers ALL \\
      --epochs 50 --output runtime/juli_meta_dnn.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from hanoon_prime.brain.learning_config import META_DNN_FILE
from hanoon_prime.brain.meta_label_dnn import INPUT_DIM, MetaDNN, expand_features
from hanoon_prime.brain.mtf import (
    compute_obi,
    compute_tf5_trend_alignment,
    compute_tf15_vol_expansion,
    compute_vpin,
)

WINDOW = 40
ATR_PERIOD = 14
VERTICAL_BARS = 30
STOP_ATR_MULT = 2.0
TARGET_ATR_MULT = 6.0


def load_fixtures(data_dir: Path, tickers: list[str]) -> dict[str, dict[str, list]]:
    """Load 1-min CSV fixtures into {ticker: {close, high, low, volume, ...}}."""
    fixtures: dict[str, dict[str, list]] = {}
    for ticker in tickers:
        csv_path = data_dir / f"{ticker}_1min.csv"
        if not csv_path.exists():
            log.warning("Fixture not found: %s", csv_path)
            continue
        lines = csv_path.read_text().strip().split("\n")
        if len(lines) < 4:
            continue
        close, high, low, volume = [], [], [], []
        for line in lines[3:]:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                o, h, l, c, v = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
            except (ValueError, IndexError):
                continue
            close.append(c)
            high.append(h)
            low.append(l)
            volume.append(v)
        if len(close) > WINDOW + 60:
            fixtures[ticker] = {
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
            }
    return fixtures


def _compute_atr(
    high: list, low: list, close: list, i: int, period: int = ATR_PERIOD
) -> float:
    """True-range ATR at bar i."""
    if i < period + 1:
        return 1e-6
    h = np.asarray(high[i - period : i + 1], dtype=float)
    l = np.asarray(low[i - period : i + 1], dtype=float)
    c = np.asarray(close[i - period : i + 1], dtype=float)
    tr = np.maximum(
        h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    return float(np.mean(tr))


def _compute_obi(high: list, low: list, close: list, i: int) -> float:
    """Route to the shared feature source of truth (brain.mtf)."""
    return compute_obi(high, low, close, end=i)


def _compute_vpin(volume: list, close: list, i: int) -> float:
    """Route to the shared feature source of truth (brain.mtf)."""
    return compute_vpin(volume, close, end=i)


def _triple_barrier_outcome(
    close: list,
    high: list,
    low: list,
    entry_idx: int,
    entry_price: float,
    atr: float,
    direction: int,
    vertical: int = VERTICAL_BARS,
    stop_mult: float = STOP_ATR_MULT,
    target_mult: float = TARGET_ATR_MULT,
) -> tuple[int, float]:
    """Scan forward from entry; return (won, r_mult).

    Mirrors labels.py geometry: stop = atr*stop_mult, target = atr*target_mult.
    Stop checked before target within each bar (pessimistic ordering).
    """
    stop = entry_price - stop_mult * atr * direction
    target = entry_price + target_mult * atr * direction
    n = len(close)
    for j in range(entry_idx + 1, min(entry_idx + vertical + 1, n)):
        lo, hi = float(low[j]), float(high[j])
        if direction == 1:
            if lo <= stop:
                return 0, -stop_mult
            if hi >= target:
                return 1, target_mult
        else:
            if hi >= stop:
                return 0, -stop_mult
            if lo <= target:
                return 1, target_mult
    last_price = float(close[min(entry_idx + vertical, n - 1)])
    pnl = (last_price / entry_price - 1.0) * direction
    return (1 if pnl > 0 else 0), pnl * (TARGET_ATR_MULT / atr) if atr > 0 else pnl


def _vol_percentile(volume: list, i: int, window: int = 200) -> float:
    """Volume percentile at bar i over trailing window."""
    start = max(0, i - window)
    vols = volume[start : i + 1]
    if len(vols) < 5:
        return 0.5
    current = volume[i]
    return float(np.mean(np.asarray(vols) <= current))


def harvest_entries(
    fixtures: dict[str, dict[str, list]], min_spacing: int = 30
) -> list[tuple[list[float], bool, float]]:
    """Generate entry training pairs directly from price data.

    At each bar, direction is inferred from short-term momentum (last 10
    bars).  Features are extracted from the local bar context.  The label
    comes from the triple-barrier outcome (stop/target/timeout).
    """
    entries: list[tuple[list[float], bool, float]] = []
    for ticker, bars in fixtures.items():
        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        volume = bars["volume"]
        n = len(close)
        if n <= WINDOW + VERTICAL_BARS + 10:
            continue
        last_entry_bar = -min_spacing
        n_before = len(entries)
        for i in range(WINDOW, n - VERTICAL_BARS):
            if i - last_entry_bar < min_spacing:
                continue
            atr = _compute_atr(high, low, close, i)
            if atr < 1e-8:
                continue
            lookback = min(10, i)
            momentum = close[i] - close[i - lookback] if lookback > 0 else 0.0
            direction = 1 if momentum > 0 else -1
            entry_price = float(close[i])
            won, r = _triple_barrier_outcome(
                close, high, low, i, entry_price, atr, direction
            )
            vol_pct = _vol_percentile(volume, i)
            confidence = (
                min(0.95, max(0.5, 0.5 + abs(momentum) / atr * 0.3)) if atr > 0 else 0.5
            )
            score = np.tanh(momentum / atr) if atr > 0 else 0.0
            atr_ratio = atr / entry_price if entry_price > 0 else 0.0
            obi = _compute_obi(high, low, close, i)
            vpin = _compute_vpin(volume, close, i)
            tf5_align = compute_tf5_trend_alignment(close, high, low, end=i)
            tf15_vol = compute_tf15_vol_expansion(close, high, low, end=i)
            feat = expand_features(
                confidence,
                score,
                vol_pct,
                int(direction),
                atr_ratio,
                obi,
                vpin,
                tf5_align,
                tf15_vol,
            )
            if len(feat) != INPUT_DIM:
                raise RuntimeError(f"feature dim {len(feat)} != INPUT_DIM {INPUT_DIM}")
            weight = max(0.1, min(1.0, 1.0 + r / TARGET_ATR_MULT))
            entries.append((feat, bool(won), weight))
            last_entry_bar = i
        log.info(
            "Generated %d entries from %s",
            len(entries) - n_before,
            ticker,
        )
    return entries


def purged_train_test(
    entries: list[tuple[list[float], bool, float]],
    n_folds: int = 5,
    test_frac: float = 0.2,
    embargo_bars: int = 30,
) -> list[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Purged walk-forward split: returns list of (X_train, y_train, sw_train, X_test, y_test, sw_test)."""
    n = len(entries)
    if n < 100:
        log.warning("Only %d entries — using simple random split", n)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        split = int(n * 0.8)
        train_idx, test_idx = idx[:split], idx[split:]
        X = np.asarray([e[0] for e in entries], dtype=np.float64)
        y = np.asarray([e[1] for e in entries], dtype=np.float64)
        sw = np.asarray([e[2] for e in entries], dtype=np.float64)
        return [
            (
                X[train_idx],
                y[train_idx],
                sw[train_idx],
                X[test_idx],
                y[test_idx],
                sw[test_idx],
            )
        ]
    fold_size = n // n_folds
    folds = []
    for fold_i in range(n_folds):
        test_start = fold_i * fold_size
        test_end = min(test_start + fold_size, n)
        train_indices = list(range(0, max(0, test_start - embargo_bars)))
        train_indices += list(range(min(n, test_end + embargo_bars), n))
        test_indices = list(range(test_start, test_end))
        if len(train_indices) < 10 or len(test_indices) < 5:
            continue
        X = np.asarray([entries[i][0] for i in train_indices], dtype=np.float64)
        y = np.asarray([entries[i][1] for i in train_indices], dtype=np.float64)
        sw = np.asarray([entries[i][2] for i in train_indices], dtype=np.float64)
        X_test = np.asarray([entries[i][0] for i in test_indices], dtype=np.float64)
        y_test = np.asarray([entries[i][1] for i in test_indices], dtype=np.float64)
        sw_test = np.asarray([entries[i][2] for i in test_indices], dtype=np.float64)
        folds.append((X, y, sw, X_test, y_test, sw_test))
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing *_1min.csv files"
    )
    parser.add_argument(
        "--tickers", default="ALL", help="Comma-separated tickers or ALL"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs per fold"
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Adam learning rate")
    parser.add_argument("--batch", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--n-folds", type=int, default=5, help="Purged CV folds")
    parser.add_argument(
        "--embargo", type=int, default=30, help="Embargo bars between train/test"
    )
    parser.add_argument(
        "--output", default=None, help="Output JSON path (default: META_DNN_FILE)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    data_dir = Path(args.data_dir)
    ticker_arg = (
        [p.stem.replace("_1min", "") for p in sorted(data_dir.glob("*_1min.csv"))]
        if args.tickers.upper() == "ALL"
        else [t.strip() for t in args.tickers.split(",")]
    )

    log.info("Loading fixtures for %d tickers from %s", len(ticker_arg), data_dir)
    fixtures = load_fixtures(data_dir, ticker_arg)
    if not fixtures:
        log.error("No fixtures loaded")
        return 1
    log.info("Loaded %d tickers", len(fixtures))

    log.info("Harvesting entry signals via OFF-variant brain replay...")
    entries = harvest_entries(fixtures)
    if len(entries) < 50:
        log.error("Only %d entries harvested — insufficient for training", len(entries))
        return 1
    log.info("Harvested %d total entries", len(entries))

    n_won = sum(1 for _, w, _ in entries if w)
    log.info(
        "Win rate: %.1f%% (%d/%d)", 100 * n_won / len(entries), n_won, len(entries)
    )

    log.info(
        "Creating purged CV splits (folds=%d, embargo=%d)...",
        args.n_folds,
        args.embargo,
    )
    folds = purged_train_test(entries, n_folds=args.n_folds, embargo_bars=args.embargo)
    if not folds:
        log.error("No valid CV folds produced")
        return 1
    log.info("Created %d valid folds", len(folds))

    out_path = Path(args.output) if args.output else META_DNN_FILE
    dnn = MetaDNN(path=out_path)

    all_train_losses = []
    all_test_losses = []
    all_test_preds = []
    all_test_labels = []
    all_test_weights = []

    for fold_i, (X_tr, y_tr, sw_tr, X_te, y_te, sw_te) in enumerate(folds):
        log.info("Fold %d: train=%d test=%d", fold_i + 1, len(X_tr), len(X_te))
        losses = dnn.train(
            X_tr,
            y_tr,
            sample_weight=sw_tr,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch,
        )
        all_train_losses.extend(losses)
        preds = np.asarray([dnn.predict(x.tolist()) for x in X_te], dtype=np.float64)
        eps = 1e-12
        test_loss = float(
            -np.mean(
                sw_te
                * (y_te * np.log(preds + eps) + (1 - y_te) * np.log(1 - preds + eps))
            )
        )
        all_test_losses.append(test_loss)
        all_test_preds.extend(preds.tolist())
        all_test_labels.extend(y_te.tolist())
        all_test_weights.extend(sw_te.tolist())
        log.info(
            "  Fold %d: final_train_loss=%.4f test_loss=%.4f",
            fold_i + 1,
            losses[-1],
            test_loss,
        )

    preds_arr = np.asarray(all_test_preds)
    labels_arr = np.asarray(all_test_labels)
    weights_arr = np.asarray(all_test_weights)
    median_pred = float(np.median(preds_arr))
    mean_pred = float(np.mean(preds_arr))
    acc = float(np.mean((preds_arr >= 0.5) == labels_arr))
    weighted_acc = (
        float(np.average((preds_arr >= 0.5) == labels_arr, weights=weights_arr))
        if weights_arr.sum() > 0
        else 0.0
    )

    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("  Folds: %d", len(folds))
    log.info(
        "  Total train loss range: [%.4f, %.4f]",
        min(all_train_losses),
        max(all_train_losses),
    )
    log.info("  Mean OOS test loss: %.4f", np.mean(all_test_losses))
    log.info("  Median P(Win): %.4f", median_pred)
    log.info("  Mean P(Win): %.4f", mean_pred)
    log.info("  OOS accuracy (0.5 cut): %.1f%%", 100 * acc)
    log.info("  OOS weighted accuracy: %.1f%%", 100 * weighted_acc)
    guard = dnn.guard_status()
    healthy = bool(guard.get("healthy"))
    log.info("  Ironclad guard: %s", "HEALTHY" if healthy else "COLLAPSED")
    if guard.get("reasons"):
        log.info("  Guard reasons: %s", ", ".join(guard["reasons"]))
    log.info(
        "  Model saved to: %s", out_path if healthy else "(refused — artifact kept)"
    )
    log.info("=" * 60)

    report = {
        "folds": len(folds),
        "total_entries": len(entries),
        "test_loss_mean": round(float(np.mean(all_test_losses)), 4),
        "p_win_median": round(median_pred, 4),
        "p_win_mean": round(mean_pred, 4),
        "accuracy": round(acc, 4),
        "weighted_accuracy": round(weighted_acc, 4),
        "guard_healthy": healthy,
        "guard_reasons": guard.get("reasons") or [],
        "output": str(out_path) if healthy else None,
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Report saved to: %s", report_path)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
