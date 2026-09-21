#!/usr/bin/env python3
"""Feature permutation ablation for the DNN gatekeeper.

Loads the trained ``MetaDNN`` artifact and measures Mean-Decrease-style
sensitivity on the pooled purged OOS test folds: for each of the 15 input
features, that feature's column is randomly shuffled N times (all other
columns and the frozen model untouched) and the resulting weighted BCE loss,
accuracy and P(Win) spread are compared against baseline.

    * delta_loss ~ 0  -> dead slot (permutation cannot change predictions)
    * large positive  -> the model genuinely leans on that feature

Writes a JSON report to ``runtime/ablation_meta_dnn.json`` when ``--output``
is not overridden.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_meta_dnn import (  # noqa: E402
    harvest_entries,
    load_fixtures,
    purged_train_test,
)

from hanoon_prime.brain.meta_label_dnn import MetaDNN  # noqa: E402

log = logging.getLogger("ablate_meta_dnn")

FEATURE_NAMES: list[str] = [
    "confidence",
    "abs_score",
    "vol_pct",
    "direction",
    "atr_ratio",
    "obi",
    "vpin",
    "regime:unknown",
    "regime:trend_up",
    "regime:trend_down",
    "regime:range",
    "regime:volatile",
    "horizon:scalp",
    "horizon:momentum",
    "horizon:swing",
]


def _eval_matrix(
    model: MetaDNN, X: np.ndarray, y: np.ndarray, sw: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Weighted BCE loss + accuracy (0.5 cut) for a full feature matrix."""
    Xs = model._transform(X)
    out, _ = model.forward(Xs)
    preds = out[:, 0]
    eps = 1e-12
    loss = float(
        -np.mean(sw * (y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps)))
    )
    acc = float(np.mean((preds >= 0.5) == y))
    return preds, loss, acc


def _permuted_column(X: np.ndarray, rng: np.random.Generator, i: int) -> np.ndarray:
    """Shuffled copy of X with feature column i permuted across rows."""
    Xp = X.copy()
    Xp[:, i] = rng.permutation(Xp[:, i])
    return Xp


def _ablate_feature(
    model: MetaDNN,
    Xe: np.ndarray,
    ye: np.ndarray,
    swe: np.ndarray,
    rng: np.random.Generator,
    i: int,
    name: str,
    n_perms: int,
    base_preds: np.ndarray,
    loss0: float,
    acc0: float,
    spread0: float,
    active_tol: float,
    weak_tol: float,
    out_rows: list[dict],
) -> None:
    """Permute one feature N times and record mean deltas into out_rows."""
    d_loss: list[float] = []
    d_acc: list[float] = []
    d_spread: list[float] = []
    for _ in range(n_perms):
        preds_p, loss_p, acc_p = _eval_matrix(
            model, _permuted_column(Xe, rng, i), ye, swe
        )
        d_loss.append(loss_p - loss0)
        d_acc.append(acc0 - acc_p)
        d_spread.append(abs(float(np.std(preds_p)) - spread0))
    dl_mean = float(np.mean(d_loss))
    dl_std = float(np.std(d_loss))
    da_mean = float(np.mean(d_acc))
    ds_mean = float(np.mean(d_spread))
    if dl_mean >= active_tol:
        tag = "active"
    elif dl_mean >= weak_tol:
        tag = "weak"
    else:
        tag = "dead"
    ci = Xe[:, i]
    corr = float(np.corrcoef(base_preds, ci)[0, 1]) if ci.std() > 1e-12 else 0.0
    out_rows.append(
        {
            "idx": i,
            "feature": name,
            "ramp": tag,
            "delta_loss_mean": round(dl_mean, 5),
            "delta_loss_std": round(dl_std, 5),
            "delta_acc_mean": round(da_mean, 5),
            "delta_spread_mean": round(ds_mean, 5),
            "pred_corr_input": round(corr, 5),
        }
    )
    log.info(
        "  [%2d] %-16s %-6s dLoss=%+.5f +/- %.5f dAcc=%+.5f dSpread=%+.5f corr=%+.3f",
        i,
        name,
        tag,
        dl_mean,
        dl_std,
        da_mean,
        ds_mean,
        corr,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Argument parser."""
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
        "--model", default=None, help="Model JSON path (default: META_DNN_FILE)"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Purged CV folds to pool OOS"
    )
    parser.add_argument(
        "--embargo", type=int, default=30, help="Embargo bars between train/test"
    )
    parser.add_argument(
        "--n-perms", type=int, default=5, help="Permutations per feature"
    )
    parser.add_argument(
        "--active-tol", type=float, default=0.002, help="dLoss to call a feature active"
    )
    parser.add_argument(
        "--weak-tol", type=float, default=0.0005, help="dLoss to call a feature weak"
    )
    parser.add_argument(
        "--seed", type=int, default=20260921, help="RNG seed for permutations"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON report path (default: runtime/ablation_meta_dnn.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Permutation ablation entrypoint; exit code 0 on success."""
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    data_dir = Path(args.data_dir)
    ticker_arg = (
        [p.stem.replace("_1min", "") for p in sorted(data_dir.glob("*_1min.csv"))]
        if args.tickers.upper() == "ALL"
        else [t.strip() for t in args.tickers.split(",")]
    )
    fixtures = load_fixtures(data_dir, ticker_arg)
    entries = harvest_entries(fixtures)
    if not entries:
        log.error("No entries harvested from %s", data_dir)
        return 1
    model = MetaDNN(path=Path(args.model) if args.model else None)
    if model._defective:
        log.error("Model artifact rejected — ablation aborted (no valid model)")
        return 1
    folds = purged_train_test(entries, n_folds=args.n_folds, embargo_bars=args.embargo)
    Xe = np.concatenate([f[3] for f in folds])
    ye = np.concatenate([f[4] for f in folds])
    swe = np.concatenate([f[5] for f in folds])
    base_preds, loss0, acc0 = _eval_matrix(model, Xe, ye, swe)
    base_spread = float(np.std(base_preds))
    log.info("=" * 94)
    log.info("Feature permutation ablation on pooled purged OOS folds")
    log.info("  Model: %s", model._path)
    log.info(
        "  OOS rows: %d, baseline BCE=%.5f acc=%.5f spread=%.5f",
        len(Xe),
        loss0,
        acc0,
        base_spread,
    )
    log.info("  Permutations per feature: %d (seed=%d)", args.n_perms, args.seed)
    log.info("=" * 94)
    out_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    for i in range(Xe.shape[1]):
        _ablate_feature(
            model,
            Xe,
            ye,
            swe,
            rng,
            i,
            FEATURE_NAMES[i],
            args.n_perms,
            base_preds,
            loss0,
            acc0,
            base_spread,
            args.active_tol,
            args.weak_tol,
            out_rows,
        )
    out_rows.sort(key=lambda r: r["delta_loss_mean"], reverse=True)
    report = {
        "baseline": {
            "oos_rows": len(Xe),
            "weighted_bce": round(loss0, 5),
            "accuracy": round(acc0, 5),
            "p_win_spread": round(base_spread, 5),
        },
        "n_perms": args.n_perms,
        "seed": args.seed,
        "active_tol": args.active_tol,
        "weak_tol": args.weak_tol,
        "features": out_rows,
    }
    out_path = (
        Path(args.output) if args.output else Path("runtime/ablation_meta_dnn.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("=" * 94)
    log.info("Ablation complete — %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
