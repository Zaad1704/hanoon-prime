"""brain.meta_label_dnn — Deep Meta-Labeler Gatekeeper (numpy MLP).

Offline-trained MLP replacing the shallow online logistic when
``META_DNN_ENABLED`` is True.  9-dim input → 32 (ReLU) → 16 (ReLU) →
1 (sigmoid). ~1k params. R1: emits P(Win) only, never a verdict.

Feature permutation ablation showed the 8 regime/horizon one-hot slots were
constant in training (zero permutation importance) and dead-weighted ~24% of
the parameter space, so they were pruned to the 7 lived continuous signals:
[confidence, |score|, vol_pct, direction, atr_ratio, obi, vpin]. Multi-timeframe
features (tf5 trend alignment, tf15 vol expansion) are appended under the same
permission gate — they wire live ONLY after permutation ablation earns them.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from .learning_config import (
    META_DNN_BATCH,
    META_DNN_EPOCHS,
    META_DNN_FILE,
    META_DNN_HIDDEN,
    META_DNN_LR,
    META_DNN_WEIGHT_DECAY,
    META_WIN_THRESHOLD,
)

__all__ = ["MetaDNN"]
log = logging.getLogger(__name__)

INPUT_DIM: int = (
    9  # conf, |score|, vol_pct, dir, atr_ratio, obi, vpin, price_entropy, vol_entropy
)

DNN_ABSTAIN_P_WIN: float = 0.5  # neutral P(Win) reported when the DNN abstains


def expand_features(
    conf: float,
    score: float,
    vol_pct: float,
    direction: int = 1,
    atr_ratio: float = 0.0,
    obi: float = 0.0,
    vpin: float = 0.0,
    price_entropy: float = 1.0,
    vol_entropy: float = 1.0,
) -> list[float]:
    """9-dim feature vector for the DNN gatekeeper.

    Shannon entropy replaces tf5/tf15 (which proved redundant with atr_ratio
    and collapsed in training).  Entropy defaults to 1.0 (max/random) so
    cold ranges degrade gracefully before live wiring.
    """
    return [
        float(conf),
        min(1.0, abs(float(score))),
        float(vol_pct),
        float(direction),
        float(atr_ratio),
        float(obi),
        float(vpin),
        float(price_entropy),
        float(vol_entropy),
    ]


def calculate_meta_size_scale(p_win: float, threshold: float = 0.52) -> float:
    """Dynamic Kelly bet sizing: P(Win) -> allocation scalar m in [0, 1].

    Uses de Prado's continuous sizing function:
        m = max(0, (p_win - threshold) / (1 - threshold))
    At threshold (0.52) -> m = 0.0 (marginal admission, minimal allocation).
    At 0.76 -> m = 0.50 (half allocation).
    At 1.00 -> m = 1.00 (full allocation).
    """
    if p_win < threshold:
        return 0.0
    scale = (p_win - threshold) / (1.0 - threshold)
    return round(max(0.0, min(1.0, scale)), 4)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    return result


def _relu(z: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.maximum(0.0, z)
    return result


def _he_init(
    fin: int, fout: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    w: np.ndarray = rng.normal(0.0, math.sqrt(2.0 / fin), (fin, fout)).astype(
        np.float64
    )
    b: np.ndarray = np.zeros(fout)
    return w, b


def _bce_loss(sw: np.ndarray, yb: np.ndarray, out: np.ndarray) -> float:
    eps = 1e-12
    return float(
        -np.mean(sw * (yb * np.log(out + eps) + (1 - yb) * np.log(1 - out + eps)))
    )


class MetaDNN:
    """Offline-trained MLP for meta-label gating."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden: tuple[int, ...] = META_DNN_HIDDEN,
        path: Path | None = None,
    ) -> None:
        self._path = path or META_DNN_FILE
        self._input_dim = input_dim
        self._hidden = hidden
        self._rng = np.random.default_rng(42)
        self._layers: list[tuple[np.ndarray, np.ndarray]] = []
        self._built = False
        self._defective: bool = False
        self._eval_count: int = 0
        self._veto_count: int = 0
        self._last_p_win: float = 0.0
        self._last_eval_ts: float = 0.0
        self._veto_window: list[int] = []
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._p_win_history: list[float] = []
        self._feature_zero_counts: dict[str, int] = {"obi": 0, "vpin": 0}
        self._last_size_scale: float = 0.0
        self._abstain_reason: str | None = None
        self._defect_reason: str | None = None
        self._load_error: str | None = None
        self._abstain_logged_ts: float = 0.0
        self._load()
        self._refresh_model_status()

    def _build(self) -> None:
        if self._built:
            return
        dims = [self._input_dim] + list(self._hidden) + [1]
        self._layers = [
            _he_init(dims[i], dims[i + 1], self._rng) for i in range(len(dims) - 1)
        ]
        self._built = True

    def _fit_scaler(self, X: np.ndarray) -> None:
        """Compute mean/std from training data for input normalization."""
        self._scaler_mean = np.mean(X, axis=0)
        self._scaler_std = np.std(X, axis=0)
        self._scaler_std = np.where(self._scaler_std < 1e-8, 1.0, self._scaler_std)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Apply Z-score normalization using fitted scaler."""
        if self._scaler_mean is None or self._scaler_std is None:
            return X
        return cast(np.ndarray, (X - self._scaler_mean) / self._scaler_std)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Forward pass; returns (output, caches) for backprop."""
        self._build()
        cache: list[np.ndarray] = [x]
        h = x
        for i, (w, b) in enumerate(self._layers):
            z = h @ w + b
            h = _sigmoid(z) if i == len(self._layers) - 1 else _relu(z)
            cache.append(h)
        return h, cache

    def predict(self, features: list[float]) -> float:
        """P(Win) for a single feature vector."""
        if self._defective:
            return DNN_ABSTAIN_P_WIN
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if self._scaler_mean is not None:
            x = (x - self._scaler_mean) / self._scaler_std
        out, _ = self.forward(x)
        return float(out[0, 0])

    @property
    def abstain_reason(self) -> str | None:
        """Why infer() abstains (None = the model is usable)."""
        return self._abstain_reason

    def drift_zscore(self) -> float:
        """Public read of the live drift signal (OOS mean=0.364, std=0.089)."""
        return self._drift_zscore()

    def _refresh_model_status(self) -> None:
        """Recompute why infer() would abstain (None = usable model)."""
        self._abstain_reason = self._status_reason()

    def _status_reason(self) -> str | None:
        """First failing readiness check, or None when usable.

        Flat early-return chain (no elif ladder) to respect the R3
        nesting contract.
        """
        if not self._path.exists():
            return "missing_artifact"
        if self._load_error is not None:
            return self._load_error
        if self._defective:
            return self._defect_reason or "defective_artifact"
        if not self._built or not self._layers:
            return "unbuilt_artifact"
        if not self._trained_report_ok():
            return "never_trained"
        return None

    def _trained_report_ok(self) -> bool:
        """A parseable .report.json carrying a train timestamp exists."""
        report_path = self._path.with_suffix(".report.json")
        if not report_path.exists():
            return False
        try:
            return bool(json.loads(report_path.read_text()).get("train_ts"))
        except (json.JSONDecodeError, OSError, ValueError):
            return False

    def infer(self, features: list[float]) -> tuple[bool, float, float]:
        """(admit, p_win, size_scale) — admit=False when P(Win) < threshold.

        FAIL-CLOSED: a missing, corrupt, defective, or never-trained model
        BLOCKS — it returns (admit=False, p=0.5, size=0.0) instead of
        passing the trade on a fabricated score, so DNN-gated entries
        cannot fire after a restart until a human retrains. The block is
        logged at CRITICAL (throttled to one line per 5 minutes) so a
        silent gatekeeper is impossible.
        """
        import time

        if self._abstain_reason is not None:
            # FAIL-CLOSED: missing, corrupt, defective, or never-trained
            # model -> block. The gatekeeper returns
            # (admit=False, p=0.5, size=0.0): DNN-gated entries cannot
            # fire until a human retrains. Blocks are NOT predictions,
            # so they stay out of the drift history.
            now = time.time()
            if now - self._abstain_logged_ts > 300.0:
                self._abstain_logged_ts = now
                log.critical(
                    "DNN BLOCK (%s): no usable model — DNN-gated entries "
                    "blocked until a human retrains",
                    self._abstain_reason or "unknown",
                )
            self._eval_count += 1
            self._last_p_win = DNN_ABSTAIN_P_WIN
            self._last_eval_ts = now
            self._last_size_scale = 0.0
            obi_val = float(features[5]) if len(features) > 5 else 0.0
            vpin_val = float(features[6]) if len(features) > 6 else 0.0
            self._feature_zero_counts["obi"] = (
                self._feature_zero_counts["obi"] + 1 if abs(obi_val) < 1e-12 else 0
            )
            self._feature_zero_counts["vpin"] = (
                self._feature_zero_counts["vpin"] + 1 if abs(vpin_val) < 1e-12 else 0
            )
            self._veto_window.append(0)
            if len(self._veto_window) > 100:
                self._veto_window = self._veto_window[-100:]
            return False, DNN_ABSTAIN_P_WIN, 0.0

        try:
            p = self.predict(features)
        except Exception as exc:
            # FAIL-CLOSED: a throwing model is not a usable model.
            log.critical("DNN inference failed (%s) — blocking", exc)
            self._abstain_reason = "inference_error"
            return False, DNN_ABSTAIN_P_WIN, 0.0
        admit = p >= META_WIN_THRESHOLD
        scale = calculate_meta_size_scale(p)
        self._eval_count += 1
        if not admit:
            self._veto_count += 1
        self._last_p_win = p
        self._last_eval_ts = time.time()
        self._last_size_scale = scale
        self._p_win_history.append(p)
        if len(self._p_win_history) > 50:
            self._p_win_history = self._p_win_history[-50:]
        obi_val = float(features[5]) if len(features) > 5 else 0.0
        vpin_val = float(features[6]) if len(features) > 6 else 0.0
        self._feature_zero_counts["obi"] = (
            self._feature_zero_counts["obi"] + 1 if abs(obi_val) < 1e-12 else 0
        )
        self._feature_zero_counts["vpin"] = (
            self._feature_zero_counts["vpin"] + 1 if abs(vpin_val) < 1e-12 else 0
        )
        self._veto_window.append(0 if admit else 1)
        if len(self._veto_window) > 100:
            self._veto_window = self._veto_window[-100:]
        return admit, p, calculate_meta_size_scale(p)

    def guard_status(self) -> dict[str, Any]:
        """Ironclad guard verdict for telemetry."""
        from . import meta_label_dnn_guard as _guard

        if self._defective or not self._built:
            return {
                "healthy": not self._defective,
                "active": False,
                "bypassed": True,
                "reasons": ["defective"] if self._defective else ["unbuilt"],
            }
        verdict = _guard.govern(self._layers, self.predict, self._input_dim)
        verdict["active"] = True
        verdict["bypassed"] = False
        return verdict

    def _drift_zscore(self) -> float:
        """Z-score of live P(Win) mean vs OOS training baseline.

        OOS baseline: mu=0.364, std=0.089 (from purged 5-fold CV).
        |Z| > 2.5 signals distribution shift / regime drift.
        """
        OOS_MEAN = 0.364
        OOS_STD = 0.089
        n = len(self._p_win_history)
        if n < 5 or OOS_STD < 1e-12:
            return 0.0
        live_mean = sum(self._p_win_history) / n
        return round((live_mean - OOS_MEAN) / OOS_STD, 4)

    def live_snapshot(self) -> dict[str, Any]:
        """Telemetry view of DNN gatekeeper live state."""
        import time

        report: dict[str, Any] = {}
        report_path = self._path.with_suffix(".report.json")
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                log.debug("DNN report load failed: %s", exc)
        window = self._veto_window[-100:] if self._veto_window else []
        return {
            "enabled": META_WIN_THRESHOLD > 0,
            "gate_active": self._built and self._layers and not self._defective,
            "defective": self._defective,
            "guard": self.guard_status(),
            "threshold": META_WIN_THRESHOLD,
            "oos_accuracy": report.get("accuracy"),
            "eval_count": self._eval_count,
            "veto_count": self._veto_count,
            "veto_rate_20": (
                round(sum(window[-20:]) / min(20, len(window)), 4) if window else 0.0
            ),
            "veto_rate_100": round(sum(window) / len(window), 4) if window else 0.0,
            "weights_loaded": self._built and len(self._layers) > 0,
            "param_count": (
                sum(w.size + b.size for w, b in self._layers) if self._layers else 0
            ),
            "last_p_win": round(self._last_p_win, 4),
            "abstain_reason": self._abstain_reason,
            "last_eval": self._last_eval_ts,
            "train_entries": report.get("total_entries"),
            "train_tickers": None,
            "train_accuracy": report.get("accuracy"),
            "cv_folds": report.get("folds"),
            "train_epochs": META_DNN_EPOCHS,
            "last_train_ts": report.get("train_ts"),
            "drift_zscore": self._drift_zscore(),
            "feature_zero_streak": dict(self._feature_zero_counts),
            "last_size_scale": round(self._last_size_scale, 4),
        }

    def _write_training_report(self, n_entries: int) -> None:
        """Write the .report.json sidecar the readiness gate requires."""
        import time

        try:
            self._path.with_suffix(".report.json").write_text(
                json.dumps(
                    {
                        "train_ts": time.time(),
                        "total_entries": int(n_entries),
                        "verdict": "healthy",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("DNN report write failed: %s", exc)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        epochs: int = META_DNN_EPOCHS,
        lr: float = META_DNN_LR,
        batch_size: int = META_DNN_BATCH,
        weight_decay: float = META_DNN_WEIGHT_DECAY,
    ) -> list[float]:
        """Mini-batch Adam training loop. Returns per-epoch BCE losses."""
        # A retrain IS the recovery path from a defective artifact: clear the
        # flags so the post-train guard probe evaluates real forward passes
        # (predict() short-circuits to a constant while _defective is set,
        # which would brand every freshly-trained model constant_output).
        self._defective = False
        self._defect_reason = None
        self._load_error = None
        self._build()
        self._fit_scaler(X)
        Xs = self._transform(X)
        rng = np.random.default_rng(0)
        n = Xs.shape[0]
        m = [np.zeros_like(w) for w, _ in self._layers]
        v = [np.zeros_like(w) for w, _ in self._layers]
        bm = [np.zeros_like(b) for _, b in self._layers]
        bv = [np.zeros_like(b) for _, b in self._layers]
        t = 0
        losses = []
        for _ in range(epochs):
            idx = rng.permutation(n)
            e_loss, n_b = 0.0, 0
            for s in range(0, n, batch_size):
                xb = Xs[idx[s : s + batch_size]]
                bs = xb.shape[0]
                yb = y[idx[s : s + batch_size]].reshape(-1, 1)
                xb_sw = (
                    sample_weight[idx[s : s + batch_size]].reshape(-1, 1)
                    if sample_weight is not None
                    else np.ones_like(yb)
                )
                out, cache = self.forward(xb)
                e_loss += _bce_loss(xb_sw, yb, out)
                n_b += 1
                t += 1
                self._adam_step(
                    (out - yb) / bs, cache, weight_decay, m, v, bm, bv, t, lr
                )
            losses.append(e_loss / max(n_b, 1))
        from . import meta_label_dnn_guard as _guard

        verdict = _guard.govern(self._layers, self.predict, self._input_dim)
        if verdict["healthy"]:
            self._save()
            self._write_training_report(n)
        else:
            self._defective = True
            self._defect_reason = "guard_rejected"
            log.critical(
                "DNN training collapsed (%s) — good artifact kept, gatekeeper abstained",
                verdict["reasons"],
            )
        self._refresh_model_status()
        return losses

    def _adam_step(
        self,
        g: np.ndarray,
        cache: list[np.ndarray],
        wd: float,
        m: list[np.ndarray],
        v: list[np.ndarray],
        bm: list[np.ndarray],
        bv: list[np.ndarray],
        t: int,
        lr: float,
    ) -> None:
        b1, b2, eps = 0.9, 0.999, 1e-8
        grad_norm = float(np.sqrt(sum(np.sum(x**2) for x in [g])))
        clip = 5.0
        if grad_norm > clip:
            g = g * (clip / grad_norm)
        for i in range(len(self._layers) - 1, -1, -1):
            dw = cache[i].T @ g + wd * self._layers[i][0]
            db = np.mean(g, axis=0)
            dw = np.clip(dw, -clip, clip)
            db = np.clip(db, -clip, clip)
            m[i] = b1 * m[i] + (1 - b1) * dw
            v[i] = b2 * v[i] + (1 - b2) * dw**2
            bm[i] = b1 * bm[i] + (1 - b1) * db
            bv[i] = b2 * bv[i] + (1 - b2) * db**2
            mh, vh = m[i] / (1 - b1**t), v[i] / (1 - b2**t)
            bmh, bvh = bm[i] / (1 - b1**t), bv[i] / (1 - b2**t)
            self._layers[i] = (
                self._layers[i][0] - lr * mh / (np.sqrt(vh) + eps),
                self._layers[i][1] - lr * bmh / (np.sqrt(bvh) + eps),
            )
            if i > 0:
                g = (g @ self._layers[i][0].T) * (cache[i] > 0).astype(np.float64)

    def _save(self) -> None:
        self._build()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "hidden": list(self._hidden),
                "input_dim": self._input_dim,
                "weights": [
                    {"w": w.tolist(), "b": b.tolist()} for w, b in self._layers
                ],
            }
            if self._scaler_mean is not None and self._scaler_std is not None:
                data["scaler_mean"] = self._scaler_mean.tolist()
                data["scaler_std"] = self._scaler_std.tolist()
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(self._path)
        except OSError as exc:
            log.debug("DNN save failed: %s", exc)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text())
            self._hidden = tuple(d.get("hidden", self._hidden))
            stored_dim = int(d.get("input_dim", self._input_dim))
            if stored_dim != self._input_dim:
                self._defective = True
                self._defect_reason = "dimension_mismatch"
                log.critical(
                    "DNN artifact dimension mismatch (code=%d artifact=%d) — gatekeeper abstained",
                    self._input_dim,
                    stored_dim,
                )
                return
            self._layers = [
                (np.array(l["w"], dtype=np.float64), np.array(l["b"], dtype=np.float64))
                for l in d.get("weights", [])
            ]
            if "scaler_mean" in d and "scaler_std" in d:
                self._scaler_mean = np.array(d["scaler_mean"], dtype=np.float64)
                self._scaler_std = np.array(d["scaler_std"], dtype=np.float64)
            if self._layers:
                self._built = True
            if self._built:
                from . import meta_label_dnn_guard as _guard

                verdict = _guard.govern(self._layers, self.predict, self._input_dim)
                if not verdict["healthy"]:
                    self._defective = True
                    self._defect_reason = "guard_rejected"
                    log.critical(
                        "DNN artifact rejected by ironclad guard (%s) — gatekeeper abstained",
                        verdict["reasons"],
                    )
                    self._layers = []
                    self._built = False
                    self._scaler_mean = None
                    self._scaler_std = None
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, OSError) as exc:
            self._load_error = "corrupt_artifact"
            log.debug("DNN load failed (fresh init): %s", exc)
