"""brain.meta_label_dnn_guard — ironclad anti-degeneration guard.

A degenerate gatekeeper is worse than none: a constant-output model (every
feature vector maps to P(Win) ~ base rate, e.g. 0.37) silently vetoes the
entire market while looking ordinary on disk.  Module makes that impossible.

Two hard checks — a model fails the guard unless BOTH hold:

  1. weights_healthy: every layer's weight matrix is finite and its std is
     >= MIN_WEIGHT_STD.  A real function of inputs needs non-trivial
     weights; the historical degenerate model had every weight ~1e-4.
  2. outputs_healthy: over a fixed deterministic probe set the model's
     P(Win) must actually vary (std >= MIN_OUTPUT_SPREAD).  This catches
     the worse failure mode — healthy-looking weights that learned the
     base rate as a constant.

MetaDNN enforces the guard at load time (reject the artifact and fail open
toward admittance), at train time (refuse to overwrite a good artifact)
and at snapshot time (telemetry exposes the verdict).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

__all__ = [
    "MIN_WEIGHT_STD",
    "MIN_OUTPUT_SPREAD",
    "PROBE_ROWS",
    "weights_healthy",
    "outputs_healthy",
    "probe_inputs",
    "govern",
]

log = logging.getLogger(__name__)

MIN_WEIGHT_STD: float = 0.01  # per-layer |W| std floor; below = dead layer
MIN_OUTPUT_SPREAD: float = 0.005  # probe P(Win) std floor; below = constant
PROBE_ROWS: int = 1024  # deterministic probe rows for the spread test


def weights_healthy(layers: list[Any]) -> tuple[bool, str]:
    """Every layer finite with non-trivial weight std?"""
    if not layers:
        return False, "no_layers"
    for i, (w, b) in enumerate(layers):
        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
            return False, f"layer_{i}_nonfinite"
        if float(np.std(w)) < MIN_WEIGHT_STD:
            return False, f"layer_{i}_dead_weights"
    return True, ""


def probe_inputs(input_dim: int) -> np.ndarray:
    """Deterministic diverse probe matrix spanning the feature space."""
    rng = np.random.default_rng(20260921)
    rows = PROBE_ROWS - 2 * input_dim - 3
    base = rng.normal(0.0, 1.0, (rows, input_dim))
    lo = np.full((1, input_dim), -2.0)
    hi = np.full((1, input_dim), 2.0)
    mid = np.zeros((1, input_dim))
    per_feat = np.eye(input_dim) * 2.0
    neg_per_feat = -per_feat
    return np.vstack([base, lo, hi, mid, per_feat, neg_per_feat]).astype(np.float64)


def outputs_healthy(
    predict_fn: Callable[[list[float]], float], input_dim: int
) -> tuple[bool, str]:
    """P(Win) spread over the probe set must exceed the floor."""
    try:
        preds = [float(predict_fn(row.tolist())) for row in probe_inputs(input_dim)]
    except Exception as exc:
        return False, f"probe_crash:{exc}"
    if not np.all(np.isfinite(preds)):
        return False, "probe_nonfinite"
    spread = float(np.std(preds))
    if spread < MIN_OUTPUT_SPREAD:
        return False, "constant_output"
    return True, f"spread={spread:.4f}"


def govern(
    layers: list[Any], predict_fn: Callable[[list[float]], float], input_dim: int
) -> dict[str, Any]:
    """Full guard verdict: (healthy, each check, reasons)."""
    w_ok, w_reason = weights_healthy(layers)
    o_ok, o_reason = outputs_healthy(predict_fn, input_dim)
    reasons = [r for r in (w_reason, o_reason) if r]
    return {
        "healthy": bool(w_ok and o_ok),
        "weights_healthy": bool(w_ok),
        "outputs_healthy": bool(o_ok),
        "reasons": reasons,
    }
