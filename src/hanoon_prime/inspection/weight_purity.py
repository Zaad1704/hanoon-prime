"""inspection.weight_purity — learned-weight budget integrity checks.

Guards the cortex's scoring budget. The live `range` regime vector once
drifted to signed_sum ≈ -3.14 (|sum| ≈ 4.16, per-weight values at
-0.55/-0.79); through the old signed-sum guard that zeroed the cortex
score, so every candidate fell to the residual modifiers and read as a
SHORT that direction_mode then rejected. These checks keep the manifest
always aware of a collapsed or corrupt weight budget — both on disk and
as the effective budget the cortex will actually score with.
"""

from __future__ import annotations

from .checks import FAIL, OK, WARN, CheckResult
from .ctx import InspectionContext
from .probe import juli_state, regime_weights


def _min_trades() -> int:
    from hanoon_prime.brain.learning_config import REGIME_MIN_TRADES

    return REGIME_MIN_TRADES


def _active_vectors(ctx: InspectionContext) -> dict[str, dict[str, float]]:
    """Regime vectors whose own regime has passed the thin-data gate."""
    rw = regime_weights(ctx)
    vectors = rw.get("vectors")
    counts = rw.get("counts")
    if not isinstance(vectors, dict):
        return {}
    counts = counts if isinstance(counts, dict) else {}
    min_trades = _min_trades()
    active: dict[str, dict[str, float]] = {}
    for regime, vec in vectors.items():
        n = counts.get(regime, 0)
        if (
            isinstance(regime, str)
            and isinstance(vec, dict)
            and isinstance(n, int)
            and n >= min_trades
        ):
            active[regime] = vec
    return active


def _abs_signed(w: dict[str, float]) -> tuple[float, float]:
    """(Σ|w|, Σw) over the numeric entries of a weight dict."""
    vals = [v for v in w.values() if isinstance(v, (int, float))]
    return sum(abs(v) for v in vals), sum(v for v in vals)


def regime_weights_bounded(ctx: InspectionContext) -> CheckResult:
    """Active regime vectors stay inside the weight-enforcer envelope.

    A drifted vector back-fills the cortex with a corrupted budget and
    previously zeroed the score via the signed-sum guard. This catches it
    while it is still on disk, before the cortex consumes it.
    """
    from hanoon_prime.brain.weight_enforcer import get_enforcer

    rw = regime_weights(ctx)
    if not isinstance(rw.get("vectors"), dict):
        return CheckResult(
            "purity", "regime_weights_bounded", FAIL, detail="vectors missing/untyped"
        )
    bad: list[str] = []
    for regime, vec in sorted(_active_vectors(ctx).items()):
        integrity = get_enforcer().check_integrity(vec)
        if not integrity["ok"]:
            bad.append(f"{regime}: " + "; ".join(integrity["issues"]))
    if bad:
        return CheckResult(
            "purity",
            "regime_weights_bounded",
            FAIL,
            detail="active regime vector out of band",
            evidence={"vectors": bad},
        )
    return CheckResult(
        "purity",
        "regime_weights_bounded",
        OK,
        evidence={"active": sorted(_active_vectors(ctx))},
    )


def _effective_budgets(ctx: InspectionContext) -> dict[str, dict[str, float]] | None:
    """What the orchestrator hands the cortex: defaults + memory + regime."""
    from hanoon_prime.brain.config import DEFAULT_WEIGHTS

    memory_w = juli_state(ctx).get("weights")
    if not isinstance(memory_w, dict):
        return None
    base = dict(DEFAULT_WEIGHTS)
    base.update(
        {k: float(v) for k, v in memory_w.items() if isinstance(v, (int, float))}
    )
    budgets: dict[str, dict[str, float]] = {"global": base}
    for regime, vec in _active_vectors(ctx).items():
        eff = dict(base)
        eff.update({k: float(v) for k, v in vec.items() if isinstance(v, (int, float))})
        budgets[regime] = eff
    return budgets


def cortex_score_degenerate(ctx: InspectionContext) -> CheckResult:
    """The effective cortex weight budget can actually form a signal.

    A collapsed absolute budget pins the score at 0.0 → the brain can never
    emit a legal verdict. A non-positive signed sum is the exact corruption
    that forced every candidate SHORT via the old signed guard — legal today
    (the cortex divides by Σ|w|) but worth surfacing.
    """
    budgets = _effective_budgets(ctx)
    if budgets is None:
        return CheckResult(
            "purity", "cortex_score_degenerate", FAIL, detail="memory weights missing"
        )
    sums = {n: _abs_signed(w) for n, w in budgets.items()}
    collapsed = [n for n, (a, _) in sums.items() if a < 0.1]
    if collapsed:
        return CheckResult(
            "purity",
            "cortex_score_degenerate",
            FAIL,
            detail="cortex weight budget collapsed (score pinned at 0.0)",
            evidence={"budgets": sums},
        )
    neg = [f"{n}(signed={s:.3f})" for n, (_, s) in sums.items() if s <= 1e-12]
    if neg:
        return CheckResult(
            "purity",
            "cortex_score_degenerate",
            WARN,
            detail="non-positive signed weight sum; scoring works via Σ|w|, monitor polarity",
            evidence={"budgets": neg},
        )
    return CheckResult(
        "purity",
        "cortex_score_degenerate",
        OK,
        evidence={"abs_sums": {n: round(a, 4) for n, (a, _) in sums.items()}},
    )
